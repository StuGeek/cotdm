import threading
import torch
from min_deploy import min_deploy, cal_lat_by_deploynum_and_avaigpu
from scheduler import get_single_model_placement
from global_data.global_class import ModelStage, SingleLayerTime
from global_data import global_config, global_var

default_use_gpu_num = len(global_var.cuda_devices)


def get_half_mem_idx(stage_mem_list):
    half_mem_idx = 0
    half_total_stage_mem = sum(stage_mem_list) / 2
    stage_mem_accu_sum = 0
    for stage_mem in stage_mem_list:
        stage_mem_accu_sum += stage_mem
        if stage_mem_accu_sum <= half_total_stage_mem:
            half_mem_idx += 1
        else:
            break

    return half_mem_idx


def deploy_bert_model_by_load_index(stage_list_cpu, load_index, deploy_device):
    deploy_params_num = load_index[0]
    total_layer_num = len(stage_list_cpu)
    stage_list = []

    before_memory, after_memory = 0, 0
    for i in range(len(global_var.cuda_devices)):
        before_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])

    is_stage_loaded = [False for i in range(total_layer_num)]
    if deploy_params_num > 0:
        stages_gpu = []
        for i in range(len(stage_list_cpu[0].stage)):
            stage_gpu = stage_list_cpu[0].stage[i].to(deploy_device)
            if i == 0:
                stage_gpu.eval()
            stages_gpu.append(stage_gpu)
        stage_list.append(ModelStage(stages_gpu, stage_list_cpu[0].type))
        is_stage_loaded[0] = True
    for i in range(1, deploy_params_num):
        stages_gpu = []
        for j in range(len(stage_list_cpu[i].stage)):
            stage_gpu = stage_list_cpu[i].stage[j].to(deploy_device)
            if i < total_layer_num - 1:
                stage_gpu.eval()
            stages_gpu.append(stage_gpu)
        stage_list.append(ModelStage(stages_gpu, stage_list_cpu[i].type))
        is_stage_loaded[i] = True

    for i in range(deploy_params_num, total_layer_num):
        stage_list.append(ModelStage(stage_list_cpu[i].stage, stage_list_cpu[i].type))

    for i in range(len(global_var.cuda_devices)):
        after_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])
    deploy_memory = after_memory - before_memory

    return stage_list, is_stage_loaded, deploy_memory


def deploy_gpt2_model_by_load_index(stage_list_cpu, load_index, deploy_device):
    deploy_params_num = load_index[0]
    total_layer_num = len(stage_list_cpu)
    stage_list = []

    before_memory, after_memory = 0, 0
    for i in range(len(global_var.cuda_devices)):
        before_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])

    is_stage_loaded = [False for i in range(total_layer_num)]
    for i in range(deploy_params_num):
        stages_gpu = []
        for j in range(len(stage_list_cpu[i].stage)):
            stage_gpu = stage_list_cpu[i].stage[j].to(deploy_device)
            stage_gpu.eval()
            stages_gpu.append(stage_gpu)
        stage_list.append(ModelStage(stages_gpu, stage_list_cpu[i].type))
        is_stage_loaded[i] = True

    for i in range(deploy_params_num, total_layer_num):
        stage_list.append(ModelStage(stage_list_cpu[i].stage, stage_list_cpu[i].type))

    for i in range(len(global_var.cuda_devices)):
        after_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])
    deploy_memory = after_memory - before_memory

    return stage_list, is_stage_loaded, deploy_memory


def deploy_single_model_cotdm(model_info, stage_list_cpu):
    stage_list = []

    total_layer_num = model_info["total_layer_num"]
    model_type = model_info["type"]
    avg_e2e_lat_ready = model_info["avg_e2e_lat_ready"]
    use_gpu_num = default_use_gpu_num
    single_layer_cal_time = model_info["single_layer_time"]["cal_time"]
    single_layer_trans_time = model_info["single_layer_time"]["trans_time"]
    single_layer_load_time = model_info["single_layer_time"]["load_time"]
    single_layer_time = SingleLayerTime(
        single_layer_cal_time, single_layer_trans_time, single_layer_load_time
    )
    stage_mem_list = model_info["stage_mem_list"]

    model_stage_cond_list = [
        threading.Condition(threading.Lock()) for i in range(total_layer_num)
    ]

    slo = round(avg_e2e_lat_ready * global_config.INIT_SLO_SCALE, 3)
    if global_config.DEPLOY_ALG == "Min Deploy":
        load_index = min_deploy(total_layer_num, slo, use_gpu_num, single_layer_time)
    elif global_config.DEPLOY_ALG == "Half Mem":
        half_mem_idx = get_half_mem_idx(stage_mem_list)
        _, load_index = cal_lat_by_deploynum_and_avaigpu(
            half_mem_idx, total_layer_num, use_gpu_num, single_layer_time
        )

    model_use_gpu_list = get_single_model_placement(stage_mem_list, load_index)
    deploy_device = model_use_gpu_list[0]

    if model_type == "bert":
        stage_list, is_stage_loaded, deploy_memory = deploy_bert_model_by_load_index(
            stage_list_cpu, load_index, deploy_device
        )
    elif model_type == "gpt2":
        stage_list, is_stage_loaded, deploy_memory = deploy_gpt2_model_by_load_index(
            stage_list_cpu, load_index, deploy_device
        )
    else:
        raise ValueError("Model type {} is not supported".format(model_type))

    return (
        stage_list,
        load_index,
        is_stage_loaded,
        model_stage_cond_list,
        model_use_gpu_list,
        deploy_memory / float(1e6),
    )
