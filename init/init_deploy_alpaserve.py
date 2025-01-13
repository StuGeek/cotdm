import threading
import torch
from scheduler import get_single_model_placement_alpaserve
from global_data.global_class import ModelStage, SingleLayerTime
from global_data import global_var
from init import init_deploy_cotdm


def deploy_bert_model_by_load_index_alpaserve(stage_list_cpu, load_index, model_use_gpu_list, is_over_limited_mem):
    total_layer_num = len(stage_list_cpu)
    stage_list = []
    is_stage_loaded = [False for i in range(total_layer_num)]

    if is_over_limited_mem:
        for i in range(total_layer_num):
            stage_list.append(
                ModelStage(stage_list_cpu[i].stage, stage_list_cpu[i].type)
            )
        return stage_list, is_stage_loaded, 0
    
    before_memory, after_memory = 0, 0
    for i in range(len(global_var.cuda_devices)):
        before_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])
    
    use_gpu_num = len(model_use_gpu_list)
    for i in range(use_gpu_num):
        deploy_device = model_use_gpu_list[i]

        begin_idx = load_index[i]
        end_idx = load_index[i + 1]
        if i == use_gpu_num - 1:
            end_idx -= 1

        for j in range(begin_idx, end_idx):
            stages_gpu = []
            for k in range(len(stage_list_cpu[j].stage)):
                stage_gpu = stage_list_cpu[j].stage[k].to(deploy_device)
                if k == 0 and j != total_layer_num - 1:
                    stage_gpu.eval()
                stages_gpu.append(stage_gpu)
            stage_list.append(ModelStage(stages_gpu, stage_list_cpu[j].type))
            is_stage_loaded[j] = True

    stages_gpu = []
    for i in range(len(stage_list_cpu[-1].stage)):
        stage_gpu = stage_list_cpu[-1].stage[i].to(model_use_gpu_list[0])
        stages_gpu.append(stage_gpu)
    stage_list.append(ModelStage(stages_gpu, stage_list_cpu[-1].type))
    is_stage_loaded[-1] = True
    
    for i in range(len(global_var.cuda_devices)):
        after_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])
    deploy_memory = after_memory - before_memory

    return stage_list, is_stage_loaded, deploy_memory


def deploy_gpt2_model_by_load_index_alpaserve(stage_list_cpu, load_index, model_use_gpu_list, is_over_limited_mem):
    total_layer_num = len(stage_list_cpu)
    stage_list = []
    is_stage_loaded = [False for i in range(total_layer_num)]

    if is_over_limited_mem:
        for i in range(total_layer_num):
            stage_list.append(
                ModelStage(stage_list_cpu[i].stage, stage_list_cpu[i].type)
            )
        return stage_list, is_stage_loaded, 0
    
    before_memory, after_memory = 0, 0
    for i in range(len(global_var.cuda_devices)):
        before_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])
    
    for i in range(len(model_use_gpu_list)):
        deploy_device = model_use_gpu_list[i]

        begin_idx = load_index[i]
        end_idx = load_index[i + 1]
        for j in range(begin_idx, end_idx):
            stages_gpu = []
            for k in range(len(stage_list_cpu[j].stage)):
                stage_gpu = stage_list_cpu[j].stage[k].to(deploy_device)
                stage_gpu.eval()
                stages_gpu.append(stage_gpu)
            stage_list.append(ModelStage(stages_gpu, stage_list_cpu[j].type))
            is_stage_loaded[j] = True

    # stages_gpu = []
    # for i in range(len(stage_list_cpu[-1].stage)):
    #     stage_gpu = stage_list_cpu[-1].stage[i].to(model_use_gpu_list[0])
    #     stages_gpu.append(stage_gpu)
    # stage_list.append(ModelStage(stages_gpu, stage_list_cpu[-1].type))
    # is_stage_loaded[-1] = True
    
    for i in range(len(global_var.cuda_devices)):
        after_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])
    deploy_memory = after_memory - before_memory

    return stage_list, is_stage_loaded, deploy_memory


def deploy_single_model_alpaserve(model_info, stage_list_cpu, is_over_limited_mem=False):
    stage_list = []

    total_layer_num = model_info["total_layer_num"]
    model_type = model_info["type"]
    use_gpu_num = init_deploy_cotdm.default_use_gpu_num
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

    load_index = []
    for i in range(use_gpu_num + 1):
        load_index.append(int(total_layer_num * i / use_gpu_num))
    use_gpu_num = len(load_index) - 1

    model_use_gpu_list = get_single_model_placement_alpaserve(stage_mem_list, load_index)

    if model_type == "bert":
        stage_list, is_stage_loaded, deploy_memory = deploy_bert_model_by_load_index_alpaserve(stage_list_cpu, load_index, model_use_gpu_list, is_over_limited_mem)
    elif model_type == "gpt2":
        stage_list, is_stage_loaded, deploy_memory = deploy_gpt2_model_by_load_index_alpaserve(stage_list_cpu, load_index, model_use_gpu_list, is_over_limited_mem)
    else:
        raise ValueError("Model type {} is not supported".format(model_type))

    if not is_over_limited_mem:
        load_index[0] = total_layer_num

    return (
        stage_list,
        load_index,
        is_stage_loaded,
        model_stage_cond_list,
        model_use_gpu_list,
        deploy_memory / float(1e6),
    )
