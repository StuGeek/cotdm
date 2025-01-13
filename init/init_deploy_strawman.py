import threading
from min_deploy import min_deploy
from scheduler import get_single_model_placement
from global_data.global_class import SingleLayerTime
from global_data import global_config
from init.init_deploy_cotdm import deploy_bert_model_by_load_index, deploy_gpt2_model_by_load_index, get_half_mem_idx
from init import init_deploy_cotdm

def deploy_single_model_strawman(model_info, stage_list_cpu):
    stage_list = []

    total_layer_num = model_info["total_layer_num"]
    model_type = model_info["type"]
    avg_e2e_lat_ready = model_info["avg_e2e_lat_ready"]
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
        load_index = min_deploy(total_layer_num, slo, init_deploy_cotdm.default_use_gpu_num, single_layer_time)
    elif global_config.DEPLOY_ALG == "Half Mem":
        half_mem_idx = get_half_mem_idx(stage_mem_list)
        load_index = [half_mem_idx, total_layer_num]

    model_use_gpu_list = get_single_model_placement(stage_mem_list, load_index)
    deploy_device = model_use_gpu_list[0]

    if model_type == "bert":
        stage_list, is_stage_loaded, deploy_memory = deploy_bert_model_by_load_index(stage_list_cpu, load_index, deploy_device)
    elif model_type == "gpt2":
        stage_list, is_stage_loaded, deploy_memory = deploy_gpt2_model_by_load_index(stage_list_cpu, load_index, deploy_device)
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
