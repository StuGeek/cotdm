import threading
from scheduler import get_single_model_placement
from init.init_deploy_cotdm import deploy_bert_model_by_load_index, deploy_gpt2_model_by_load_index


def deploy_single_model_ready(model_info, stage_list_cpu):
    stage_list = []

    total_layer_num = model_info["total_layer_num"]
    model_type = model_info["type"]
    stage_mem_list = model_info["stage_mem_list"]

    model_stage_cond_list = [
        threading.Condition(threading.Lock()) for i in range(total_layer_num)
    ]

    load_index = [total_layer_num, total_layer_num]

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
