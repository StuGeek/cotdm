import threading
from scheduler import get_single_model_placement
from init.init_deploy_cotdm import deploy_bert_model_by_load_index, deploy_gpt2_model_by_load_index
from global_data import global_config, global_var


def deploy_single_model_totoff(model_info, stage_list_cpu):
    memory_fully_loaded_list = global_var.memory_fully_loaded_list

    stage_list = []

    total_layer_num = model_info["total_layer_num"]
    model_type = model_info["type"]

    model_stage_cond_list = [
        threading.Condition(threading.Lock()) for i in range(total_layer_num)
    ]

    load_index = [0, total_layer_num]

    min_mem_device_idx = 0
    for i in range(1, len(memory_fully_loaded_list)):
        if memory_fully_loaded_list[i] < memory_fully_loaded_list[min_mem_device_idx]:
            min_mem_device_idx = i
    model_use_gpu_list = [global_var.cuda_devices[min_mem_device_idx]]
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
