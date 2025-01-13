import torch
from global_data import global_config, global_var
from min_deploy import min_deploy, cal_lat_by_deploynum_and_avaigpu
from threading import Thread
import pynvml

pynvml.nvmlInit()


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


def get_single_model_placement(stage_mem_list, load_index, deploy_device=None):
    load_index_len = len(load_index)
    use_gpu_num = load_index_len - 1

    if use_gpu_num == 1 and deploy_device is not None:
        return [deploy_device]

    devices_avai_mem_list = []
    device_sort_index_list = []
    group_sort_list = []

    use_mem_in_each_device = [0 for i in range(use_gpu_num)]
    use_mem_in_each_device[0] = sum(stage_mem_list[0:load_index[0]])
    for i in range(1, load_index_len):
        begin_idx = load_index[i - 1]
        end_idx = load_index[i]
        use_mem_in_each_device[i - 1] += sum(stage_mem_list[begin_idx:end_idx])

    cuda_devices_len = len(global_var.cuda_devices)
    for gpu_id in range(cuda_devices_len):
        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)

        avai_mem = meminfo.free
        sort_index = 0
        for i in range(len(device_sort_index_list)):
            if avai_mem <= devices_avai_mem_list[i]:
                sort_index += 1
            else:
                device_sort_index_list[i] += 1

        devices_avai_mem_list.append(avai_mem)
        device_sort_index_list.append(sort_index)

    device_sort_list = [None for i in range(len(device_sort_index_list))]
    for i in range(len(device_sort_index_list)):
        device_sort_list[device_sort_index_list[i]] = global_var.cuda_devices[i]

    for i in range(len(use_mem_in_each_device)):
        used_mem = use_mem_in_each_device[i]
        sort_index = 0
        for j in range(i):
            if used_mem <= use_mem_in_each_device[j]:
                sort_index += 1
            else:
                group_sort_list[j] += 1
        group_sort_list.append(sort_index)

    model_use_gpu_list = []
    if deploy_device is not None:
        model_use_gpu_list.append(deploy_device)
    for i in group_sort_list:
        if device_sort_list[i] != deploy_device:
            model_use_gpu_list.append(device_sort_list[i])
    # model_use_gpu_list = [device_sort_list[i] for i in group_sort_list]
    return model_use_gpu_list 


def get_single_model_placement_alpaserve(stage_mem_list, load_index, deploy_device=None):
    memory_fully_loaded_list = global_var.memory_fully_loaded_list
    
    load_index_len = len(load_index)
    use_gpu_num = load_index_len - 1

    if use_gpu_num == 1 and deploy_device is not None:
        return [deploy_device]

    devices_avai_mem_list = []
    device_sort_index_list = []
    group_sort_list = []

    use_mem_in_each_device = [0 for i in range(use_gpu_num)]
    use_mem_in_each_device[0] = sum(stage_mem_list[0:load_index[0]])
    for i in range(1, load_index_len):
        begin_idx = load_index[i - 1]
        end_idx = load_index[i]
        use_mem_in_each_device[i - 1] += sum(stage_mem_list[begin_idx:end_idx])

    cuda_devices_len = len(global_var.cuda_devices)
    for gpu_id in range(cuda_devices_len):
        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)

        avai_mem = meminfo.total - memory_fully_loaded_list[gpu_id]
        sort_index = 0
        for i in range(len(device_sort_index_list)):
            if avai_mem <= devices_avai_mem_list[i]:
                sort_index += 1
            else:
                device_sort_index_list[i] += 1

        devices_avai_mem_list.append(avai_mem)
        device_sort_index_list.append(sort_index)

    device_sort_list = [None for i in range(len(device_sort_index_list))]
    for i in range(len(device_sort_index_list)):
        device_sort_list[device_sort_index_list[i]] = global_var.cuda_devices[i]

    for i in range(len(use_mem_in_each_device)):
        used_mem = use_mem_in_each_device[i]
        sort_index = 0
        for j in range(i):
            if used_mem <= use_mem_in_each_device[j]:
                sort_index += 1
            else:
                group_sort_list[j] += 1
        group_sort_list.append(sort_index)

    model_use_gpu_list = []
    if deploy_device is not None:
        model_use_gpu_list.append(deploy_device)
    for i in group_sort_list:
        if device_sort_list[i] != deploy_device:
            model_use_gpu_list.append(device_sort_list[i])
    return model_use_gpu_list


def readjust_func(model_idx, use_gpu_num):
    import time
    start_time = time.time()

    global_var.is_model_using_lists[model_idx].clear()

    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    stage_mem_list = global_var.model_stage_mem_list[model_idx]
    deploy_device = use_gpu_list[0]
    slo = global_var.model_slo_list[model_idx]
    single_layer_time = global_var.single_layer_time_list[model_idx]
    load_index = global_var.load_index_lists[model_idx]
    model_type = global_var.config_lists[model_idx].model_type

    # deploy_scheme = global_config.DEPLOY_SCHEME
    # deploy_scheme_idx = global_config.DEPLOY_SCHEME_DICT[deploy_scheme]
    # deploy_memory_list = global_var.deploy_memory_list[deploy_scheme_idx]

    total_layer_num = len(stage_mem_list)
    if global_config.DEBUG_MODE_SCHEDULER:
        print("min deploy bedin", model_idx)
    if use_gpu_num <= 2 or global_config.DEPLOY_ALG == "Min Deploy":
        new_load_index = min_deploy(total_layer_num, slo / global_config.INIT_SLO_SCALE, use_gpu_num, single_layer_time)
        new_load_index = [total_layer_num, total_layer_num]
    elif global_config.DEPLOY_ALG == "Half Mem":
        half_mem_idx = get_half_mem_idx(stage_mem_list)
        _, new_load_index = cal_lat_by_deploynum_and_avaigpu(half_mem_idx, total_layer_num, use_gpu_num, single_layer_time)
    if global_config.DEBUG_MODE_SCHEDULER:
        print("min deploy end", model_idx)
    new_use_gpu_list = get_single_model_placement(stage_mem_list, new_load_index, deploy_device)
    
    # print("global req num, wait offload num:", model_idx, global_var.cur_request_num.get(), global_var.cur_wait_offload_num.get())
    # print("use gpu num:", model_idx, use_gpu_num, load_index, new_load_index)
    # print("old load index:", load_index, "new load index:", new_load_index)
    # print("old use gpu list:", use_gpu_list, "new use gpu list:", new_use_gpu_list, model_idx)

    deploy_num = load_index[0]
    new_deploy_num = new_load_index[0]
    if deploy_num == new_deploy_num:
        # print("dengyu:", new_load_index, new_use_gpu_list)
        global_var.is_model_using_lists[model_idx].set()
        return
    elif deploy_num < new_deploy_num:
        for i in range(deploy_num, new_deploy_num):
            with stage_cond_list[i]:
                for j in range(len(stage_list_gpu[i].stage)):
                    stage_list_gpu[i].stage[j] = (
                        stage_list_gpu[i]
                        .stage[j]
                        .to(deploy_device, non_blocking=True)
                    )
                    if model_type == "bert" and ((i == 0 and j > 0) or (i == total_layer_num - 1)):
                        continue
                    stage_list_gpu[i].stage[j].eval()

                is_stage_loaded[i] = True
                stage_cond_list[i].notify()
    else:
        for i in range(new_deploy_num, deploy_num):
            with stage_cond_list[i]:
                is_stage_loaded[i] = False

                if stage_list_gpu[i] != None and stage_list_gpu[i].stage != None:
                    for j in range(len(stage_list_gpu[i].stage)):
                        stage_list_gpu[i].stage[j] = (
                            stage_list_gpu[i].stage[j].cpu()
                        )

            torch.cuda.empty_cache()

    global_var.load_index_lists[model_idx] = new_load_index
    global_var.model_use_gpu_lists[model_idx] = new_use_gpu_list
    # deploy_memory_list[model_idx] = sum(stage_mem_list[0:new_deploy_num])

    global_var.is_model_using_lists[model_idx].set()

    # print("readjust time:", time.time() - start_time)

    # if global_config.DEBUG_MODE_SCHEDULER:
    # print("old load index:", load_index, "new load index:", new_load_index)
    # print("old use gpu list:", use_gpu_list, "new use gpu list:", new_use_gpu_list)


def scheduler_inc():
    is_system_running = global_var.is_system_running
    stage_lists_gpu = global_var.stage_lists_gpu
    scheduler_inc_sem = global_var.scheduler_inc_sem

    inc_gpu_num = 1
    import time
    
    while is_system_running:
        scheduler_inc_sem.acquire()
        
        start_time = time.time()
        for model_idx in range(len(stage_lists_gpu)):
            if global_var.cur_request_num.get() > 0 or not is_system_running:
                break

            if global_var.is_model_waiting_offload_lists[model_idx].is_set():
                continue
            
            readjust_thread = Thread(
                target=readjust_func, args=(model_idx, inc_gpu_num)
            )
            readjust_thread.start()
            # if global_config.DEBUG_MODE_SCHEDULER:
            # print("scheduler inc begin:", model_idx)
        # print("sche inc time:", time.time() - start_time)


def scheduler_dec():
    is_system_running = global_var.is_system_running
    stage_lists_gpu = global_var.stage_lists_gpu
    scheduler_dec_sem = global_var.scheduler_dec_sem

    dec_gpu_num = len(global_var.cuda_devices)
    
    import time
    while is_system_running:
        scheduler_dec_sem.acquire()

        start_time = time.time()
        for model_idx in range(len(stage_lists_gpu)):
            if global_var.cur_request_num.get() > 0 or not is_system_running:
                break

            if global_var.is_model_waiting_offload_lists[model_idx].is_set():
                continue
            
            readjust_thread = Thread(
                target=readjust_func, args=(model_idx, dec_gpu_num)
            )
            readjust_thread.start()
            # if global_config.DEBUG_MODE_SCHEDULER:
            # print("scheduler dec begin:", model_idx)
        # print("sche dec time:", time.time() - start_time)


def start_scheduler():
    global_var.cur_request_num.set(0)
    global_var.cur_wait_offload_num.set(0)
    scheduler_inc_thread = Thread(target=scheduler_inc)
    scheduler_dec_thread = Thread(target=scheduler_dec)
    scheduler_inc_thread.start()
    scheduler_dec_thread.start()
