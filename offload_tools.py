import torch
import time
from global_data import global_var, global_config

eps = 1e-4

def offload_stages_from_gpu_to_cpu(model_idx):
    stage_lists_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    load_index_list = global_var.load_index_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]

    begin_idx = load_index_list[0]
    end_idx = load_index_list[-1]
    for i in range(begin_idx, end_idx):
        with stage_cond_list[i]:
            is_stage_loaded[i] = False

            if stage_lists_gpu[i] != None and stage_lists_gpu[i].stage != None:
                for j in range(len(stage_lists_gpu[i].stage)):
                    stage_lists_gpu[i].stage[j] = stage_lists_gpu[i].stage[j].cpu()

    torch.cuda.empty_cache()


def offload_model_from_gpu_to_cpu(model_idx):
    stage_list = global_var.stage_lists_gpu[model_idx]
    total_layer_num = len(stage_list)
    is_stage_loaded_list = global_var.is_stage_loaded_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]

    for i in range(total_layer_num):
        with stage_cond_list[i]:
            is_stage_loaded_list[i] = False

            if stage_list[i] != None and stage_list[i].stage != None:
                for j in range(len(stage_list[i].stage)):
                    stage_list[i].stage[j] = stage_list[i].stage[j].cpu()

    torch.cuda.empty_cache()


def offload_all_models_from_gpu_to_cpu():
    stage_lists_gpu = global_var.stage_lists_gpu
    is_stage_loaded = global_var.is_stage_loaded_lists

    for i in range(len(stage_lists_gpu)):
        if stage_lists_gpu[i] != None:
            for j in range(len(stage_lists_gpu[i])):
                if stage_lists_gpu[i][j] != None:
                    for k in range(len(stage_lists_gpu[i][j].stage)):
                        stage_lists_gpu[i][j].stage[k] = stage_lists_gpu[i][j].stage[k].cpu()
                        # stage_lists_gpu[i][j].stage[k] = None
                is_stage_loaded[i][j] = False


    torch.cuda.empty_cache()


def offload_func(model_idx, countdown=0):
    if global_var.model_request_num_list[model_idx] > 0:
        if global_config.DEBUG_MODE_SCHEDULER:
            print("global var cur_wait_offload_num:", global_var.cur_wait_offload_num.get())
        if global_var.cur_wait_offload_num.decrementAndGet() == 0:
            if global_config.DEBUG_MODE_SCHEDULER:
                print("global var cur_wait_offload_num decrease:", global_var.cur_wait_offload_num.get())
            global_var.scheduler_dec_sem.release()
        return

    offload_time = time.time() + countdown
    global_var.offload_time_list[model_idx] = offload_time
    time.sleep(countdown)


    # offload_start_time = time.time()
    if abs(offload_time - global_var.offload_time_list[model_idx]) > eps:
        if global_config.DEBUG_MODE_SCHEDULER:
            print("global var cur_wait_offload_num:", global_var.cur_wait_offload_num.get())
        if global_var.cur_wait_offload_num.decrementAndGet() == 0:
            if global_config.DEBUG_MODE_SCHEDULER:
                print("global var cur_wait_offload_num decrease:", global_var.cur_wait_offload_num.get())
            global_var.scheduler_dec_sem.release()
        return

    global_var.is_model_using_lists[model_idx].clear()
    global_var.is_model_waiting_offload_lists[model_idx].clear()
    offload_stages_from_gpu_to_cpu(model_idx)
    global_var.is_model_using_lists[model_idx].set()

    if global_config.DEBUG_MODE_SCHEDULER:
        print("global var cur_wait_offload_num:", global_var.cur_wait_offload_num.get())
    if global_var.cur_wait_offload_num.decrementAndGet() == 0:
        if global_config.DEBUG_MODE_SCHEDULER:
            print("global var cur_wait_offload_num decrease:", global_var.cur_wait_offload_num.get())
        global_var.scheduler_dec_sem.release()
    # offload_end_time = time.time()
    # print("offload time:", offload_end_time - offload_start_time)