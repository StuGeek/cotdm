import torch
import threading
from offload_tools import offload_all_models_from_gpu_to_cpu
from global_data.global_class import AtomicInteger

inf_start_time = None

is_system_running = False
scheduler_inc_sem = threading.Semaphore(0)
scheduler_dec_sem = threading.Semaphore(0)

stage_lists_gpu = []
stage_lists_cpu = []
is_stage_loaded_lists = []
load_index_lists = []
model_use_gpu_lists = []
config_lists = []
tokenizer_lists = []
model_stage_cond_lists = []
is_model_using_lists = []
is_model_waiting_offload_lists = []
offload_time_list = []
model_name_slo_idx_dict = {}
model_request_num_list = []
model_stage_mem_list = []

single_layer_time_list = []
model_slo_list = []
cur_request_num = AtomicInteger(0)
cur_wait_offload_num = AtomicInteger(0)

device_count = torch.cuda.device_count()
memory_fully_loaded_list = [0 for i in range(device_count)]
cuda_devices = ["cuda:" + str(i) for i in range(device_count)]

def clear_global_var():
    global scheduler_inc_sem
    global scheduler_dec_sem
    global stage_lists_gpu
    global stage_lists_cpu
    global is_stage_loaded_lists
    global model_use_gpu_lists
    global load_index_lists
    global config_lists
    global tokenizer_lists
    global offload_time_list
    global model_stage_cond_lists
    global is_model_using_lists
    global is_model_waiting_offload_lists
    global model_name_slo_idx_dict
    global model_request_num_list
    global model_stage_mem_list

    global single_layer_time_list
    global model_slo_list

    global cur_request_num
    global cur_wait_offload_num

    global memory_fully_loaded_list

    scheduler_inc_sem = threading.Semaphore(0)
    scheduler_dec_sem = threading.Semaphore(0)
    offload_all_models_from_gpu_to_cpu()

    stage_lists_gpu = []
    stage_lists_cpu = []
    is_stage_loaded_lists = []
    load_index_lists = []
    model_use_gpu_lists = []
    config_lists = []
    tokenizer_lists = []
    model_stage_cond_lists = []
    is_model_using_lists = []
    is_model_waiting_offload_lists = []
    offload_time_list = []
    model_name_slo_idx_dict = {}
    model_request_num_list = []
    model_stage_mem_list = []

    single_layer_time_list = []
    model_slo_list = []

    cur_request_num = AtomicInteger(0)
    cur_wait_offload_num = AtomicInteger(0)
    memory_fully_loaded_list = [0 for i in range(device_count)]
