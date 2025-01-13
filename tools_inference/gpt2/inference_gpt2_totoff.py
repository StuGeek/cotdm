import time
import torch
from global_data import global_var
from transformers import logging
from threading import Thread
from global_data.global_class import ThreadWithReturnValue
from tools_inference.gpt2.inference_gpt2_cotdm import cal_func, load_func, get_next_token, get_generated_text_cotdm

logging.set_verbosity_error()


def inference_gpt2_totoff(encoded_input, model_idx):
    start_inf_time = time.time()
    global_var.offload_time_list[model_idx] = 0
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]

    cal_stream = torch.cuda.Stream()
    cal_thread = ThreadWithReturnValue(
        target=cal_func, args=(encoded_input, cal_stream, model_idx)
    )

    load_index_list = global_var.load_index_lists[model_idx]
    load_index_len = len(load_index_list)
    if load_index_list[0] != -1:
        load_streams = []
        for i in range(load_index_len - 1):
            load_streams.append(torch.cuda.Stream())

        load_threads = []
        for i in range(load_index_len - 1):
            load_thread = Thread(
                target=load_func,
                args=(
                    load_streams[i],
                    load_index_list[i],
                    load_index_list[i + 1],
                    use_gpu_list[i],
                    model_idx,
                ),
            )
            load_threads.append(load_thread)

    if load_index_list[0] != -1:
        for load_thread in load_threads:
            load_thread.start()

    if load_index_list[0] != -1:
        for load_thread in load_threads:
            load_thread.join()

    cal_thread.start()
    lm_logits = cal_thread.join()
    next_token = get_next_token(lm_logits)

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    ttft = inf_lat
    return next_token, inf_lat, ttft


def get_generated_text_totoff(encoded_input, model_idx):
    return get_generated_text_cotdm(encoded_input, model_idx, inference_gpt2_totoff)