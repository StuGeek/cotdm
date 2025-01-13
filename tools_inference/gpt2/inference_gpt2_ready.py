import time
import torch
from global_data import global_var
from global_data.global_class import ThreadWithReturnValue
from tools_inference.gpt2.inference_gpt2_cotdm import cal_func, get_next_token, get_generated_text_cotdm


def inference_gpt2_ready(encoded_input, model_idx):
    start_inf_time = time.time()
    global_var.offload_time_list[model_idx] = 0

    cal_stream = torch.cuda.Stream()
    cal_thread = ThreadWithReturnValue(
        target=cal_func, args=(encoded_input, cal_stream, model_idx)
    )

    cal_thread.start()
    lm_logits = cal_thread.join()
    next_token = get_next_token(lm_logits)

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    ttft = inf_lat
    return next_token, inf_lat, ttft


def get_generated_text_ready(encoded_input, model_idx):
    return get_generated_text_cotdm(encoded_input, model_idx, inference_gpt2_ready)
