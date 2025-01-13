import time
import torch
from global_data import global_var
from global_data.global_class import ThreadWithReturnValue
from tools_inference.bert.inference_bert_cotdm import cal_func
from layer_inf.layer_inf_bert import get_predicted_token


def inference_bert_ready(encoded_input, model_idx):
    start_inf_time = time.time()
    global_var.offload_time_list[model_idx] = 0
    global_var.inf_start_time = start_inf_time
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    tokenizer = global_var.tokenizer_lists[model_idx]

    cal_stream = torch.cuda.Stream()
    cal_thread = ThreadWithReturnValue(
        target=cal_func, args=(encoded_input, cal_stream, model_idx)
    )

    cal_thread.start()
    prediction_scores = cal_thread.join()
    predicted_token = get_predicted_token(prediction_scores, encoded_input, tokenizer)

    # torch.cuda.synchronize()

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    ttft = inf_lat
    return predicted_token, inf_lat, ttft
