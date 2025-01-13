import time
import torch
from global_data import global_var
from global_data.global_class import ThreadWithReturnValue
from tools_inference.bert.inference_bert_cotdm import cal_func, load_func
from layer_inf.layer_inf_bert import get_predicted_token
from threading import Thread


def inference_bert_totoff(encoded_input, model_idx):
    start_inf_time = time.time()
    global_var.inf_start_time = start_inf_time
    global_var.offload_time_list[model_idx] = 0
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    tokenizer = global_var.tokenizer_lists[model_idx]

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
    prediction_scores = cal_thread.join()
    predicted_token = get_predicted_token(prediction_scores, encoded_input, tokenizer)

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    ttft = inf_lat
    return predicted_token, inf_lat, ttft
