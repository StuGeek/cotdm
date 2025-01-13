import time
import torch
from threading import Thread
from torch.nn import functional as F
from layer_inf.layer_inf_gpt2 import (
    gpt2_preprocess,
    wte_forward,
    wpe_forward,
    drop_forward,
    block_preprocess,
    gpt2_block_forward_i,
    ln_f_forward,
    lm_head_forward,
)
from global_data import global_var
from global_data.global_class import ThreadWithReturnValue
from tools_inference.gpt2.inference_gpt2_cotdm import get_next_token


def cal_model(encoded_input, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    load_index = global_var.load_index_lists[model_idx]
    config = global_var.config_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]

    cal_index = 0

    cal_device = torch.device(use_gpu_list[0])
    total_h_layer_num = config.total_h_layer_num
    total_layer_num = config.total_layer_num

    while True:
        with stage_cond_list[cal_index]:
            if is_stage_loaded[cal_index] == False:
                stage_cond_list[cal_index].wait()

        model_stages = stage_list_gpu[cal_index].stage
        stage_type = stage_list_gpu[cal_index].type

        with torch.no_grad():
            if stage_type == 0:
                stage_wte = model_stages[0]
                preprocess_output = gpt2_preprocess(encoded_input, config, total_h_layer_num)
                hidden_states = wte_forward(stage_wte, preprocess_output)
            elif stage_type == 1:
                stage_wpe = model_stages[0]
                if hidden_states != None and hidden_states.device != cal_device:
                    preprocess_output = preprocess_output.to(cal_device, non_blocking=True)
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)
                hidden_states = wpe_forward(
                    stage_wpe, preprocess_output, hidden_states
                )
            elif stage_type == 2:
                stage_drop = model_stages[0]
                if hidden_states.device != cal_device:
                    preprocess_output = preprocess_output.to(cal_device, non_blocking=True)
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)
                hidden_states = drop_forward(stage_drop, hidden_states)
                block_preprocess_output = block_preprocess(
                    preprocess_output, hidden_states, config
                )
            elif stage_type == 3:
                stage_h = model_stages[0]
                if hidden_states.device != cal_device:
                    preprocess_output = preprocess_output.to(cal_device, non_blocking=True)
                    block_preprocess_output = block_preprocess_output.to(cal_device, non_blocking=True)
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)
                preprocess_output, block_preprocess_output, hidden_states = (
                    gpt2_block_forward_i(
                        stage_h,
                        cal_index - 3,
                        preprocess_output,
                        block_preprocess_output,
                        hidden_states,
                        config,
                    )
                )
            elif stage_type == 4:
                stage_ln_f = model_stages[0]
                # print(hidden_states.device, cal_device)
                if hidden_states.device != cal_device:
                    preprocess_output = preprocess_output.to(cal_device, non_blocking=True)
                    block_preprocess_output = block_preprocess_output.to(cal_device, non_blocking=True)
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)
                transformer_outputs = ln_f_forward(
                    stage_ln_f,
                    preprocess_output,
                    block_preprocess_output,
                    hidden_states,
                )
                hidden_states = transformer_outputs[0]

                cal_device = torch.device(use_gpu_list[0])
                if hidden_states.device != cal_device:
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)

                stage_lm_head_weight = stage_list_gpu[0].stage[0].state_dict()["weight"]
                lm_logits = F.linear(hidden_states, stage_lm_head_weight)

        cal_index += 1

        if cal_index >= total_layer_num:
            break

    # print("total_cal_time: ", total_cal_time, "aver: ", total_cal_time / (total_stage_num))
    # print("wait times: ", wait_times)
    # print("cost time: ", cost_time, "aver: ", cost_time / wait_times)
    torch.cuda.synchronize()
    return lm_logits


def load_model(begin_idx, end_idx, from_device, to_device, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]

    for i in range(begin_idx, end_idx):
        with stage_cond_list[i]:
            if is_stage_loaded[i] == True:
                continue

            for j in range(len(stage_list_gpu[i].stage)):
                stage_list_gpu[i].stage[j] = (
                    stage_list_gpu[i].stage[j].to(from_device, non_blocking=True)
                )

            stage_cond_list[i].notify()

    for i in range(begin_idx, end_idx):
        with stage_cond_list[i]:
            if is_stage_loaded[i] == True:
                continue

            for j in range(len(stage_list_gpu[i].stage)):
                stage_list_gpu[i].stage[j] = (
                    stage_list_gpu[i].stage[j].to(to_device, non_blocking=True)
                )
                stage_list_gpu[i].stage[j].eval()

            is_stage_loaded[i] = True
            stage_cond_list[i].notify()


def cal_func(encoded_input, stream, model_idx):
    with torch.cuda.stream(stream):
        outputs = cal_model(encoded_input, model_idx)
    return outputs


def load_func(stream, begin_idx, end_idx, from_device, to_device, model_idx):
    with torch.cuda.stream(stream):
        load_model(begin_idx, end_idx, from_device, to_device, model_idx)


def inference_gpt2_deepplan(encoded_input, model_idx):
    start_inf_time = time.time()
    global_var.inf_start_time = start_inf_time
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
                    use_gpu_list[0],
                    model_idx,
                ),
            )
            load_threads.append(load_thread)

    cal_thread.start()
    if load_index_list[0] != -1:
        for load_thread in load_threads:
            load_thread.start()

    lm_logits = cal_thread.join()
    # if load_index_list[0] != -1:
    #     for load_thread in load_threads:
    #         load_thread.join()

    next_token = get_next_token(lm_logits)

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    ttft = inf_lat
    return next_token, inf_lat, ttft


def get_generated_text_deepplan(encoded_input, model_idx):
    start_inf_time = time.time()
    global_var.offload_time_list[model_idx] = 0
    
    GPT2_MAX_LENGTH = 800
    max_length = GPT2_MAX_LENGTH
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    tokenizer = global_var.tokenizer_lists[model_idx]

    generated_tokens = encoded_input["input_ids"][0].tolist()
    attention_mask = encoded_input["attention_mask"][0].tolist()

    with torch.no_grad():
        next_token, _, ttft = inference_gpt2_deepplan(encoded_input, model_idx)
    generated_tokens.append(next_token)
    attention_mask.append(1)
  
    while len(generated_tokens) < max_length:
        encoded_input["input_ids"] = torch.tensor([generated_tokens], dtype=torch.long, device=use_gpu_list[0])
        encoded_input["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long, device=use_gpu_list[0])
        with torch.no_grad():
            lm_logits = cal_model(encoded_input, model_idx)
        next_token = get_next_token(lm_logits)
        generated_tokens.append(next_token)
        attention_mask.append(1)
         
        # if next_token == tokenizer.eos_token_id:  
        #     break  
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    return generated_text, inf_lat, ttft