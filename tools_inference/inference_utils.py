import torch
import time
import sys
sys.path.append(".")
sys.path.append("..")
from global_data import global_var
from layer_inf.layer_inf_gpt2 import (
    gpt2_preprocess,
    wte_forward,
    wpe_forward,
    drop_forward,
    block_preprocess,
    gpt2_block_forward_i,
    ln_f_forward,
    get_next_token,
)
from torch.nn import functional as F


def cal_model_gpt2(encoded_input, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    config = global_var.config_lists[model_idx]

    total_h_layer_num = config.total_h_layer_num
    preprocess_output = gpt2_preprocess(encoded_input, config, total_h_layer_num)
    hidden_states = wte_forward(stage_list_gpu[0].stage[0], preprocess_output)
    hidden_states = wpe_forward(
        stage_list_gpu[1].stage[0], preprocess_output, hidden_states
    )
    hidden_states = drop_forward(stage_list_gpu[2].stage[0], hidden_states)
    block_preprocess_output = block_preprocess(preprocess_output, hidden_states, config)
    for i in range(total_h_layer_num):
        preprocess_output, block_preprocess_output, hidden_states = (
            gpt2_block_forward_i(
                stage_list_gpu[i + 3].stage[0],
                i,
                preprocess_output,
                block_preprocess_output,
                hidden_states,
                config,
            )
        )
    transformer_outputs = ln_f_forward(
        stage_list_gpu[total_h_layer_num + 3].stage[0],
        preprocess_output,
        block_preprocess_output,
        hidden_states,
    )
    hidden_states = transformer_outputs[0]
    stage_lm_head_weight = stage_list_gpu[0].stage[0].state_dict()["weight"]
    lm_logits = F.linear(hidden_states, stage_lm_head_weight)

    torch.cuda.synchronize()
    return lm_logits


def get_generated_text_gpt2(encoded_input, model_idx, inference_gpt2_func):
    start_inf_time = time.time()
    global_var.offload_time_list[model_idx] = 0

    GPT2_MAX_LENGTH = 300
    max_length = GPT2_MAX_LENGTH
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    tokenizer = global_var.tokenizer_lists[model_idx]

    generated_tokens = encoded_input["input_ids"][0].tolist()
    attention_mask = encoded_input["attention_mask"][0].tolist()

    with torch.no_grad():
        next_token, _, ttft = inference_gpt2_func(encoded_input, model_idx)
    generated_tokens.append(next_token)
    attention_mask.append(1)

    while len(generated_tokens) < max_length:
        # start_token_time = time.time()
        encoded_input["input_ids"] = torch.tensor(
            [generated_tokens], dtype=torch.long, device=use_gpu_list[0]
        )
        encoded_input["attention_mask"] = torch.tensor(
            [attention_mask], dtype=torch.long, device=use_gpu_list[0]
        )
        with torch.no_grad():
            lm_logits = cal_model_gpt2(encoded_input, model_idx)
        next_token = get_next_token(lm_logits)
        generated_tokens.append(next_token)
        attention_mask.append(1)
        # print(time.time() - start_token_time)

        # if next_token == tokenizer.eos_token_id:
        #     break
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    return generated_text, inf_lat, ttft