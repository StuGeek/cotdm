import time
import torch
from threading import Thread
import sys
sys.path.append(".")
sys.path.append("..")
from layer_inf.layer_inf_bert import (
    bert_inference_preprocess,
    bert_embeddings_preprocess,
    bert_word_embeddings_forward,
    bert_token_type_embeddings_forward,
    bert_position_embeddings_forward,
    bert_embeddings_LayerNorm_or_dropout_forward,
    bert_embeddings_foward,
    bert_encoder_preprocess,
    bert_layer_forward_i,
    bert_encoder_layer_forward,
    bert_encoder_layer_attention_forward,
    bert_encoder_layer_intermediate_output_forward,
    bert_encoder_layer_intermediate_forward,
    bert_encoder_layer_output_forward,
    get_encoder_outputs,
    bert_pooler_forward,
    # bert_only_mlm_head_forward,
    bert_predictions_head_transform_forward,
    bert_decoder_forward,
    bert_hidden_states_forward,
    get_predicted_token,
)
from global_data import global_var
from global_data.global_class import ThreadWithReturnValue
from torch.nn import functional as F
from .inference_bert_cotdm import trans_intermediate_data_to_device


def cal_model(encoded_input, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    config = global_var.config_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]

    cal_index = 0

    cal_device = torch.device(use_gpu_list[0])
    total_encoder_layer_num = config.total_encoder_layer_num
    total_layer_num = config.total_layer_num

    while True:
        with stage_cond_list[cal_index]:
            if not is_stage_loaded[cal_index]:
                stage_cond_list[cal_index].wait()

        model_stages = stage_list_gpu[cal_index].stage
        stage_type = stage_list_gpu[cal_index].type

        with torch.no_grad():
            # if stage_type == 0:
            #     stage_embeddings = model_stages[0]
            #     inf_intermediate_data = bert_inference_preprocess(
            #         encoded_input, config
            #     )
            #     bert_embeddings_preprocess(inf_intermediate_data, config)
            #     bert_embeddings_foward(stage_embeddings, inf_intermediate_data)
            #     bert_encoder_preprocess(inf_intermediate_data, config)

            #     print(cal_index, "cal_time: ", time.time() - start)
            if stage_type == 0:
                stage_word_embeddings = model_stages[0]
                stage_embeddings_position_ids = model_stages[1]
                stage_embeddings_token_type_ids = model_stages[2]
                inf_intermediate_data = bert_inference_preprocess(encoded_input, config)
                bert_embeddings_preprocess(
                    stage_embeddings_position_ids,
                    stage_embeddings_token_type_ids,
                    inf_intermediate_data,
                    config,
                )
                bert_word_embeddings_forward(
                    stage_word_embeddings, inf_intermediate_data
                )
            elif stage_type == 1:
                stage_token_type_embeddings = model_stages[0]
                # if inf_intermediate_data.hidden_states.device != cal_device:
                #     trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                bert_token_type_embeddings_forward(
                    stage_token_type_embeddings, inf_intermediate_data
                )
            elif stage_type == 2:
                stage_position_embeddings = model_stages[0]
                # if inf_intermediate_data.input_ids != cal_device:
                #     inf_intermediate_data.input_ids = inf_intermediate_data.input_ids.to(cal_device, non_blocking=True)
                bert_position_embeddings_forward(
                    stage_position_embeddings, inf_intermediate_data, config
                )
            elif stage_type == 3:
                stage_embeddings_LayerNorm_or_dropout = model_stages[0]
                # if inf_intermediate_data.hidden_states.device != cal_device:
                #     trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                bert_embeddings_LayerNorm_or_dropout_forward(
                    stage_embeddings_LayerNorm_or_dropout, inf_intermediate_data
                )
                if cal_index == 4:
                    bert_encoder_preprocess(inf_intermediate_data, config)
            elif stage_type == 4:
                stage_encoder_layer = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device or inf_intermediate_data.extended_attention_mask.device != cal_device:
                    trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                # start = time.time()

                # bert_layer_forward_i(
                #     stage_encoder_layer, cal_index - 1, inf_intermediate_data, config
                # )
                bert_layer_forward_i(
                    stage_encoder_layer, 0, inf_intermediate_data, config
                )

                if cal_index == total_encoder_layer_num + 4:
                    encoder_outputs = get_encoder_outputs(inf_intermediate_data)
                    hidden_states = encoder_outputs.last_hidden_state
            elif stage_type == 5:
                stage_pooler = model_stages[0]
                if encoder_outputs.last_hidden_state.device != cal_device:
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.to(
                        cal_device, non_blocking=True
                    )
                # start = time.time()
                pooler_outputs = bert_pooler_forward(
                    stage_pooler,
                    encoder_outputs,
                    inf_intermediate_data.return_dict,
                )
                hidden_states = pooler_outputs.last_hidden_state
            elif stage_type == 6:
                stage_cls = model_stages[0]
                if hidden_states.device != cal_device:
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)

                hidden_states = bert_hidden_states_forward(stage_cls, hidden_states)

            elif stage_type == 7:
                stage_cls_decoder_bias = model_stages[0]
                stage_cls_decoder_weight = (
                    stage_list_gpu[0].stage[0].state_dict()["weight"]
                )

                if hidden_states.device != cal_device:
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)

                hidden_states = F.linear(
                    hidden_states, stage_cls_decoder_weight, stage_cls_decoder_bias
                )

        cal_index += 1

        if cal_index >= total_layer_num:
            break

    torch.cuda.synchronize()

    prediction_scores = hidden_states
    return prediction_scores


def load_model(begin_idx, end_idx, from_device, to_device, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    total_layer_num = len(stage_list_gpu)

    if end_idx == total_layer_num:
        end_idx -= 1

    for i in range(begin_idx, end_idx):
        with stage_cond_list[i]:
            if is_stage_loaded[i] == True:
                continue

            for j in range(len(stage_list_gpu[i].stage)):
                stage_list_gpu[i].stage[j] = (
                    stage_list_gpu[i].stage[j].to(from_device, non_blocking=True)
                )
                if i != 0 or j == 0:
                    stage_list_gpu[i].stage[j].eval()

            if from_device == use_gpu_list[0]:
                is_stage_loaded[i] = True
            stage_cond_list[i].notify()

    if from_device == use_gpu_list[0]:
        with stage_cond_list[-1]:
            if is_stage_loaded[-1] == True:
                stage_cond_list[-1].notify()
                return

            # start = time.time()
            for i in range(len(stage_list_gpu[-1].stage)):
                stage_list_gpu[-1].stage[i] = (
                    stage_list_gpu[-1].stage[i].to(from_device, non_blocking=True)
                )
                # print(i, "load time1: ", time.time() - start, start)
                # stage_list_gpu[-1].stage[i].eval()
            is_stage_loaded[-1] = True
            stage_cond_list[-1].notify()
        return

    for i in range(begin_idx, end_idx):
        with stage_cond_list[i]:
            if is_stage_loaded[i] == True:
                continue

            for j in range(len(stage_list_gpu[i].stage)):
                stage_list_gpu[i].stage[j] = (
                    stage_list_gpu[i].stage[j].to(to_device, non_blocking=True)
                )
                if i != 0 or j == 0:
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


def inference_bert_deepplan(encoded_input, model_idx):
    start_inf_time = time.time()
    global_var.inf_start_time = start_inf_time
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
                    use_gpu_list[0],
                    model_idx,
                ),
            )
            load_threads.append(load_thread)

    cal_thread.start()
    if load_index_list[0] != -1:
        for load_thread in load_threads:
            load_thread.start()

    prediction_scores = cal_thread.join()
    # if load_index_list[0] != -1:
    #     for load_thread in load_threads:
    #         load_thread.join()

    predicted_token = get_predicted_token(prediction_scores, encoded_input, tokenizer)

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    ttft = inf_lat
    return predicted_token, inf_lat, ttft
