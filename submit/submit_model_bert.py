import time
import torch
import asyncio
import os
import sys
from threading import Thread
import argparse
import json
from aiohttp import web

sys.path.append(".")
sys.path.append("..")

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from min_deploy import min_deploy
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertPooler,
)
from transformers.activations import ACT2FN
from torch.nn import functional as F
from layer_inf.layer_inf_bert import (
    bert_inference_preprocess,
    bert_embeddings_preprocess,
    bert_word_embeddings_forward,
    bert_token_type_embeddings_forward,
    bert_position_embeddings_forward,
    bert_embeddings_LayerNorm_or_dropout_forward,
    bert_encoder_preprocess,
    bert_layer_forward_i,
    get_encoder_outputs,
    bert_pooler_forward,
    bert_hidden_states_forward,
    get_predicted_token,
)
import threading
from global_data import global_config, global_var
from global_data.global_class import (
    ModelStage,
    SingleLayerTime,
    ThreadWithReturnValue,
    ModelNameAndSlo,
)
from tools_inference.bert.inference_bert_cotdm import trans_intermediate_data_to_device
from torch import nn
from server_utils import get_model_idx
from min_deploy import cal_lat_by_deploynum_and_avaigpu
from scheduler import get_single_model_placement
from client.client_single_inf_req import send_single_inf_req
from submit.submit_utils import (
    count_parameters,
    print_parameters_num,
    get_stage_mem_list,
    print_model_size,
)
from offload_tools import offload_stages_from_gpu_to_cpu, offload_model_from_gpu_to_cpu

TEST_TIME = 10
DEBUG_MODE_SUBMIT = False
SUBMIT_PORT = 8085

cal_time_list = []
extra_trans_time_list = []
trans_time_list = []
load_time_list = []

def cal_model_ready(encoded_input, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    load_index = global_var.load_index_lists[model_idx]
    config = global_var.config_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]
    load_index_len = len(load_index)
    use_gpu_num = len(use_gpu_list)

    cal_index = 0
    hidden_states = None
    global cal_time_list, extra_trans_time_list, trans_time_list

    total_encoder_layer_num = config.total_encoder_layer_num
    total_layer_num = len(stage_list_gpu)
    while True:
        cal_device = torch.device(use_gpu_list[0])
        for i, device_name in enumerate(use_gpu_list):
            if i < load_index_len - 1 and cal_index < load_index[i + 1]:
                cal_device = torch.device(device_name)
                next_cal_device = torch.device(
                    use_gpu_list[(i + 1) % use_gpu_num]
                )
                break

        with stage_cond_list[cal_index]:
            if not is_stage_loaded[cal_index]:
                stage_cond_list[cal_index].wait()

        model_stages = stage_list_gpu[cal_index].stage
        stage_type = stage_list_gpu[cal_index].type

        start_cal_time = time.time()

        with torch.no_grad():
            # if stage_type == 0:
            #     stage_embeddings = model_stages[0]
            #     inf_intermediate_data = bert_inference_preprocess(
            #         stage_embeddings, encoded_input, config
            #     )
            #     embedding_output = bert_embeddings_foward(
            #         stage_embeddings, inf_intermediate_data
            #     )
            #     bert_encoder_preprocess(inf_intermediate_data, config)

            #     inf_intermediate_data.hidden_states = embedding_output

            #     end_cal_time = time.time()
            #     cal_time_list[cal_index] += end_cal_time - start_cal_time

            #     start_trans_time = time.time()

            #     trans_intermediate_data_to_device(inf_intermediate_data, next_cal_device)
            #     trans_intermediate_data_to_device(inf_intermediate_data, cal_device)

            #     end_trans_time = time.time()
            #     trans_time_list[cal_index] += end_trans_time - start_trans_time
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

                end_cal_time = time.time()
                cal_time_list[cal_index] += end_cal_time - start_cal_time

                start_trans_time = time.time()
                inf_intermediate_data.inputs_embeds = (
                    inf_intermediate_data.inputs_embeds.to(
                        next_cal_device, non_blocking=True
                    )
                )
                inf_intermediate_data.token_type_ids = (
                    inf_intermediate_data.token_type_ids.to(
                        next_cal_device, non_blocking=True
                    )
                )
                inf_intermediate_data.inputs_embeds = (
                    inf_intermediate_data.inputs_embeds.to(
                        cal_device, non_blocking=True
                    )
                )
                inf_intermediate_data.token_type_ids = (
                    inf_intermediate_data.token_type_ids.to(
                        cal_device, non_blocking=True
                    )
                )
                end_trans_time = time.time()
                trans_time_list[cal_index] += end_trans_time - start_trans_time
            elif stage_type == 1:
                stage_token_type_embeddings = model_stages[0]
                if inf_intermediate_data.inputs_embeds.device != cal_device:
                    start_extra_trans_time = time.time()
                    inf_intermediate_data.inputs_embeds = (
                        inf_intermediate_data.inputs_embeds.to(
                            cal_device, non_blocking=True
                        )
                    )
                    inf_intermediate_data.token_type_ids = (
                        inf_intermediate_data.token_type_ids.to(
                            cal_device, non_blocking=True
                        )
                    )
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
                bert_token_type_embeddings_forward(
                    stage_token_type_embeddings, inf_intermediate_data
                )

                end_cal_time = time.time()
                cal_time_list[cal_index] += end_cal_time - start_cal_time

                start_trans_time = time.time()
                inf_intermediate_data.hidden_states = (
                    inf_intermediate_data.hidden_states.to(
                        next_cal_device, non_blocking=True
                    )
                )
                inf_intermediate_data.position_ids = (
                    inf_intermediate_data.position_ids.to(
                        next_cal_device, non_blocking=True
                    )
                )
                inf_intermediate_data.hidden_states = (
                    inf_intermediate_data.hidden_states.to(
                        cal_device, non_blocking=True
                    )
                )
                inf_intermediate_data.position_ids = (
                    inf_intermediate_data.position_ids.to(cal_device, non_blocking=True)
                )
                end_trans_time = time.time()
                trans_time_list[cal_index] += end_trans_time - start_trans_time
            elif stage_type == 2:
                stage_position_embeddings = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device:
                    start_extra_trans_time = time.time()
                    inf_intermediate_data.hidden_states = (
                        inf_intermediate_data.hidden_states.to(
                            cal_device, non_blocking=True
                        )
                    )
                    inf_intermediate_data.position_ids = (
                        inf_intermediate_data.position_ids.to(
                            cal_device, non_blocking=True
                        )
                    )
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
                bert_position_embeddings_forward(
                    stage_position_embeddings, inf_intermediate_data, config
                )

                end_cal_time = time.time()
                cal_time_list[cal_index] += end_cal_time - start_cal_time

                start_trans_time = time.time()
                trans_intermediate_data_to_device(
                    inf_intermediate_data, next_cal_device
                )
                trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                end_trans_time = time.time()
                trans_time_list[cal_index] += end_trans_time - start_trans_time
            elif stage_type == 3:
                stage_embeddings_LayerNorm_or_dropout = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device:
                    start_extra_trans_time = time.time()
                    trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
                bert_embeddings_LayerNorm_or_dropout_forward(
                    stage_embeddings_LayerNorm_or_dropout, inf_intermediate_data
                )
                if cal_index == 4:
                    bert_encoder_preprocess(inf_intermediate_data, config)

                end_cal_time = time.time()
                cal_time_list[cal_index] += end_cal_time - start_cal_time

                start_trans_time = time.time()
                trans_intermediate_data_to_device(
                    inf_intermediate_data, next_cal_device
                )
                trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                end_trans_time = time.time()
                trans_time_list[cal_index] += end_trans_time - start_trans_time
            elif stage_type == 4:
                stage_encoder_layer = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device:
                    start_extra_trans_time = time.time()
                    trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
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

                end_cal_time = time.time()
                cal_time_list[cal_index] += end_cal_time - start_cal_time

                start_trans_time = time.time()
                trans_intermediate_data_to_device(
                    inf_intermediate_data, next_cal_device
                )
                trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                if hidden_states != None:
                    hidden_states = hidden_states.to(next_cal_device, non_blocking=True)
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)

                end_trans_time = time.time()
                trans_time_list[cal_index] += end_trans_time - start_trans_time
            elif stage_type == 5:
                stage_pooler = model_stages[0]
                if encoder_outputs.last_hidden_state.device != cal_device:
                    start_extra_trans_time = time.time()
                    trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
                pooler_outputs = bert_pooler_forward(
                    stage_pooler,
                    encoder_outputs,
                    inf_intermediate_data.return_dict,
                )
                hidden_states = pooler_outputs.last_hidden_state
                end_cal_time = time.time()
                cal_time_list[cal_index] += end_cal_time - start_cal_time

                start_trans_time = time.time()
                hidden_states = hidden_states.to(next_cal_device, non_blocking=True)
                hidden_states = hidden_states.to(cal_device, non_blocking=True)
                end_trans_time = time.time()
                trans_time_list[cal_index] += end_trans_time - start_trans_time
            elif stage_type == 6:
                stage_cls = model_stages[0]
                if hidden_states.device != cal_device:
                    start_extra_trans_time = time.time()
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )

                hidden_states = bert_hidden_states_forward(stage_cls, hidden_states)
                end_cal_time = time.time()
                cal_time_list[cal_index] += end_cal_time - start_cal_time

                start_trans_time = time.time()
                hidden_states = hidden_states.to(next_cal_device, non_blocking=True)
                hidden_states = hidden_states.to(cal_device, non_blocking=True)
                end_trans_time = time.time()
                trans_time_list[cal_index] += end_trans_time - start_trans_time
            elif stage_type == 7:
                stage_cls_decoder_bias = model_stages[0]
                stage_cls_decoder_weight = (
                    stage_list_gpu[0].stage[0].state_dict()["weight"]
                )
                cal_device = torch.device(global_var.model_use_gpu_lists[model_idx][0])

                if hidden_states.device != cal_device:
                    start_extra_trans_time = time.time()
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )

                hidden_states = F.linear(
                    hidden_states, stage_cls_decoder_weight, stage_cls_decoder_bias
                )
                end_cal_time = time.time()
                cal_time_list[cal_index] += end_cal_time - start_cal_time

        cal_index += 1

        if cal_index >= total_layer_num:
            break

    torch.cuda.synchronize()
    prediction_scores = hidden_states
    return prediction_scores


def cal_model_cotdm(encoded_input, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    load_index = global_var.load_index_lists[model_idx]
    config = global_var.config_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]
    load_index_len = len(load_index)

    cal_index = 0
    hidden_states = None
    global cal_time_list, extra_trans_time_list, trans_time_list

    total_encoder_layer_num = config.total_encoder_layer_num
    total_layer_num = len(stage_list_gpu)
    while True:
        cal_device = torch.device(use_gpu_list[0])
        for i, device_name in enumerate(use_gpu_list):
            if i < load_index_len - 1 and cal_index < load_index[i + 1]:
                cal_device = torch.device(device_name)
                break

        with stage_cond_list[cal_index]:
            if not is_stage_loaded[cal_index]:
                stage_cond_list[cal_index].wait()

        model_stages = stage_list_gpu[cal_index].stage
        stage_type = stage_list_gpu[cal_index].type

        start_cal_time = time.time()

        with torch.no_grad():
            # if stage_type == 0:
            #     stage_embeddings = model_stages[0]
            #     inf_intermediate_data = bert_inference_preprocess(
            #         stage_embeddings, encoded_input, config
            #     )
            #     embedding_output = bert_embeddings_foward(
            #         stage_embeddings, inf_intermediate_data
            #     )
            #     bert_encoder_preprocess(inf_intermediate_data, config)

            #     inf_intermediate_data.hidden_states = embedding_output

            #     end_cal_time = time.time()
            #     cal_time_list[cal_index] += end_cal_time - start_cal_time

            #     start_trans_time = time.time()

            #     trans_intermediate_data_to_device(inf_intermediate_data, next_cal_device)
            #     trans_intermediate_data_to_device(inf_intermediate_data, cal_device)

            #     end_trans_time = time.time()
            #     trans_time_list[cal_index] += end_trans_time - start_trans_time
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
                if inf_intermediate_data.inputs_embeds.device != cal_device:
                    start_extra_trans_time = time.time()
                    inf_intermediate_data.inputs_embeds = (
                        inf_intermediate_data.inputs_embeds.to(
                            cal_device, non_blocking=True
                        )
                    )
                    inf_intermediate_data.token_type_ids = (
                        inf_intermediate_data.token_type_ids.to(
                            cal_device, non_blocking=True
                        )
                    )
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
                bert_token_type_embeddings_forward(
                    stage_token_type_embeddings, inf_intermediate_data
                )
            elif stage_type == 2:
                stage_position_embeddings = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device or inf_intermediate_data.position_ids.device != cal_device:
                    start_extra_trans_time = time.time()
                    inf_intermediate_data.hidden_states = (
                        inf_intermediate_data.hidden_states.to(
                            cal_device, non_blocking=True
                        )
                    )
                    inf_intermediate_data.position_ids = (
                        inf_intermediate_data.position_ids.to(
                            cal_device, non_blocking=True
                        )
                    )
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
                bert_position_embeddings_forward(
                    stage_position_embeddings, inf_intermediate_data, config
                )
            elif stage_type == 3:
                stage_embeddings_LayerNorm_or_dropout = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device:
                    start_extra_trans_time = time.time()
                    trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
                bert_embeddings_LayerNorm_or_dropout_forward(
                    stage_embeddings_LayerNorm_or_dropout, inf_intermediate_data
                )
                if cal_index == 4:
                    bert_encoder_preprocess(inf_intermediate_data, config)
            elif stage_type == 4:
                stage_encoder_layer = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device or inf_intermediate_data.extended_attention_mask.device != cal_device:
                    start_extra_trans_time = time.time()
                    trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
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
                    start_extra_trans_time = time.time()
                    trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )
                pooler_outputs = bert_pooler_forward(
                    stage_pooler,
                    encoder_outputs,
                    inf_intermediate_data.return_dict,
                )
                hidden_states = pooler_outputs.last_hidden_state
            elif stage_type == 6:
                stage_cls = model_stages[0]
                if hidden_states.device != cal_device:
                    start_extra_trans_time = time.time()
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )

                hidden_states = bert_hidden_states_forward(stage_cls, hidden_states)
            elif stage_type == 7:
                stage_cls_decoder_bias = model_stages[0]
                stage_cls_decoder_weight = (
                    stage_list_gpu[0].stage[0].state_dict()["weight"]
                )
                cal_device = torch.device(global_var.model_use_gpu_lists[model_idx][0])

                if hidden_states.device != cal_device:
                    start_extra_trans_time = time.time()
                    hidden_states = hidden_states.to(cal_device, non_blocking=True)
                    end_extra_trans_time = time.time()
                    extra_trans_time_list[cal_index] += (
                        end_extra_trans_time - start_extra_trans_time
                    )

                hidden_states = F.linear(
                    hidden_states, stage_cls_decoder_weight, stage_cls_decoder_bias
                )

        end_cal_time = time.time()
        cal_time_list[cal_index] += end_cal_time - start_cal_time

        cal_index += 1

        if cal_index >= total_layer_num:
            break

    torch.cuda.synchronize()
    prediction_scores = hidden_states
    return prediction_scores


def load_model(begin_idx, end_idx, device, model_idx):
    global load_time_list

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

            start_load_time = time.time()

            for j in range(len(stage_list_gpu[i].stage)):
                stage_list_gpu[i].stage[j] = (
                    stage_list_gpu[i].stage[j].to(device, non_blocking=True)
                )
                if i != 0 or j == 0:
                    stage_list_gpu[i].stage[j].eval()
            is_stage_loaded[i] = True
            stage_cond_list[i].notify()

            end_load_time = time.time()
            load_time_list[i] += end_load_time - start_load_time

    if device == use_gpu_list[0]:
        with stage_cond_list[-1]:
            if is_stage_loaded[-1] == True:
                stage_cond_list[-1].notify()
                return

            start_load_time = time.time()

            for i in range(len(stage_list_gpu[-1].stage)):
                stage_list_gpu[-1].stage[i] = (
                    stage_list_gpu[-1].stage[i].to(device, non_blocking=True)
                )
            is_stage_loaded[-1] = True
            stage_cond_list[-1].notify()

            end_load_time = time.time()
            load_time_list[-1] += end_load_time - start_load_time


def cal_func_ready(encoded_input, stream, model_idx):
    with torch.cuda.stream(stream):
        outputs = cal_model_ready(encoded_input, model_idx)
    return outputs


def cal_func_cotdm(encoded_input, stream, model_idx):
    with torch.cuda.stream(stream):
        outputs = cal_model_cotdm(encoded_input, model_idx)
    return outputs


def load_func(stream, begin_idx, end_idx, device, model_idx):
    with torch.cuda.stream(stream):
        load_model(begin_idx, end_idx, device, model_idx)


def inference_func_get_time_list_ready(encoded_input, model_idx):
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    load_index = global_var.load_index_lists[model_idx]
    tokenizer = global_var.tokenizer_lists[model_idx]

    load_stream = torch.cuda.Stream()
    load_thread = Thread(
        target=load_func,
        args=(
            load_stream,
            load_index[0],
            load_index[1],
            use_gpu_list[0],
            model_idx,
        ),
    )
    load_thread.start()
    load_thread.join()

    start_inf_time = time.time()

    cal_stream = torch.cuda.Stream()
    cal_thread = ThreadWithReturnValue(
        target=cal_func_ready, args=(encoded_input, cal_stream, model_idx)
    )

    cal_thread.start()
    prediction_scores = cal_thread.join()
    prediction_token = get_predicted_token(prediction_scores, encoded_input, tokenizer)

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time

    return inf_lat


def inference_func_get_time_list_cotdm(encoded_input, model_idx):
    start_inf_time = time.time()
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    tokenizer = global_var.tokenizer_lists[model_idx]

    cal_stream = torch.cuda.Stream()
    cal_thread = ThreadWithReturnValue(
        target=cal_func_cotdm, args=(encoded_input, cal_stream, model_idx)
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

    cal_thread.start()
    if load_index_list[0] != -1:
        for load_thread in load_threads:
            load_thread.start()

    prediction_scores = cal_thread.join()
    predicted_token = get_predicted_token(prediction_scores, encoded_input, tokenizer)

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    ttft = inf_lat
    
    return predicted_token, inf_lat, ttft


def inference_by_deploy_scheme(input_text, model_idx):
    global_var.cur_request_num.incrementAndGet()
    if global_config.DEBUG_MODE_SCHEDULER:
        print("global var cur_request_num increase:", global_var.cur_request_num.get())

    inference_func = inference_func_get_time_list_cotdm

    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    encoded_input = global_var.tokenizer_lists[model_idx](
        input_text, return_tensors="pt"
    ).to(use_gpu_list[0])
    global_var.model_request_num_list[model_idx] += 1
    global_var.is_model_using_lists[model_idx].wait()
    result, inf_lat, ttft = inference_func(encoded_input, model_idx)

    if global_config.DEBUG_MODE_SCHEDULER:
        print("global var cur_request_num:", global_var.cur_request_num.get())
    if global_var.cur_request_num.decrementAndGet() == 0:
        if global_config.DEBUG_MODE_SCHEDULER:
            print("global var cur_request_num decrease:", global_var.cur_request_num.get())
        global_var.scheduler_inc_sem.release()

    global_var.model_request_num_list[model_idx] -= 1

    return result, inf_lat, ttft


async def inference_then_offload(input_text, model_name, slo=1.0):
    model_idx = await get_model_idx(model_name, slo)
    result, inf_lat, ttft = inference_by_deploy_scheme(input_text, model_idx)

    offload_thread = Thread(target=offload_stages_from_gpu_to_cpu, args=(model_idx,))
    offload_thread.start()

    offload_start_time = time.time()
    offload_thread.join()
    offload_end_time = time.time()

    offload_time = offload_end_time - offload_start_time

    return result, inf_lat, ttft, offload_time


async def handle_inference_with_offload(request):
    data = await request.json()
    model_name = data.get("model_name", None)
    input_text = data.get("input_text", None)
    if model_name is None:
        return web.json_response({"text": "Model name is empty!", "inf_lat": -1, "ttft": -1, "offload_time": -1})

    if input_text is None:
        return web.json_response({"text": "Input is empty!", "inf_lat": -1, "ttft": -1, "offload_time": -1})

    result, inf_lat, ttft, offload_time = await inference_then_offload(input_text, model_name)
    if result is None:
        return web.json_response({"text": "Inference failed!", "inf_lat": -1, "ttft": -1, "offload_time": -1})

    data = {"result": result, "inf_lat": inf_lat, "ttft": ttft, "offload_time": offload_time}
    return web.json_response(data)


async def start_server():
    app = web.Application()
    app.router.add_post("/inference", handle_inference_with_offload)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "127.0.0.1", SUBMIT_PORT)
    await site.start()

    return runner, site


async def close_server(server_site, server_runner, model_idx):
    offload_model_from_gpu_to_cpu(model_idx)
    await server_site.stop()
    await server_runner.cleanup()


def save_splitted_model_bert(model_masked_lm, config, model_path, is_save):
    stage_list_cpu = []

    # embeddings_construct = BertEmbeddings(config)
    # embeddings_checkpoint = model_masked_lm.bert.embeddings.state_dict()
    # embeddings_construct.load_state_dict(embeddings_checkpoint)
    # stage_list_cpu.append(ModelStage([embeddings_construct], 0))
    # torch.save(embeddings_checkpoint, model_path + "/embeddings.pth")

    word_embeddings_construct = nn.Embedding(
        config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
    )
    word_embeddings_checkpoint = (
        model_masked_lm.bert.embeddings.word_embeddings.state_dict()
    )
    word_embeddings_construct.load_state_dict(word_embeddings_checkpoint)
    stage_embeddings_position_ids = torch.arange(config.max_position_embeddings).expand(
        (1, -1)
    )
    stage_embeddings_token_type_ids = torch.zeros(
        stage_embeddings_position_ids.size(), dtype=torch.long
    )
    stage_list_cpu.append(
        ModelStage(
            [
                word_embeddings_construct,
                stage_embeddings_position_ids,
                stage_embeddings_token_type_ids,
            ],
            0,
        )
    )
    if is_save:
        torch.save(word_embeddings_checkpoint, model_path + "/word_embeddings.pth")

    token_type_embeddings_construct = nn.Embedding(
        config.type_vocab_size, config.hidden_size
    )
    token_type_embeddings_checkpoint = (
        model_masked_lm.bert.embeddings.token_type_embeddings.state_dict()
    )
    token_type_embeddings_construct.load_state_dict(token_type_embeddings_checkpoint)
    stage_list_cpu.append(ModelStage([token_type_embeddings_construct], 1))
    if is_save:
        torch.save(
            token_type_embeddings_checkpoint, model_path + "/token_type_embeddings.pth"
        )

    position_embeddings_construct = nn.Embedding(
        config.max_position_embeddings, config.hidden_size
    )
    position_embeddings_checkpoint = (
        model_masked_lm.bert.embeddings.position_embeddings.state_dict()
    )
    position_embeddings_construct.load_state_dict(position_embeddings_checkpoint)
    stage_list_cpu.append(ModelStage([position_embeddings_construct], 2))
    if is_save:
        torch.save(position_embeddings_checkpoint, model_path + "/position_embeddings.pth")

    embeddings_LayerNorm_construct = nn.LayerNorm(
        config.hidden_size, eps=config.layer_norm_eps
    )
    embeddings_LayerNorm_checkpoint = (
        model_masked_lm.bert.embeddings.LayerNorm.state_dict()
    )
    embeddings_LayerNorm_construct.load_state_dict(embeddings_LayerNorm_checkpoint)
    stage_list_cpu.append(ModelStage([embeddings_LayerNorm_construct], 3))
    if is_save:
        torch.save(
            embeddings_LayerNorm_checkpoint, model_path + "/embeddings_LayerNorm.pth"
        )

    embeddings_dropout_construct = nn.Dropout(config.hidden_dropout_prob)
    embeddings_dropout_checkpoint = model_masked_lm.bert.embeddings.dropout.state_dict()
    embeddings_dropout_construct.load_state_dict(embeddings_dropout_checkpoint)
    stage_list_cpu.append(ModelStage([embeddings_dropout_construct], 3))
    if is_save:
        torch.save(embeddings_dropout_checkpoint, model_path + "/embeddings_dropout.pth")

    num_hidden_layers = config.num_hidden_layers
    total_encoder_layer_num = config.total_encoder_layer_num
    for i in range(num_hidden_layers):
        layer_construct = BertLayer(config)
        layer_checkpoint = model_masked_lm.bert.encoder.layer[i].state_dict()
        layer_construct.load_state_dict(layer_checkpoint)
        stage_list_cpu.append(ModelStage([layer_construct], 4))
        if is_save:
            torch.save(
                layer_checkpoint,
                model_path + "/encoder_layer_{}.pth".format(i),
            )

    if total_encoder_layer_num > num_hidden_layers:
        for i in range(num_hidden_layers, total_encoder_layer_num):
            layer_construct = BertLayer(config)
            layer_checkpoint = torch.load(
                model_path + "/encoder_layer_{}.pth".format(i % num_hidden_layers),
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
            layer_construct.load_state_dict(layer_checkpoint)
            stage_list_cpu.append(ModelStage([layer_construct], 4))

    config.add_pooling_layer = False
    if model_masked_lm.bert.pooler is not None:
        pooler_construct = BertPooler(config)
        pooler_checkpoint = model_masked_lm.bert.pooler.state_dict()
        pooler_construct.load_state_dict(pooler_checkpoint)
        stage_list_cpu.append(ModelStage([pooler_construct], 5))
        if is_save:
            torch.save(pooler_checkpoint, model_path + "/pooler.pth")
        config.add_pooling_layer = True

    dense_construct = nn.Linear(config.hidden_size, config.hidden_size)
    dense_checkpoint = model_masked_lm.cls.predictions.transform.dense.state_dict()
    dense_construct.load_state_dict(dense_checkpoint)
    stage_list_cpu.append(ModelStage([dense_construct], 6))
    if is_save:
        torch.save(dense_checkpoint, model_path + "/transform_dense.pth")

    if isinstance(config.hidden_act, str):
        transform_act_fn_construct = ACT2FN[config.hidden_act]
    else:
        transform_act_fn_construct = config.hidden_act
    transform_act_fn_checkpoint = (
        model_masked_lm.cls.predictions.transform.transform_act_fn.state_dict()
    )
    transform_act_fn_construct.load_state_dict(transform_act_fn_checkpoint)
    stage_list_cpu.append(ModelStage([transform_act_fn_construct], 6))
    if is_save:
        torch.save(transform_act_fn_checkpoint, model_path + "/transform_act_fn.pth")

    transform_LayerNorm_construct = nn.LayerNorm(
        config.hidden_size, eps=config.layer_norm_eps
    )
    transform_LayerNorm_checkpoint = (
        model_masked_lm.cls.predictions.transform.LayerNorm.state_dict()
    )
    transform_LayerNorm_construct.load_state_dict(transform_LayerNorm_checkpoint)
    stage_list_cpu.append(ModelStage([transform_LayerNorm_construct], 6))
    if is_save:
        torch.save(transform_LayerNorm_checkpoint, model_path + "/transform_LayerNorm.pth")

    decoder_bias_checkpoint = model_masked_lm.cls.predictions.decoder.state_dict()[
        "bias"
    ]
    stage_list_cpu.append(ModelStage([decoder_bias_checkpoint], 7))
    if is_save:
        torch.save(decoder_bias_checkpoint, model_path + "/decoder_bias.pth")

    return stage_list_cpu


def deploy_bert_model_by_load_index(stage_list_cpu, load_index, deploy_device):
    deploy_params_num = load_index[0]
    total_layer_num = len(stage_list_cpu)
    stage_list_gpu = []

    before_memory, after_memory = 0, 0
    for i in range(len(global_var.cuda_devices)):
        before_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])

    is_stage_loaded = [False for i in range(total_layer_num)]
    if deploy_params_num > 0:
        stages_gpu = []
        for i in range(len(stage_list_cpu[0].stage)):
            stage_gpu = stage_list_cpu[0].stage[i].to(deploy_device)
            if i == 0:
                stage_gpu.eval()
            stages_gpu.append(stage_gpu)
        stage_list_gpu.append(ModelStage(stages_gpu, stage_list_cpu[0].type))
        is_stage_loaded[0] = True
    for i in range(1, deploy_params_num):
        stages_gpu = []
        for j in range(len(stage_list_cpu[i].stage)):
            stage_gpu = stage_list_cpu[i].stage[j].to(deploy_device)
            if i < total_layer_num - 1:
                stage_gpu.eval()
            stages_gpu.append(stage_gpu)
        stage_list_gpu.append(ModelStage(stages_gpu, stage_list_cpu[i].type))
        is_stage_loaded[i] = True

    for i in range(deploy_params_num, total_layer_num):
        stage_list_gpu.append(
            ModelStage(stage_list_cpu[i].stage, stage_list_cpu[i].type)
        )

    for i in range(len(global_var.cuda_devices)):
        after_memory += torch.cuda.memory_allocated(global_var.cuda_devices[i])
    deploy_memory = after_memory - before_memory

    return stage_list_gpu, is_stage_loaded, deploy_memory


async def test_e2e_time(model_name, model_idx, test_time):
    global cal_time_list, extra_trans_time_list, trans_time_list, load_time_list
    total_layer_num = len(global_var.stage_lists_gpu[model_idx])

    server_runner, server_site = await start_server()
    inf_lat_list = []
    ttft_list = []
    e2e_lat_list = []
    for i in range(test_time + 1):
        _, inf_lat, ttft, e2e_lat = await send_single_inf_req(
            model_name, global_config.DEFAULT_INPUT_TEXT_BERT, port=SUBMIT_PORT
        )

        if i == 0:
            cal_time_list = [0 for j in range(total_layer_num)]
            extra_trans_time_list = [0 for j in range(total_layer_num)]
            trans_time_list = [0 for j in range(total_layer_num)]
            load_time_list = [0 for j in range(total_layer_num)]

            continue
        
        inf_lat_list.append(inf_lat)
        ttft_list.append(ttft)
        e2e_lat_list.append(e2e_lat)
    
    await close_server(server_site, server_runner, model_idx)

    avg_inf_lat = sum(inf_lat_list) / len(inf_lat_list) - sum(trans_time_list) / TEST_TIME
    avg_ttft = sum(ttft_list) / len(ttft_list) - sum(trans_time_list) / TEST_TIME
    avg_e2e_lat = sum(e2e_lat_list) / len(e2e_lat_list) - sum(trans_time_list) / TEST_TIME

    return avg_inf_lat, avg_ttft, avg_e2e_lat


async def submit_model_bert(model_name, slo=1.0):
    model_name_slo_idx_dict = global_var.model_name_slo_idx_dict
    model_name_and_slo = ModelNameAndSlo(model_name, slo)
    model_idx = len(global_var.stage_lists_gpu)

    if model_name_and_slo not in model_name_slo_idx_dict:
        model_name_slo_idx_dict[model_name_and_slo] = [model_idx]
    else:
        model_name_slo_idx_dict[model_name_and_slo].append(model_idx)

    global_var.offload_time_list.append(0)
    global_var.model_request_num_list.append(0)
    global cal_time_list, extra_trans_time_list, trans_time_list, load_time_list

    is_save = False
    if model_name.startswith("bert-base-uncased") and len(model_name) != len("bert-base-uncased"):
        model_masked_lm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        config = AutoConfig.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", clean_up_tokenization_spaces=False
        )
        total_encoder_layer_num = config.num_hidden_layers
        model_path = global_config.MODEL_SAVE_DIR + "bert-base-uncased"
    elif model_name.startswith("bert-large-uncased") and len(model_name) != len("bert-large-uncased"):
        model_masked_lm = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")
        config = AutoConfig.from_pretrained("bert-large-uncased")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased", clean_up_tokenization_spaces=False
        )
        total_encoder_layer_num = config.num_hidden_layers
        model_path = global_config.MODEL_SAVE_DIR + "bert-large-uncased"
    elif model_name == "bert-1.2B":
        model_masked_lm = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")
        config = AutoConfig.from_pretrained("bert-large-uncased")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased", clean_up_tokenization_spaces=False
        )
        # total_encoder_layer_num = 100
        total_encoder_layer_num = config.num_hidden_layers * 4
        model_path = global_config.MODEL_SAVE_DIR + "bert-large-uncased"
    elif model_name == "bert-2.5B":
        model_masked_lm = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")
        config = AutoConfig.from_pretrained("bert-large-uncased")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased", clean_up_tokenization_spaces=False
        )
        # total_encoder_layer_num = 200
        total_encoder_layer_num = config.num_hidden_layers * 8
        model_path = global_config.MODEL_SAVE_DIR + "bert-large-uncased"
    elif model_name == "bert-6.7B":
        model_masked_lm = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")
        config = AutoConfig.from_pretrained("bert-large-uncased")
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-large-uncased", clean_up_tokenization_spaces=False
        )
        # total_encoder_layer_num = 550
        total_encoder_layer_num = config.num_hidden_layers * 22
        model_path = global_config.MODEL_SAVE_DIR + "bert-large-uncased"
    else:
        model_masked_lm = AutoModelForMaskedLM.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=False
        )
        total_encoder_layer_num = config.num_hidden_layers
        model_path = global_config.MODEL_SAVE_DIR + model_name

    if not os.path.exists(model_path):
        is_save = True
        os.makedirs(model_path)

    global_var.tokenizer_lists.append(tokenizer)
    if is_save:
        tokenizer.save_pretrained(model_path)

    config.total_encoder_layer_num = total_encoder_layer_num
    model_masked_lm = model_masked_lm.cpu()

    stage_list_cpu = save_splitted_model_bert(model_masked_lm, config, model_path, is_save)
    global_var.stage_lists_cpu.append(stage_list_cpu)

    print("Submit model name:", model_name)
    parameters_num = count_parameters(stage_list_cpu)
    print_parameters_num(parameters_num)
    stage_mem_list = get_stage_mem_list(stage_list_cpu)
    global_var.model_stage_mem_list.append(stage_mem_list)
    model_size = sum(stage_mem_list)
    print_model_size(model_size)

    total_layer_num = len(stage_list_cpu)
    config._name_or_path = model_path
    config.total_layer_num = total_layer_num
    config.model_type = "bert"
    if is_save:
        config.save_pretrained(model_path)
    global_var.config_lists.append(config)
    global_var.model_stage_cond_lists.append(
        [threading.Condition(threading.Lock()) for i in range(total_layer_num)]
    )

    use_gpu_list = [global_var.cuda_devices[0]]
    global_var.model_use_gpu_lists.append(use_gpu_list)
    deploy_device = use_gpu_list[0]
    input_text = global_config.DEFAULT_INPUT_TEXT_BERT
    encoded_input = tokenizer(input_text, return_tensors="pt").to(deploy_device)

    load_index = [0, total_layer_num]
    # load_index = []
    # device_count = len(global_var.cuda_devices)
    # for i in range(device_count + 1):
    #     load_index.append(int(total_layer_num * i / device_count))
    global_var.load_index_lists.append(load_index)

    stage_list_gpu, is_stage_loaded, _ = deploy_bert_model_by_load_index(
        stage_list_cpu, load_index, deploy_device
    )
    global_var.stage_lists_gpu.append(stage_list_gpu)
    global_var.is_stage_loaded_lists.append(is_stage_loaded)

    cal_time_list = [0 for i in range(total_layer_num)]
    extra_trans_time_list = [0 for i in range(total_layer_num)]
    trans_time_list = [0 for i in range(total_layer_num)]
    load_time_list = [0 for i in range(total_layer_num)]

    for i in range(TEST_TIME + 1):
        inference_func_get_time_list_ready(encoded_input, model_idx)
        offload_model_from_gpu_to_cpu(model_idx)

        if i == 0:
            cal_time_list = [0 for j in range(total_layer_num)]
            extra_trans_time_list = [0 for j in range(total_layer_num)]
            trans_time_list = [0 for j in range(total_layer_num)]
            load_time_list = [0 for j in range(total_layer_num)]

    cal_time_list_ready = [(cal_time_list[i] - extra_trans_time_list[i]) / TEST_TIME for i in range(len(cal_time_list))]
    trans_time_list_ready = [trans_time / (2 * TEST_TIME) for trans_time in trans_time_list]
    load_time_list_ready = [load_time / TEST_TIME for load_time in load_time_list]

    is_model_using = threading.Event()
    is_model_using.set()
    global_var.is_model_using_lists.append(is_model_using)
    is_model_waiting_offload = threading.Event()
    is_model_waiting_offload.clear()
    global_var.is_model_waiting_offload_lists.append(is_model_waiting_offload)

    cal_time_threadnum_lists = []
    trans_time_threadnum_lists = []
    load_time_threadnum_lists = []
    extra_e2e_time_list = []
    single_layer_time_ready = SingleLayerTime(cal_time_list_ready, trans_time_list_ready, load_time_list_ready)

    load_index = [total_layer_num, total_layer_num]
    global_var.load_index_lists[model_idx] = load_index
    stage_list_gpu, is_stage_loaded, _ = deploy_bert_model_by_load_index(stage_list_cpu, load_index, deploy_device)
    global_var.stage_lists_gpu[model_idx] = stage_list_gpu
    global_var.is_stage_loaded_lists[model_idx] = is_stage_loaded

    _, _, avg_e2e_lat_ready = await test_e2e_time(model_name, model_idx, TEST_TIME)
    print("average end to end latency(Fully deployed): {:.4f}s".format(avg_e2e_lat_ready))

    default_use_gpu_num = len(global_var.cuda_devices)
    ideal_inf_lat, load_index = cal_lat_by_deploynum_and_avaigpu(
        0, total_layer_num, default_use_gpu_num, single_layer_time_ready
    )
    use_gpu_list = get_single_model_placement(stage_mem_list, load_index)
    global_var.load_index_lists[model_idx] = load_index
    global_var.model_use_gpu_lists[model_idx] = use_gpu_list
    
    _, _, avg_e2e_lat = await test_e2e_time(model_name, model_idx, TEST_TIME)

    cal_time_threadnum = [(cal_time_list[i] - extra_trans_time_list[i]) / TEST_TIME for i in range(len(cal_time_list))]
    trans_time_threadnum = [trans_time / (2 * TEST_TIME) for trans_time in trans_time_list]
    load_time_threadnum = [load_time / TEST_TIME for load_time in load_time_list]

    cal_time_threadnum_lists.append(cal_time_threadnum)
    trans_time_threadnum_lists.append(trans_time_threadnum)
    load_time_threadnum_lists.append(load_time_threadnum)

    single_layer_time_threadnum = SingleLayerTime(cal_time_threadnum_lists[-1], trans_time_threadnum_lists[-1], load_time_threadnum_lists[-1])
    global_var.single_layer_time_list.append(single_layer_time_threadnum)
    ideal_inf_lat, _ = cal_lat_by_deploynum_and_avaigpu(
        0, total_layer_num, default_use_gpu_num, single_layer_time_threadnum
    )
    # extra_e2e_time = avg_e2e_lat - sum(cal_time_threadnum_lists[-1])
    extra_e2e_time = avg_e2e_lat - ideal_inf_lat
    if extra_e2e_time < 0:
        extra_e2e_time = 0
    extra_e2e_time_list.append(extra_e2e_time)

    slo_scale = 8
    slo = avg_e2e_lat_ready * slo_scale - extra_e2e_time
    global_var.model_slo_list.append(slo)
    load_index = min_deploy(total_layer_num, slo, i + 1, single_layer_time_threadnum)
    use_gpu_list = get_single_model_placement(stage_mem_list, load_index)
    global_var.load_index_lists[model_idx] = load_index
    global_var.model_use_gpu_lists[model_idx] = use_gpu_list
    stage_list_gpu, is_stage_loaded, _ = deploy_bert_model_by_load_index(stage_list_cpu, load_index, use_gpu_list[0])
    global_var.stage_lists_gpu[model_idx] = stage_list_gpu
    global_var.is_stage_loaded_lists[model_idx] = is_stage_loaded
    _, _, avg_e2e_lat = await test_e2e_time(model_name, model_idx, TEST_TIME)

    for i in range(len(use_gpu_list)):
        device_idx = int(use_gpu_list[i][5:])
        begin_idx = load_index[i]
        end_idx = load_index[i + 1]
        if i == 0:
            begin_idx = 0

        global_var.memory_fully_loaded_list[device_idx] += sum(stage_mem_list[begin_idx:end_idx])

    stage_list_gpu, is_stage_loaded, _ = deploy_bert_model_by_load_index(stage_list_cpu, load_index, use_gpu_list[0])
    global_var.stage_lists_gpu[model_idx] = stage_list_gpu
    global_var.is_stage_loaded_lists[model_idx] = is_stage_loaded

    config_dict = {}
    config_dict["path"] = model_path
    config_dict["name"] = model_name
    config_dict["type"] = "bert"
    config_dict["total_layer_num"] = total_layer_num
    config_dict["avg_e2e_lat_ready"] = avg_e2e_lat_ready
    config_dict["extra_e2e_time"] = extra_e2e_time_list
    config_dict["single_layer_cal_time"] = cal_time_threadnum_lists
    config_dict["single_layer_trans_time"] = trans_time_threadnum_lists
    config_dict["single_layer_load_time"] = load_time_threadnum_lists
    config_dict["single_layer_time"] = {
        "cal_time": cal_time_list_ready,
        "trans_time": trans_time_list_ready,
        "load_time": load_time_list_ready,
    }
    config_dict["stage_mem_list"] = stage_mem_list

    model_name_and_slo = model_name + "_" + str(round(slo, 3)).replace(".", "_")

    if not os.path.exists(global_config.MODEL_CONFIG_DIR):
        os.makedirs(global_config.MODEL_CONFIG_DIR)
    
    config_path = global_config.MODEL_CONFIG_DIR + model_name_and_slo + ".json"
    with open(config_path, "w") as write_f:
        json.dump(config_dict, write_f, indent=4, ensure_ascii=False)

    print("Model save path:", model_path)
    print("Info config save path:", config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="bert-large-uncased")
    args = parser.parse_args()
    model_name = args.name
    asyncio.run(submit_model_bert(model_name))
