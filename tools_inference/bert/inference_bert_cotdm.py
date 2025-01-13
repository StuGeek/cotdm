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

DEBUG_MODE = False

def trans_intermediate_data_to_device(inf_intermediate_data, cal_device):
    if inf_intermediate_data.hidden_states != None:
        inf_intermediate_data.hidden_states = inf_intermediate_data.hidden_states.to(
            cal_device, non_blocking=True
        )

    if inf_intermediate_data.all_hidden_states != None:
        inf_intermediate_data.all_hidden_states = inf_intermediate_data.all_hidden_states.to(
            cal_device, non_blocking=True,
        )

    if inf_intermediate_data.extended_attention_mask != None:
        inf_intermediate_data.extended_attention_mask = (
            inf_intermediate_data.extended_attention_mask.to(
                cal_device, non_blocking=True
            )
        )

    if inf_intermediate_data.encoder_hidden_states != None:
        inf_intermediate_data.encoder_hidden_states = (
            inf_intermediate_data.encoder_hidden_states.to(
                cal_device, non_blocking=True
            )
        )

    if inf_intermediate_data.encoder_extended_attention_mask != None:
        inf_intermediate_data.encoder_extended_attention_mask = (
            inf_intermediate_data.encoder_extended_attention_mask.to(
                cal_device, non_blocking=True
            )
        )


def cal_model(encoded_input, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    load_index = global_var.load_index_lists[model_idx]
    config = global_var.config_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]
    load_index_len = len(load_index)

    cal_index = 0

    total_encoder_layer_num = config.total_encoder_layer_num
    total_layer_num = config.total_layer_num
    while True:
        cal_device = torch.device(use_gpu_list[0])
        for i, device_name in enumerate(use_gpu_list):
            if i < load_index_len - 1 and cal_index < load_index[i + 1]:
                cal_device = torch.device(device_name)
                break

        if cal_index == total_layer_num - 1:
            cal_device = torch.device(use_gpu_list[0])

        with stage_cond_list[cal_index]:
            if not is_stage_loaded[cal_index]:
                stage_cond_list[cal_index].wait()

        model_stages = stage_list_gpu[cal_index].stage
        stage_type = stage_list_gpu[cal_index].type

        # print("cal index:", cal_index)

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
                if inf_intermediate_data.inputs_embeds.device != cal_device:
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
                bert_token_type_embeddings_forward(
                    stage_token_type_embeddings, inf_intermediate_data
                )
            elif stage_type == 2:
                stage_position_embeddings = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device or inf_intermediate_data.position_ids.device != cal_device:
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
                bert_position_embeddings_forward(
                    stage_position_embeddings, inf_intermediate_data, config
                )
            elif stage_type == 3:
                stage_embeddings_LayerNorm_or_dropout = model_stages[0]
                if inf_intermediate_data.hidden_states.device != cal_device:
                    trans_intermediate_data_to_device(inf_intermediate_data, cal_device)
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
                
                # print(hidden_states.device, stage_cls_decoder_weight.device, stage_cls_decoder_bias.device)
                hidden_states = F.linear(
                    hidden_states, stage_cls_decoder_weight, stage_cls_decoder_bias
                )

        cal_index += 1

        if DEBUG_MODE:
            print(
                cal_device,
                "cal index:",
                cal_index - 1,
                time.time() - global_var.inf_start_time,
                "time3",
            )

        if cal_index >= total_layer_num:
            break

    torch.cuda.synchronize()

    prediction_scores = hidden_states
    return prediction_scores


def load_model(begin_idx, end_idx, device, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    total_layer_num = len(stage_list_gpu)

    inf_start_time = global_var.inf_start_time

    if end_idx == total_layer_num:
        end_idx -= 1

    for i in range(begin_idx, end_idx):
        with stage_cond_list[i]:

            if DEBUG_MODE:
                load_begin_time = time.time() - inf_start_time
                print(device, "load index begin:", i, load_begin_time)

            if is_stage_loaded[i] == True:
                continue

            for j in range(len(stage_list_gpu[i].stage)):
                stage_list_gpu[i].stage[j] = (
                    stage_list_gpu[i].stage[j].to(device, non_blocking=True)
                )
                # print(i, "load time1: ", time.time() - start, start)
                if i != 0 or j == 0:
                    stage_list_gpu[i].stage[j].eval()

            is_stage_loaded[i] = True
            stage_cond_list[i].notify()

            if DEBUG_MODE:
                load_end_time = time.time() - inf_start_time
                print(device, "load index end:", i, load_end_time)

    if device == use_gpu_list[0]:
        with stage_cond_list[-1]:

            if DEBUG_MODE:
                load_begin_time = time.time() - inf_start_time
                print(device, "load index begin:", len(stage_cond_list), load_begin_time)


            if is_stage_loaded[-1] == True:
                stage_cond_list[-1].notify()
                if DEBUG_MODE:
                    load_begin_time = time.time() - inf_start_time
                    print(device, "load index begin:", len(stage_cond_list), load_begin_time)

                return

            # start = time.time()
            for i in range(len(stage_list_gpu[-1].stage)):
                stage_list_gpu[-1].stage[i] = (
                    stage_list_gpu[-1].stage[i].to(device, non_blocking=True)
                )
                # print(i, "load time1: ", time.time() - start, start)
                # stage_list_gpu[-1].stage[i].eval()
            is_stage_loaded[-1] = True
            stage_cond_list[-1].notify()
            # print(i, "load time2: ", time.time() - start, start)
            if DEBUG_MODE:
                load_end_time = time.time() - inf_start_time
                print(device, "load index end:", len(stage_cond_list), load_end_time)


def cal_func(encoded_input, stream, model_idx):
    with torch.cuda.stream(stream):
        outputs = cal_model(encoded_input, model_idx)
    return outputs


def load_func(stream, begin_idx, end_idx, device, model_idx):
    with torch.cuda.stream(stream):
        load_model(begin_idx, end_idx, device, model_idx)


def inference_bert_cotdm(encoded_input, model_idx):
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
    # print(load_index_list)
    # print(use_gpu_list)
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
    cal_thread.start()

    prediction_scores = cal_thread.join()
    # if load_index_list[0] != -1:
    #     for load_thread in load_threads:
    #         load_thread.join()

    predicted_token = get_predicted_token(prediction_scores, encoded_input, tokenizer)

    end_inf_time = time.time()
    inf_lat = end_inf_time - start_inf_time
    ttft = inf_lat
    return predicted_token, inf_lat, ttft


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer
    from init.init_tools import load_model_from_disk_bert
    import threading
    import global_data.global_config as global_config
    from global_data.global_class import ModelStage
    from offload_tools import offload_stages_from_gpu_to_cpu

    # bert_model_name = "bert-base-uncased"
    bert_model_name = "bert-large-uncased"
    model_path = global_config.MODEL_SAVE_DIR + bert_model_name

    input_text = global_config.DEFAULT_INPUT_TEXT_BERT
    config = AutoConfig.from_pretrained(model_path)
    config._name_or_path = model_path
    # config.total_encoder_layer_num = config.num_hidden_layers * 15
    config.total_encoder_layer_num = config.num_hidden_layers
    # config.total_layer_num = config.total_encoder_layer_num + 5
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenization_spaces=False
    )
    stage_list_cpu = load_model_from_disk_bert(config)
    total_layer_num = len(stage_list_cpu)
    use_gpu_list = ["cuda:0"]

    is_stage_loaded = [False for i in range(total_layer_num)]
    load_index = [total_layer_num, total_layer_num]
    load_index = [5, total_layer_num]
    deploy_params_num = load_index[0]
    encoded_input = tokenizer(input_text, return_tensors="pt").to(use_gpu_list[0])
    stage_cond_list = [
        threading.Condition(threading.Lock()) for i in range(total_layer_num)
    ]
    stage_list_gpu = []
    if deploy_params_num > 0:
        stages_gpu = []
        for i in range(len(stage_list_cpu[0].stage)):
            stage_gpu = stage_list_cpu[0].stage[i].to(use_gpu_list[0])
            if i == 0:
                stage_gpu.eval()
            stages_gpu.append(stage_gpu)
        stage_list_gpu.append(ModelStage(stages_gpu, stage_list_cpu[0].type))
        is_stage_loaded[0] = True
    for i in range(1, deploy_params_num):
        stages_gpu = []
        for j in range(len(stage_list_cpu[i].stage)):
            stage_gpu = stage_list_cpu[i].stage[j].to(use_gpu_list[0])
            if i < total_layer_num - 1:
                stage_gpu.eval()
            stages_gpu.append(stage_gpu)
        stage_list_gpu.append(ModelStage(stages_gpu, stage_list_cpu[i].type))
        with stage_cond_list[i]:
            is_stage_loaded[i] = True
            stage_cond_list[i].notify()

    for i in range(deploy_params_num, total_layer_num):
        stage_list_gpu.append(
            ModelStage(stage_list_cpu[i].stage, stage_list_cpu[i].type)
        )

    global_var.stage_lists_gpu = [stage_list_gpu]
    global_var.is_stage_loaded_lists = [is_stage_loaded]
    global_var.model_use_gpu_lists = [use_gpu_list]
    global_var.load_index_lists = [load_index]
    global_var.config_lists = [config]
    global_var.tokenizer_lists = [tokenizer]
    global_var.model_stage_cond_lists = [stage_cond_list]
    global_var.offload_time_list = [0]

    predicted_token, inf_lat, ttft = inference_bert_cotdm(encoded_input, 0)
    offload_stages_from_gpu_to_cpu(0)
    predicted_token, inf_lat, ttft = inference_bert_cotdm(encoded_input, 0)

    # print("Deploy scheme: cotdm")
    print("input:", input_text)
    print("output:", predicted_token)
    print("inference latency: {:.4f}".format(inf_lat))

    global_var.is_system_running = False
