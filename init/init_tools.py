import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertLayer,
    BertPooler,
)
from transformers.activations import ACT2FN
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import os
import json
import threading
import sys
sys.path.append(".")
sys.path.append("..")
from global_data.global_class import ModelStage, ModelNameAndSlo, SingleLayerTime
from global_data import global_config, global_var
from offload_tools import offload_stages_from_gpu_to_cpu
from submit.submit_model_bert import submit_model_bert
from submit.submit_model_gpt2 import submit_model_gpt2
from scheduler import start_scheduler
import string
from scheduler import min_deploy


def load_model_from_disk_bert(config):
    stage_list = []

    model_path = config._name_or_path
    num_hidden_layers = config.num_hidden_layers
    total_encoder_layer_num = config.total_encoder_layer_num

    # embeddings_construct = BertEmbeddings(config)
    # embeddings_checkpoint = torch.load(
    #     model_path + "/embeddings.pth",
    #     map_location=lambda storage, loc: storage,
    #     weights_only=False,
    # )
    # embeddings_construct.load_state_dict(embeddings_checkpoint)
    # stage_list.append(ModelStage([embeddings_construct], 0))

    word_embeddings_construct = nn.Embedding(
        config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
    )
    word_embeddings_checkpoint = torch.load(
        model_path + "/word_embeddings.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    word_embeddings_construct.load_state_dict(word_embeddings_checkpoint)
    stage_embeddings_position_ids = torch.arange(config.max_position_embeddings).expand(
        (1, -1)
    )
    stage_embeddings_token_type_ids = torch.zeros(
        stage_embeddings_position_ids.size(), dtype=torch.long
    )
    stage_list.append(
        ModelStage(
            [
                word_embeddings_construct,
                stage_embeddings_position_ids,
                stage_embeddings_token_type_ids,
            ],
            0,
        )
    )

    token_type_embeddings_construct = nn.Embedding(
        config.type_vocab_size, config.hidden_size
    )
    token_type_embeddings_checkpoint = torch.load(
        model_path + "/token_type_embeddings.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    token_type_embeddings_construct.load_state_dict(token_type_embeddings_checkpoint)
    stage_list.append(ModelStage([token_type_embeddings_construct], 1))

    position_embeddings_construct = nn.Embedding(
        config.max_position_embeddings, config.hidden_size
    )
    position_embeddings_checkpoint = torch.load(
        model_path + "/position_embeddings.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    position_embeddings_construct.load_state_dict(position_embeddings_checkpoint)
    stage_list.append(ModelStage([position_embeddings_construct], 2))

    embeddings_LayerNorm_construct = nn.LayerNorm(
        config.hidden_size, eps=config.layer_norm_eps
    )
    embeddings_LayerNorm_checkpoint = torch.load(
        model_path + "/embeddings_LayerNorm.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    embeddings_LayerNorm_construct.load_state_dict(embeddings_LayerNorm_checkpoint)
    stage_list.append(ModelStage([embeddings_LayerNorm_construct], 3))

    embeddings_dropout_construct = nn.Dropout(config.hidden_dropout_prob)
    embeddings_dropout_checkpoint = torch.load(
        model_path + "/embeddings_dropout.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    embeddings_dropout_construct.load_state_dict(embeddings_dropout_checkpoint)
    stage_list.append(ModelStage([embeddings_dropout_construct], 3))

    for i in range(num_hidden_layers):
        layer_construct = BertLayer(config)
        # layer_checkpoint = torch.load(
        #     model_path + "/encoder_0.pth".format(i),
        #     map_location=lambda storage, loc: storage,
        #     weights_only=False,
        # )
        layer_checkpoint = torch.load(
            model_path + "/encoder_layer_{}.pth".format(i),
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        layer_construct.load_state_dict(layer_checkpoint)
        stage_list.append(ModelStage([layer_construct], 4))

        # layer_attention_construct = BertAttention(config)
        # layer_attention_checkpoint = layer_construct.attention.state_dict()
        # layer_attention_construct.load_state_dict(layer_attention_checkpoint)
        # torch.save(layer_attention_checkpoint, model_path + "/encoder_layer_attention_{}.pth".format(i))
        # stage_list.append(ModelStage([layer_attention_construct], 1))

        # layer_intermediate_construct = BertIntermediate(config)
        # layer_intermediate_checkpoint = layer_construct.intermediate.state_dict()
        # layer_intermediate_construct.load_state_dict(layer_intermediate_checkpoint)
        # torch.save(layer_intermediate_checkpoint, model_path + "/encoder_intermediate_attention_{}.pth".format(i))

        # layer_output_construct = BertOutput(config)
        # layer_output_checkpoint = layer_construct.output.state_dict()
        # layer_output_construct.load_state_dict(layer_output_checkpoint)
        # torch.save(layer_output_checkpoint, model_path + "/encoder_output_attention_{}.pth".format(i))

        # if config.chunk_size_feed_forward > 0:
        #     stage_list.append(ModelStage([layer_intermediate_construct, layer_output_construct], 2))
        # else:
        #     stage_list.append(ModelStage([layer_intermediate_construct], 3))
        #     stage_list.append(ModelStage([layer_output_construct], 4))

    if total_encoder_layer_num > num_hidden_layers:
       for i in range(num_hidden_layers, total_encoder_layer_num):
            layer_construct = BertLayer(config)
            # layer_checkpoint = torch.load(
            #     model_path + "/encoder_0.pth".format(i),
            #     map_location=lambda storage, loc: storage,
            #     weights_only=False,
            # )
            layer_checkpoint = torch.load(
                model_path + "/encoder_layer_{}.pth".format(i % num_hidden_layers),
                map_location=lambda storage, loc: storage,
                weights_only=False,
            )
            layer_construct.load_state_dict(layer_checkpoint)
            stage_list.append(ModelStage([layer_construct], 4))

    if config.add_pooling_layer:
        pooler_construct = BertPooler(config)
        pooler_checkpoint = torch.load(
            model_path + "/pooler.pth",
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        pooler_construct.load_state_dict(pooler_checkpoint)
        stage_list.append(ModelStage([pooler_construct], 5))

    dense_construct = nn.Linear(config.hidden_size, config.hidden_size)
    dense_checkpoint = torch.load(
        model_path + "/transform_dense.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    dense_construct.load_state_dict(dense_checkpoint)
    stage_list.append(ModelStage([dense_construct], 6))

    if isinstance(config.hidden_act, str):
        transform_act_fn_construct = ACT2FN[config.hidden_act]
    else:
        transform_act_fn_construct = config.hidden_act
    transform_act_fn_checkpoint = torch.load(
        model_path + "/transform_act_fn.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    transform_act_fn_construct.load_state_dict(transform_act_fn_checkpoint)
    stage_list.append(ModelStage([transform_act_fn_construct], 6))

    transform_LayerNorm_construct = nn.LayerNorm(
        config.hidden_size, eps=config.layer_norm_eps
    )
    transform_LayerNorm_checkpoint = torch.load(
        model_path + "/transform_LayerNorm.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    transform_LayerNorm_construct.load_state_dict(transform_LayerNorm_checkpoint)
    stage_list.append(ModelStage([transform_LayerNorm_construct], 6))

    decoder_bias_checkpoint = torch.load(
        model_path + "/decoder_bias.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    stage_list.append(ModelStage([decoder_bias_checkpoint], 7))

    config.total_layer_num = len(stage_list)
    return stage_list


def load_model_from_disk_gpt2(config):
    stage_list = []

    model_path = config._name_or_path
    total_h_layer_num = config.total_h_layer_num

    wte_construct = nn.Embedding(config.vocab_size, config.hidden_size)
    wte_checkpoint = torch.load(
        model_path + "/wte.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    wte_construct.load_state_dict(wte_checkpoint)
    stage_list.append(ModelStage([wte_construct], 0))

    wpe_construct = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    wpe_checkpoint = torch.load(
        model_path + "/wpe.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    wpe_construct.load_state_dict(wpe_checkpoint)
    stage_list.append(ModelStage([wpe_construct], 1))

    drop_construct = nn.Dropout(config.embd_pdrop)
    drop_checkpoint = torch.load(
        model_path + "/drop.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    drop_construct.load_state_dict(drop_checkpoint)
    stage_list.append(ModelStage([drop_construct], 2))

    for i in range(total_h_layer_num):
        h_layer_construct = GPT2Block(config, layer_idx=i)
        h_checkpoint = torch.load(
            model_path + "/h_{}.pth".format(i),
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        h_layer_construct.load_state_dict(h_checkpoint)
        stage_list.append(ModelStage([h_layer_construct], 3))
    # h_layer = GPT2Block(config, layer_idx=0)
    # h_checkpoint = torch.load(model_path + '/h_{}.pth'.format(0), map_location=lambda storage, loc: storage, weights_only=False)
    # h_layer.load_state_dict(h_checkpoint)
    # for i in range(total_h_layer_num):
    #     temp = copy.deepcopy(layer)
    #     stage_list_cpu[1].append(temp)

    ln_f_construct = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    ln_f_checkpoint = torch.load(
        model_path + "/ln_f.pth",
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    ln_f_construct.load_state_dict(ln_f_checkpoint)
    stage_list.append(ModelStage([ln_f_construct], 4))

    config.total_layer_num = len(stage_list)
    return stage_list


def load_model_from_disk_by_name(config):
    model_type = config.model_type

    if model_type == "bert":
        stage_list = load_model_from_disk_bert(config)
    elif model_type == "gpt2":
        stage_list = load_model_from_disk_gpt2(config)
    else:
        raise ValueError(("The model type of {} is not supported.").format(model_type))

    return stage_list


async def load_all_models_to_cpu_from_disk(model_name_list=None):
    models_info = []
    
    if model_name_list is None:
        for filename in os.listdir(global_config.MODEL_CONFIG_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(global_config.MODEL_CONFIG_DIR, filename)
                with open(file_path, "r") as f:
                    json_data = json.load(f)
                    models_info.append(json_data)
    else:
        for model_name in model_name_list:
            model_name_no_tail_num = model_name.rstrip(string.digits)
            if model_name_no_tail_num == "bert-1.2B" or model_name_no_tail_num == "bert-2.5B" or model_name_no_tail_num == "bert-6.7B":
                splitted_model_save_path = global_config.MODEL_SAVE_DIR + "bert-large-uncased"
            else:
                splitted_model_save_path = global_config.MODEL_SAVE_DIR + model_name_no_tail_num
            if not os.path.exists(splitted_model_save_path):
                if model_name_no_tail_num.find("bert") == 0:
                    await submit_model_bert(model_name_no_tail_num)
                elif model_name_no_tail_num.find("gpt2") == 0:
                    await submit_model_gpt2(model_name_no_tail_num)

            for filename in os.listdir(global_config.MODEL_CONFIG_DIR):
                if filename.startswith(model_name_no_tail_num) and filename.endswith(".json"):
                    file_path = os.path.join(global_config.MODEL_CONFIG_DIR, filename)
                    with open(file_path, "r") as f:
                        json_data = json.load(f)
                        models_info.append(json_data)
                    break
    
    global_var.clear_global_var()
    model_num = len(models_info)

    global_var.is_model_using_lists = [threading.Event() for i in range(model_num)]
    global_var.is_model_waiting_offload_lists = [threading.Event() for i in range(model_num)]
    global_var.offload_time_list = [0 for i in range(model_num)]
    global_var.model_request_num_list = [0 for i in range(model_num)]

    for i in range(model_num):
        info = models_info[i]
        model_path = info["path"]
        if model_name_list is None:
            model_name = info["name"]
        else:
            model_name = model_name_list[i]
        avg_e2e_lat_ready = info["avg_e2e_lat_ready"]

        slo = round(avg_e2e_lat_ready * global_config.INIT_SLO_SCALE, 3)

        global_var.model_slo_list.append(slo)

        stage_mem_list = info["stage_mem_list"]
        global_var.model_stage_mem_list.append(stage_mem_list)

        single_layer_time_info = info["single_layer_time"]
        single_layer_time = SingleLayerTime(
            single_layer_time_info["cal_time"],
            single_layer_time_info["trans_time"],
            single_layer_time_info["load_time"],
        )
        global_var.single_layer_time_list.append(single_layer_time)
        
        model_name_and_slo = ModelNameAndSlo(model_name, slo)

        if model_name_and_slo not in global_var.model_name_slo_idx_dict:
            global_var.model_name_slo_idx_dict[model_name_and_slo] = [i]
        else:
            global_var.model_name_slo_idx_dict[model_name_and_slo].append(i)

        config = AutoConfig.from_pretrained(model_path)
        model_name_no_tail_num = model_name.rstrip(string.digits)
        if model_name_no_tail_num == "bert-1.2B":
            config.total_encoder_layer_num = config.num_hidden_layers * 4
        elif model_name_no_tail_num == "bert-2.5B":
            config.total_encoder_layer_num = config.num_hidden_layers * 8
        elif model_name_no_tail_num == "bert-6.7B":
            config.total_encoder_layer_num = config.num_hidden_layers * 22
        global_var.config_lists.append(config)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, clean_up_tokenization_spaces=False
        )
        global_var.tokenizer_lists.append(tokenizer)

        stage_list = load_model_from_disk_by_name(config)
        global_var.stage_lists_cpu.append(stage_list)

        global_var.is_model_using_lists[i].set()
        global_var.is_model_waiting_offload_lists[i].clear()

    return models_info


def get_half_mem_idx(stage_mem_list):
    half_mem_idx = 0
    half_total_stage_mem = sum(stage_mem_list) / 2
    stage_mem_accu_sum = 0
    for stage_mem in stage_mem_list:
        stage_mem_accu_sum += stage_mem
        if stage_mem_accu_sum <= half_total_stage_mem:
            half_mem_idx += 1
        else:
            break

    return half_mem_idx


def get_total_deploy_mem_cotdm():
    model_stage_mem_list = global_var.model_stage_mem_list
    single_layer_time_list = global_var.single_layer_time_list
    model_slo_list = global_var.model_slo_list
    default_use_gpu_num = len(global_var.cuda_devices)
    use_gpu_num = default_use_gpu_num
    model_num = len(model_stage_mem_list)
    total_deploy_mem = 0
    for i in range(model_num):
        if global_config.DEPLOY_ALG == "Min Deploy":
            load_index = min_deploy(len(model_stage_mem_list[i]), model_slo_list[i], use_gpu_num, single_layer_time_list[i])
            total_deploy_mem += sum(model_stage_mem_list[i][:load_index[0]])
        elif global_config.DEPLOY_ALG == "Half Mem":
            half_mem_idx = get_half_mem_idx(model_stage_mem_list[i])
            total_deploy_mem += sum(model_stage_mem_list[i][:half_mem_idx])

    return total_deploy_mem / 1e6


def get_is_deploy_model_list_alpaserve():
    model_stage_mem_list = global_var.model_stage_mem_list
    deploy_mem_cotdm_sum = get_total_deploy_mem_cotdm()
    model_num = len(global_var.stage_lists_cpu)
    model_stage_mem_sum_list = [sum(mem_list) / float(1e6) for mem_list in model_stage_mem_list]

    is_deploy_model_list = [False for i in range(model_num)]
    cur_deploy_mem_sum = 0
    deploy_model_num = 0

    while deploy_model_num < model_num and cur_deploy_mem_sum < deploy_mem_cotdm_sum:
        max_mem_idx = 0
        for i in range(model_num):
            if is_deploy_model_list[i]:
                continue

            if model_stage_mem_sum_list[i] > model_stage_mem_sum_list[max_mem_idx]:
                max_mem_idx = i

        is_deploy_model_list[max_mem_idx] = True
        cur_deploy_mem_sum += model_stage_mem_sum_list[max_mem_idx]
        deploy_model_num += 1

    return is_deploy_model_list


def deploy_all_models(models_info):
    stage_lists_cpu = global_var.stage_lists_cpu
    model_stage_mem_list = global_var.model_stage_mem_list
    init_deploy_func = global_config.INIT_DEPLOY_FUNC
    deploy_scheme = global_config.DEPLOY_SCHEME

    stage_lists_gpu = []
    load_index_lists = []
    is_stage_loaded_lists = []
    model_stage_cond_lists = []
    model_use_gpu_lists = []
    # deploy_mem_list = []

    if deploy_scheme == "AlpaServe":
        is_deploy_model_list = get_is_deploy_model_list_alpaserve()

    for i in range(len(models_info)):
        info = models_info[i]

        if deploy_scheme == "AlpaServe":
            is_over_limited_mem = not is_deploy_model_list[i]
            (
                stage_list,
                load_index,
                is_stage_loaded,
                model_stage_cond_list,
                model_use_gpu_list,
                deploy_memory,
            ) = init_deploy_func(info, stage_lists_cpu[i], is_over_limited_mem)
        else:
            (
                stage_list,
                load_index,
                is_stage_loaded,
                model_stage_cond_list,
                model_use_gpu_list,
                deploy_memory,
            ) = init_deploy_func(info, stage_lists_cpu[i])
        
        stage_lists_gpu.append(stage_list)
        load_index_lists.append(load_index)
        is_stage_loaded_lists.append(is_stage_loaded)                                                                                                   
        model_stage_cond_lists.append(model_stage_cond_list)
        model_use_gpu_lists.append(model_use_gpu_list)
        # deploy_mem_list.append(deploy_memory)

        for j in range(len(model_use_gpu_list)):
            device_idx = int(model_use_gpu_list[j][5:])
            begin_idx = load_index[j]
            end_idx = load_index[j + 1]
            if j == 0:
                begin_idx = 0

            global_var.memory_fully_loaded_list[device_idx] += sum(model_stage_mem_list[i][begin_idx:end_idx])
    
    global_var.stage_lists_gpu = stage_lists_gpu
    global_var.load_index_lists = load_index_lists
    global_var.is_stage_loaded_lists = is_stage_loaded_lists
    global_var.model_stage_cond_lists = model_stage_cond_lists
    global_var.model_use_gpu_lists = model_use_gpu_lists
    # deploy_scheme_idx = global_config.DEPLOY_SCHEME_DICT[
    #     global_config.DEPLOY_SCHEME
    # ]
    # global_var.deploy_memory_list[deploy_scheme_idx] = deploy_mem_list


def warm_up():
    # temp_tensor = torch.tensor([1])
    # for device in global_var.cuda_devices:
    #     temp_tensor = temp_tensor.to(device, non_blocking=True)

    model_num = len(global_var.stage_lists_gpu)
    for model_idx in range(model_num):
        use_gpu_list = global_var.model_use_gpu_lists[model_idx]
        encoded_input = global_var.tokenizer_lists[model_idx](
            "a [MASK].", return_tensors="pt"
        ).to(use_gpu_list[0])
        inference_func = global_config.INFERENCE_FUNC
        inference_func(encoded_input, model_idx)
        offload_stages_from_gpu_to_cpu(model_idx)


def init_deploy_stages_to_gpu_from_cpu(models_info):
    deploy_all_models(models_info)
    warm_up()

async def init_deploy(model_name_list=None):
    global_var.is_system_running = True
    global_var.clear_global_var()
    models_info = await load_all_models_to_cpu_from_disk(model_name_list)
    init_deploy_stages_to_gpu_from_cpu(models_info)
    # if global_config.DEPLOY_SCHEME == "CoTDM":
    #     start_scheduler()
