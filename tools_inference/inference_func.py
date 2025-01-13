import sys
sys.path.append(".")
sys.path.append("..")
from global_data import global_var
from tools_inference.bert.inference_bert_cotdm import inference_bert_cotdm
from tools_inference.bert.inference_bert_ready import inference_bert_ready
from tools_inference.bert.inference_bert_totoff import inference_bert_totoff
from tools_inference.bert.inference_bert_strawman import inference_bert_strawman
from tools_inference.bert.inference_bert_alpaserve import inference_bert_alpaserve
from tools_inference.bert.inference_bert_deepplan import inference_bert_deepplan
from tools_inference.gpt2.inference_gpt2_cotdm import inference_gpt2_cotdm, get_generated_text_cotdm
from tools_inference.gpt2.inference_gpt2_ready import inference_gpt2_ready, get_generated_text_ready
from tools_inference.gpt2.inference_gpt2_totoff import inference_gpt2_totoff, get_generated_text_totoff
from tools_inference.gpt2.inference_gpt2_strawman import inference_gpt2_strawman, get_generated_text_strawman
from tools_inference.gpt2.inference_gpt2_alpaserve import inference_gpt2_alpaserve, get_generated_text_alpaserve
from tools_inference.gpt2.inference_gpt2_deepplan import inference_gpt2_deepplan, get_generated_text_deepplan
from tools_inference.inference_utils import get_generated_text_gpt2
is_generated_text = False


def inference_func_cotdm(encoded_input, model_idx):
    model_type = global_var.config_lists[model_idx].model_type
    if model_type == "bert":
        return inference_bert_cotdm(encoded_input, model_idx)
    elif model_type == "gpt2":
        if is_generated_text:
            return get_generated_text_gpt2(encoded_input, model_idx, inference_gpt2_cotdm)
            # return get_generated_text_cotdm(encoded_input, model_idx)
        else:
            return inference_gpt2_cotdm(encoded_input, model_idx)


def inference_func_ready(encoded_input, model_idx):
    model_type = global_var.config_lists[model_idx].model_type
    if model_type == "bert":
        return inference_bert_ready(encoded_input, model_idx)
    elif model_type == "gpt2":
        if is_generated_text:
            return get_generated_text_gpt2(encoded_input, model_idx, inference_gpt2_ready)
            # return get_generated_text_ready(encoded_input, model_idx)
        else:
            return inference_gpt2_ready(encoded_input, model_idx)


def inference_func_totoff(encoded_input, model_idx):
    model_type = global_var.config_lists[model_idx].model_type
    if model_type == "bert":
        return inference_bert_totoff(encoded_input, model_idx)
    elif model_type == "gpt2":
        if is_generated_text:
            return get_generated_text_gpt2(encoded_input, model_idx, inference_gpt2_totoff)
            # return get_generated_text_totoff(encoded_input, model_idx)
        else:
            return inference_gpt2_totoff(encoded_input, model_idx)


def inference_func_strawman(encoded_input, model_idx):
    model_type = global_var.config_lists[model_idx].model_type
    if model_type == "bert":
        return inference_bert_strawman(encoded_input, model_idx)
    elif model_type == "gpt2":
        if is_generated_text:
            return get_generated_text_gpt2(encoded_input, model_idx, inference_gpt2_strawman)
            # return get_generated_text_strawman(encoded_input, model_idx)
        else:
            return inference_gpt2_strawman(encoded_input, model_idx)


def inference_func_alpaserve(encoded_input, model_idx):
    model_type = global_var.config_lists[model_idx].model_type
    if model_type == "bert":
        return inference_bert_alpaserve(encoded_input, model_idx)
    elif model_type == "gpt2":
        if is_generated_text:
            return get_generated_text_gpt2(encoded_input, model_idx, inference_gpt2_alpaserve)
            # return get_generated_text_alpaserve(encoded_input, model_idx)
        else:
            return inference_gpt2_alpaserve(encoded_input, model_idx)


def inference_func_deepplan(encoded_input, model_idx):
    model_type = global_var.config_lists[model_idx].model_type
    if model_type == "bert":
        return inference_bert_deepplan(encoded_input, model_idx)
    elif model_type == "gpt2":
        if is_generated_text:
            return get_generated_text_gpt2(encoded_input, model_idx, inference_gpt2_deepplan)
            # return get_generated_text_deepplan(encoded_input, model_idx)
        else:
            return inference_gpt2_deepplan(encoded_input, model_idx)


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer
    from init.init_tools import load_model_from_disk_gpt2
    import threading
    import global_data.global_config as global_config
    from global_data.global_class import ModelStage
    from offload_tools import offload_stages_from_gpu_to_cpu
    import copy

    gpt2_model_name = "gpt2-medium"
    model_path = global_config.MODEL_SAVE_DIR + gpt2_model_name

    input_text = "You can now dive in the comfort of your Ford Explorer red lights, which let you take exquisite picture. Open the lens and aim to take a blurry window that shows a Ford Norris road that's not registered at Flickr. And, view each car in full view? You can use both of your Leica reflex sightlamoms, boosting your perceived resolution to 3160pfx when shooting in north-west California. (And that's right, driving always leaves a innocent minor site \u2014 like Flickr) Read more about runs in Oakland Auto Repair. The big goal: the drive-thru feuds where one manufacturer does just fine and the other manufacturer seems driven to get away with just fine.\n\nIn other words, don't get saved for good \u2013 captured photos and closed-circuit TV wins none of those kind of plicks.\n\nSeriously. In case there wasn't enough \u2014 you can try Huawei AimBit zoomless shooting below by using your camera with a desired zoom level. Again, that is, it means first thing in all the morning, then slowly losing the ability to view the final passage of the image when you look up. While my Ford Explorer is good, great and just plain fun, expecting to run the lights down, that bit of a little tweaking may prove scaring the hell out of you.\n\nLike this: Like Loading..."
    config = AutoConfig.from_pretrained(model_path)
    config._name_or_path = model_path
    # config.total_encoder_num = config.num_hidden_layers
    config.total_h_layer_num = config.n_layer
    # config.total_layer_num = config.total_h_layer_num + 4
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenization_spaces=False
    )
    encoded_input = tokenizer(input_text, return_tensors="pt").to("cuda:0")
    stage_list_cpu = load_model_from_disk_gpt2(config)
    total_layer_num = len(stage_list_cpu)
    use_gpu_list = ["cuda:0"]
    deploy_device = use_gpu_list[0]

    is_stage_loaded = [False for i in range(total_layer_num)]
    load_index = [total_layer_num, total_layer_num]
    deploy_params_num = load_index[0]
    stage_cond_list = [
        threading.Condition(threading.Lock()) for i in range(total_layer_num)
    ]
    stage_list_gpu = []

    for i in range(deploy_params_num):
        stages_gpu = []
        for j in range(len(stage_list_cpu[i].stage)):
            stage_gpu = stage_list_cpu[i].stage[j].to(deploy_device)
            stage_gpu.eval()
            stages_gpu.append(stage_gpu)
        stage_list_gpu.append(ModelStage(stages_gpu, stage_list_cpu[i].type))
        is_stage_loaded[i] = True

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

    temp_encoded_input = copy.deepcopy(encoded_input)
    generated_text, inf_lat, ttft = get_generated_text_gpt2(temp_encoded_input, 0, inference_gpt2_cotdm)
    offload_stages_from_gpu_to_cpu(0)
    temp_encoded_input = copy.deepcopy(encoded_input)
    generated_text, inf_lat, ttft = get_generated_text_gpt2(temp_encoded_input, 0, inference_gpt2_cotdm)

    print("input:", input_text)
    print("output:", generated_text)
    print("inference latency: {:.4f}s, TTFT: {:.4f}s".format(inf_lat, ttft))

    global_var.is_system_running = False
