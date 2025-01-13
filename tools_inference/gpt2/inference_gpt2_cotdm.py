import time
import torch
from threading import Thread
import sys
sys.path.append(".")
sys.path.append("..")
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
    get_next_token,
)
from global_data import global_var
from global_data.global_class import ThreadWithReturnValue

def cal_model(encoded_input, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    load_index = global_var.load_index_lists[model_idx]
    config = global_var.config_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]
    load_index_len = len(load_index)

    cal_index = 0
    # total_cal_time = 0
    # cost_time = 0
    # wait_times = 0
    # inf_gpu_list = get_inf_gpu_list(load_index, use_gpu_list)

    total_h_layer_num = config.total_h_layer_num
    total_layer_num = len(stage_list_gpu)
    while True:
        cal_device = torch.device(use_gpu_list[0])
        for i, device_name in enumerate(use_gpu_list):
            if i < load_index_len - 1 and cal_index < load_index[i + 1]:
                cal_device = torch.device(device_name)
                break

        # cal_device = inf_gpu_list[cal_index]

        # start = time.time()

        # s1 = time.time()

        with stage_cond_list[cal_index]:
            if is_stage_loaded[cal_index] == False:
                stage_cond_list[cal_index].wait()

        # print(cal_index, is_stage_loaded)
        # print(gpt2_sem_list)
        # print("123")
        # stage_event_list[cal_index].wait()
        model_stages = stage_list_gpu[cal_index].stage
        stage_type = stage_list_gpu[cal_index].type

        # print(cal_device, "cal index:", cal_index, time.time() - global_var.inf_start_time, "time2")

        # print(cal_index, gpt2_sem_list)

        # print(time.time() - s1)

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
            # print(cal_index, "cal_time: ", time.time() - start)
            # print()

        # end = time.time()
        # total_cal_time += end - start

        cal_index += 1

        # print(cal_device, "cal index:", cal_index - 1, time.time() - global_var.inf_start_time, "time3")

        if cal_index >= total_layer_num:
            break

    # print("total_cal_time: ", total_cal_time, "aver: ", total_cal_time / (total_stage_num))
    # print("wait times: ", wait_times)
    # print("cost time: ", cost_time, "aver: ", cost_time / wait_times)
    torch.cuda.synchronize()
    return lm_logits


def load_model(begin_idx, end_idx, device, model_idx):
    # total_load_time = 0
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]

    for i in range(begin_idx, end_idx):
        with stage_cond_list[i]:
            if is_stage_loaded[i] == True:
                continue

            for j in range(len(stage_list_gpu[i].stage)):
                # start = time.time()
                stage_list_gpu[i].stage[j] = (
                    stage_list_gpu[i].stage[j].to(device, non_blocking=True)
                )
                # print(i, "load time1: ", time.time() - start, start)
                stage_list_gpu[i].stage[j].eval()
            # print(i, "load time2: ", time.time() - start, start)
            # print(device, "load index:", i, time.time() - global_var.inf_start_time)

            is_stage_loaded[i] = True
            stage_cond_list[i].notify()


def cal_func(encoded_input, stream, model_idx):
    with torch.cuda.stream(stream):
        result = cal_model(encoded_input, model_idx)
    return result


def load_func(stream, begin_idx, end_idx, device, model_idx):
    with torch.cuda.stream(stream):
        load_model(begin_idx, end_idx, device, model_idx)


def inference_gpt2_cotdm(encoded_input, model_idx):
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


def get_generated_text_cotdm(encoded_input, model_idx, inference_gpt2_func=inference_gpt2_cotdm):
    start_inf_time = time.time()
    global_var.offload_time_list[model_idx] = 0
    
    GPT2_MAX_LENGTH = 800
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


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer
    from init.init_tools import load_model_from_disk_gpt2
    import threading
    import global_data.global_config as global_config
    from global_data.global_class import ModelStage
    from offload_tools import offload_stages_from_gpu_to_cpu

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

    generated_text, inf_lat, ttft = get_generated_text_cotdm(encoded_input, 0, inference_gpt2_cotdm)
    offload_stages_from_gpu_to_cpu(0)
    generated_text, inf_lat, ttft = get_generated_text_cotdm(encoded_input, 0, inference_gpt2_cotdm)

    print("input:", input_text)
    print("output:", generated_text)
    print("inference latency: {:.4f}s, TTFT: {:.4f}s".format(inf_lat, ttft))

    global_var.is_system_running = False