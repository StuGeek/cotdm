import torch
from global_data import global_var

def count_parameters(stage_list):
    total_params = 0
    for i in range(len(stage_list)):
        for stage in stage_list[i].stage:
            if type(stage) == torch.Tensor:
                total_params += stage.numel()
            else:
                for name, param in stage.named_parameters():
                    if param.requires_grad:
                        # print(param)
                        param_size = param.numel()
                        total_params += param_size

    return total_params


def print_parameters_num(parameters_num):
    parameters_str = str(parameters_num)
    parameters_bit_num = len(parameters_str)
    if parameters_bit_num < 4:
        print("The number of parameters: {}".format(parameters_num))
    elif parameters_bit_num == 4:
        print("The number of parameters: {:.1f}K".format(parameters_num / float(1e3)))
    elif parameters_bit_num < 7:
        print("The number of parameters: {:.0f}K".format(parameters_num / float(1e3)))
    elif parameters_bit_num == 7:
        print("The number of parameters: {:.1f}M".format(parameters_num / float(1e6)))
    elif parameters_bit_num < 10:
        print("The number of parameters: {:.0f}M".format(parameters_num / float(1e6)))
    elif parameters_bit_num == 10:
        print("The number of parameters: {:.1f}B".format(parameters_num / float(1e9)))
    elif parameters_bit_num < 13:
        print("The number of parameters: {:.0f}B".format(parameters_num / float(1e9)))
    elif parameters_bit_num == 13:
        print("The number of parameters: {:.1f}T".format(parameters_num / float(1e12)))
    else:
        print("The number of parameters: {:.0f}T".format(parameters_num / float(1e12)))


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


def get_stage_mem_list(model_stage_list):
    stage_mem_list = [0 for i in range(len(model_stage_list))]

    for i in range(len(model_stage_list)):
        stage_mem_before = 0
        for j in range(len(global_var.cuda_devices)):
            stage_mem_before += torch.cuda.memory_allocated(global_var.cuda_devices[j])
        
        for j in range(len(model_stage_list[i].stage)):
            model_stage_list[i].stage[j] = model_stage_list[i].stage[j].to(global_var.cuda_devices[i % 4])
        
        stage_mem_after = 0
        for j in range(len(global_var.cuda_devices)):
            stage_mem_after += torch.cuda.memory_allocated(global_var.cuda_devices[j])
        stage_mem_list[i] = stage_mem_after - stage_mem_before

    for i in range(len(model_stage_list)):
        for j in range(len(model_stage_list[i].stage)):
            model_stage_list[i].stage[j] = model_stage_list[i].stage[j].cpu()

    torch.cuda.empty_cache()

    return stage_mem_list


def print_model_size(model_size):
    model_size_str = str(model_size)
    model_size_bit_num = len(model_size_str)
    if model_size_bit_num < 4:
        print("The size of model: {}Bytes".format(model_size))
    elif model_size_bit_num == 4:
        print("The size of model: {:.1f}KB".format(model_size / float(1e3)))
    elif model_size_bit_num < 7:
        print("The size of model: {:.0f}KB".format(model_size / float(1e3)))
    elif model_size_bit_num == 7:
        print("The size of model: {:.1f}MB".format(model_size / float(1e6)))
    elif model_size_bit_num < 10:
        print("The size of model: {:.0f}MB".format(model_size / float(1e6)))
    elif model_size_bit_num == 10:
        print("The size of model: {:.1f}GB".format(model_size / float(1e9)))
    elif model_size_bit_num < 13:
        print("The size of model: {:.0f}GB".format(model_size / float(1e9)))
    elif model_size_bit_num == 13:
        print("The size of model: {:.1f}TB".format(model_size / float(1e12)))
    else:
        print("The size of model: {:.0f}TB".format(model_size / float(1e12)))