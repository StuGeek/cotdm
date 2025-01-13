import asyncio
import torch
import os
import sys
sys.path.append("..")
from init.init_tools import load_all_models_to_cpu_from_disk, init_deploy_stages_to_gpu_from_cpu
from global_data import global_config, global_var
from scheduler import get_single_model_placement
from client.client_single_inf_req import send_single_inf_req
from min_deploy import cal_lat_by_deploynum_and_avaigpu
from server_utils import get_model_idx, start_server_with_offload_sync, close_server
import time
import json

# import gc
# gc.disable()

TEST_TIME = 100
EXP_PORT = 8081
log_file_pointer = None

model_name_list = [
    "bert-large-uncased",
    "bert-1.2B",
    "bert-2.5B",
    "bert-6.7B",
    "gpt2-medium",
    "gpt2-large",
]

deploy_memory_list = []

def print_log(message, end="\n"):
    global log_file_pointer
    print(message, end=end)
    log_file_pointer.write(message + end)


async def get_lat_list(model_name):
    inf_lat_list = []
    ttft_list = []
    e2e_lat_list = []
    for i in range(TEST_TIME):
        _, inf_lat, ttft, e2e_lat = await send_single_inf_req(
            model_name, global_config.DEFAULT_INPUT_TEXT_BERT, port=EXP_PORT
        )
        inf_lat_list.append(inf_lat)
        ttft_list.append(ttft)
        e2e_lat_list.append(e2e_lat)

    return inf_lat_list, ttft_list, e2e_lat_list


async def get_deploymem_with_sloatta_p99(l_nums, r_nums, slo, model_name, model_idx):
    stage_list_gpu = global_var.stage_lists_gpu[model_idx]
    is_stage_loaded = global_var.is_stage_loaded_lists[model_idx]
    stage_cond_list = global_var.model_stage_cond_lists[model_idx]
    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    load_index = global_var.load_index_lists[model_idx]
    deploy_scheme = global_config.DEPLOY_SCHEME
    deploy_scheme_idx = global_config.DEPLOY_SCHEME_DICT[deploy_scheme]
    single_layer_time = global_var.single_layer_time_list[model_idx]

    global deploy_memory_list

    deploy_device = use_gpu_list[0]
    deploy_memory = deploy_memory_list[model_idx]
    total_layer_num = len(stage_list_gpu)
    default_avai_gpu_num = len(global_var.cuda_devices)

    l = l_nums
    r = r_nums
    while l < r:
        mid = (int)(l + (r - l) / 2)
        deploy_params_num = load_index[0]

        if mid > deploy_params_num:
            before_memory, after_memory = 0, 0
            for i in range(len(global_var.cuda_devices)):
                before_memory += torch.cuda.memory_allocated(
                    global_var.cuda_devices[i]
                )

            for i in range(deploy_params_num, mid):
                with stage_cond_list[i]:
                    if is_stage_loaded[i] == True:
                        continue

                    for j in range(len(stage_list_gpu[i].stage)):
                        stage_list_gpu[i].stage[j] = (
                            stage_list_gpu[i]
                            .stage[j]
                            .to(deploy_device, non_blocking=True)
                        )
                        if i != 0 or j == 0:
                            stage_list_gpu[i].stage[j].eval()

                    is_stage_loaded[i] = True
                    stage_cond_list[i].notify()

            for i in range(len(global_var.cuda_devices)):
                after_memory += torch.cuda.memory_allocated(
                    global_var.cuda_devices[i]
                )

        else:
            before_memory, after_memory = 0, 0
            for i in range(len(global_var.cuda_devices)):
                before_memory += torch.cuda.memory_allocated(
                    global_var.cuda_devices[i]
                )

            for i in range(mid, deploy_params_num):
                with stage_cond_list[i]:
                    is_stage_loaded[i] = False

                    if stage_list_gpu[i] != None and stage_list_gpu[i].stage != None:
                        for j in range(len(stage_list_gpu[i].stage)):
                            stage_list_gpu[i].stage[j] = (
                                stage_list_gpu[i].stage[j].cpu()
                            )

            torch.cuda.empty_cache()

            for i in range(len(global_var.cuda_devices)):
                after_memory += torch.cuda.memory_allocated(
                    global_var.cuda_devices[i]
                )

        deploy_memory += (after_memory - before_memory) / float(1e6)
        deploy_memory_list[deploy_scheme_idx][model_idx] = deploy_memory
        if deploy_scheme == "Strawman":
            load_index[0] = mid
        elif deploy_scheme == "DeepPlan":
            _, load_index = cal_lat_by_deploynum_and_avaigpu(
                mid,
                total_layer_num,
                default_avai_gpu_num,
                single_layer_time,
            )

            global_var.load_index_lists[model_idx] = load_index
            stage_mem_list = global_var.model_stage_mem_list[model_idx]
            use_gpu_list = get_single_model_placement(
                stage_mem_list, load_index, deploy_device=use_gpu_list[0]
            )
            global_var.model_use_gpu_lists[model_idx] = use_gpu_list

        inf_lat_list, _, _ = await get_lat_list(model_name)
        sorted_lat_list = sorted(inf_lat_list)
        lat_p99 = sorted_lat_list[int(0.99 * len(sorted_lat_list))]

        if lat_p99 < slo:
            r = mid
        else:
            l = mid + 1


async def test_exp1(exp_time):
    global deploy_memory_list

    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    deploy_scheme_num = len(deploy_scheme_dict)
    deploy_scheme_list = ["Totally-Offload", "Ready", "CoTDM", "Strawman", "DeepPlan"]
    mean_lat_list_all_schemes = [[] for i in range(deploy_scheme_num)]
    thr_list_all_schemes = [[] for i in range(deploy_scheme_num)]
    deploy_memory_list = [[] for i in range(deploy_scheme_num)]
    lat_p99_cotdm = []
    eps = 1e-6

    global_var.is_system_running = True
    global_var.clear_global_var()
    models_info = await load_all_models_to_cpu_from_disk(model_name_list)

    for deploy_scheme in deploy_scheme_list:
        scheme_idx = deploy_scheme_dict[deploy_scheme]
        global_config.set_deploy_scheme(deploy_scheme)
        print_log("deploy scheme: {}".format(deploy_scheme))
        init_deploy_stages_to_gpu_from_cpu(models_info)
        model_stage_mem_list = global_var.model_stage_mem_list
        load_index_lists = global_var.load_index_lists
        for i in range(len(load_index_lists)):
            stage_mem_list = model_stage_mem_list[i]
            load_index = load_index_lists[i]
            deploy_memory_list[scheme_idx].append(stage_mem_list[0:load_index[0]])
        server_runner, server_site = await start_server_with_offload_sync(EXP_PORT)

        mean_lat_list_all_models = []
        thr_list_all_models = []

        for i in range(len(model_name_list)):
            model_name = model_name_list[i]
            inf_lat_list, ttft_list, e2e_lat_list = await get_lat_list(model_name)
            avg_inf_lat = sum(inf_lat_list) / len(inf_lat_list)
            avg_ttft = sum(ttft_list) / len(ttft_list)
            avg_e2e_lat = sum(e2e_lat_list) / len(e2e_lat_list)
            throughput = len(e2e_lat_list) / sum(e2e_lat_list)

            print_log(
                "{}: avg_inf_lat={:.4f}s, avg_e2e_lat={:.4f}s, throughput={:.4f}req/s".format(
                    model_name, avg_inf_lat, avg_e2e_lat, throughput
                )
            )
            mean_lat_list_all_models.append(avg_inf_lat)
            thr_list_all_models.append(throughput)

            if deploy_scheme == "CoTDM":
                sorted_lat_list = sorted(inf_lat_list)
                lat_p99_cotdm.append(sorted_lat_list[int(0.99 * len(sorted_lat_list))])
            elif deploy_scheme == "Strawman" or deploy_scheme == "DeepPlan":
                sorted_lat_list = sorted(inf_lat_list)
                lat_p99 = sorted_lat_list[int(0.99 * len(sorted_lat_list))]

                if abs(lat_p99 - lat_p99_cotdm[i]) < eps:
                    continue

                model_idx = await get_model_idx(model_name)
                deploy_params_num = global_var.load_index_lists[model_idx][0]
                total_layer_num = len(global_var.stage_lists_gpu[model_idx])
                if lat_p99 < lat_p99_cotdm[i]:
                    await get_deploymem_with_sloatta_p99(
                        0,
                        deploy_params_num,
                        lat_p99_cotdm[i],
                        model_name,
                        model_idx,
                    )
                else:
                    await get_deploymem_with_sloatta_p99(
                        deploy_params_num,
                        total_layer_num,
                        lat_p99_cotdm[i],
                        model_name,
                        model_idx,
                    )

        mean_lat_list_all_schemes[scheme_idx] = mean_lat_list_all_models
        thr_list_all_schemes[scheme_idx] = thr_list_all_models

        await close_server(server_site, server_runner)
        print_log("")

    global_var.clear_global_var()
    data_dict = {}
    data_dict["mean_lat_list_all_schemes"] = mean_lat_list_all_schemes
    data_dict["thr_list_all_schemes"] = thr_list_all_schemes
    data_dict["cost_list_all_schemes"] = deploy_memory_list

    exp_res_save_dir = global_config.EXP_RES_DIR + "exp1/"
    exp1_data_path = exp_res_save_dir + "exp1_" + str(exp_time) + ".json"
    with open(exp1_data_path, "w") as write_f:
        json.dump(data_dict, write_f, indent=4, ensure_ascii=False)
    print_log("Experiment1 datas save path: {}".format(exp1_data_path))


async def main():
    exp_res_save_dir = global_config.EXP_RES_DIR + "exp1/"
    if not os.path.exists(exp_res_save_dir):
        os.makedirs(exp_res_save_dir)

    global log_file_pointer
    exp_time = int(time.time())
    exp1_log_file_path = exp_res_save_dir + "exp1_" + str(exp_time) + ".log"
    with open(exp1_log_file_path, 'w') as f:
        log_file_pointer = f
        await test_exp1(exp_time)
        print_log("Experiment1 logs save path: {}".format(exp1_log_file_path))


if __name__ == "__main__":
    asyncio.run(main())
