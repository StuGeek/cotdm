from aiohttp import web
import asyncio
import os
import time
import sys
sys.path.append("..")
from init.init_tools import load_all_models_to_cpu_from_disk, init_deploy_stages_to_gpu_from_cpu
from global_data import global_config, global_var
from global_data.global_class import WorkloadInferenceRequest
from workload import GammaProcess
from client.client_workload import send_request_by_workload_with_submit_idx
from server_utils import inference_return_with_offload_thread, close_server
from collections import OrderedDict
import argparse
import json

EXP_PORT = 8082
log_file_pointer = None
modelset_name = None

offload_thread_list = []

duration = 60
arrival_rate = 2
cv_list = [0.01, 0.5, 1, 1.5, 2, 2.5]
seed_list = [0, 2104, 4770, 71388, 141689, 34638]

def print_log(message, end="\n"):
    global log_file_pointer
    print(message, end=end)
    log_file_pointer.write(message + end)


def generate_model_name_list(model_set):
    model_name_list = []
    for name, model_num in model_set.items():
        for i in range(model_num):
            model_name_list.append(name + str(i + 1))

    return model_name_list


async def handle_inference_with_offload_thread(request):
    global offload_thread_list

    data = await request.json()
    model_name = data.get("model_name", None)
    input_text = data.get("input_text", None)
    submit_idx = int(data.get("submit_idx", None))

    if model_name is None:
        return web.json_response({"text": "Model name is empty!", "inf_lat": -1, "ttft": -1})

    if input_text is None:
        return web.json_response({"text": "Input is empty!", "inf_lat": -1, "ttft": -1})

    result, inf_lat, ttft, offload_thread = await inference_return_with_offload_thread(
        input_text, model_name
    )
    if result is None:
        return web.json_response({"text": "Inference failed!", "inf_lat": -1, "ttft": -1})
    offload_thread_list[submit_idx] = offload_thread

    data = {"text": result, "inf_lat": inf_lat, "ttft": ttft}
    return web.json_response(data)


async def start_server():
    app = web.Application()
    app.router.add_post("/inference", handle_inference_with_offload_thread)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "127.0.0.1", EXP_PORT)
    await site.start()

    return runner, site


def generate_gamma_workload_by_cv_list(model_name_list, cv_list, seed_list):
    workload_list = []
    actual_rate_list = []
    actual_cv_list = []
    for i in range(len(cv_list)):
        gamma_workload = GammaProcess(
            arrival_rate=arrival_rate, cv=cv_list[i]
        ).generate_workload_by_model_name_list(
            model_name_list, start=0, duration=duration, seed=seed_list[i]
        )
        actual_rate_list.append(gamma_workload.rate)
        actual_cv_list.append(gamma_workload.cv)
        workload_list.append(
            WorkloadInferenceRequest(gamma_workload.requests, gamma_workload.arrivals)
        )

    return workload_list, actual_rate_list, actual_cv_list


def get_lat_p99_by_modelset(e2e_lat_list, modelset):
    model_type_num = len(modelset)
    model_num_list = list(modelset.values())
    model_num = sum(model_num_list)
    model_idx_list = []
    for i in range(len(model_num_list)):
        model_idx_list += [i for j in range(model_num_list[i])]

    lat_list_all_model = [[] for i in range(model_type_num)]
    for i in range(len(e2e_lat_list)):
        lat_list_all_model[model_idx_list[i % model_num]].append(e2e_lat_list[i])

    lat_p99_list_all_model = [sorted(lat_list_all_model)[int(0.99 * len(lat_list_all_model))] for i in range(len(lat_list_all_model))]
    
    return lat_p99_list_all_model


async def get_lat_list_all_workload(workload_list, cv_list):
    global offload_thread_list

    e2e_lat_list_all_workload = []

    for i in range(len(workload_list)):
        workload_len = len(workload_list[i].inf_req_list)
        offload_thread_list = [None for i in range(workload_len)]
        e2e_lat_list = await send_request_by_workload_with_submit_idx(
            workload_list[i], EXP_PORT
        )
        e2e_lat_list_all_workload.append(e2e_lat_list)
        sorted_e2e_lat_list = sorted(e2e_lat_list)
        lat_p90 = sorted_e2e_lat_list[int(0.90 * len(sorted_e2e_lat_list))]
        lat_p95 = sorted_e2e_lat_list[int(0.95 * len(sorted_e2e_lat_list))]
        lat_p99 = sorted_e2e_lat_list[int(0.99 * len(sorted_e2e_lat_list))]
        print_log(
            "When cv={:.4f}, lat_p90={:.4f}s, lat_p95={:.4f}s, lat_p99={:.4f}s".format(
                cv_list[i], lat_p90, lat_p95, lat_p99
            )
        )

        for offload_thread in offload_thread_list:
            if offload_thread is not None:
                offload_thread.join()

    return e2e_lat_list_all_workload


async def test_exp2(exp_time, modelset, scheme):
    global offload_thread_list

    deploy_scheme = None
    if scheme == "cotdm":
        deploy_scheme = "CoTDM"
    elif scheme == "totoff":
        deploy_scheme = "Totally-Offload"
    elif scheme == "strawman":
        deploy_scheme = "Strawman"
    elif scheme == "deepplan":
        deploy_scheme = "DeepPlan"
    elif scheme == "alpaserve":
        deploy_scheme = "AlpaServe"

    model_name_list = generate_model_name_list(modelset)
    workload_list, actual_rate_list, actual_cv_list = generate_gamma_workload_by_cv_list(
        model_name_list, cv_list, seed_list
    )

    print_log("For {},".format(modelset_name))
    server_runner, server_site = await start_server()

    global_var.is_system_running = True
    global_var.clear_global_var()
    models_info = await load_all_models_to_cpu_from_disk(model_name_list)

    global_config.set_deploy_scheme(deploy_scheme)
    print_log("deploy scheme: {}".format(deploy_scheme))

    init_deploy_stages_to_gpu_from_cpu(models_info)
    e2e_lat_list_all_workload = await get_lat_list_all_workload(workload_list, actual_cv_list)

    data_dict = {}
    data_dict["deploy_scheme"] = deploy_scheme
    data_dict["rate_list"] = actual_rate_list
    data_dict["cv_list"] = actual_cv_list
    data_dict["avg_e2e_lat_ready"] = [models_info[j]["avg_e2e_lat_ready"] for j in range(len(models_info))]
    data_dict["duration_list"] = [workload_list[i].arri_time_list[-1] - workload_list[i].arri_time_list[0] for i in range(len(workload_list))]
    data_dict["e2e_lat_list_all_workload"] = e2e_lat_list_all_workload

    await close_server(server_site, server_runner)
    global_var.clear_global_var()

    exp_res_save_dir = global_config.EXP_RES_DIR + "exp2/"
    exp2_modelset_data_path = exp_res_save_dir + "exp2_" + modelset_name + "_" + scheme + "_" + str(exp_time) + ".json"
    with open(exp2_modelset_data_path, "w") as write_f:
        json.dump(data_dict, write_f, indent=4, ensure_ascii=False)
    print_log("Experiment2 {} datas save path: {}".format(modelset_name, exp2_modelset_data_path))


async def main(modelset, scheme):
    exp_res_save_dir = global_config.EXP_RES_DIR + "exp2/"
    if not os.path.exists(exp_res_save_dir):
        os.makedirs(exp_res_save_dir)

    global log_file_pointer
    exp_time = int(time.time())
    exp2_log_file_path = exp_res_save_dir + "exp2_" + modelset_name + "_" + scheme + "_" + str(exp_time) + ".log"
    with open(exp2_log_file_path, "w") as f:
        log_file_pointer = f
        await test_exp2(exp_time, modelset, scheme)
        print_log("Experiment2 logs save path: {}\n".format(exp2_log_file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", type=str, default="cotdm")
    parser.add_argument("--modelset-path", type=str, default="../modelsets/modelset1.json")
    args = parser.parse_args()
    scheme = args.scheme
    modelset_path = args.modelset_path

    with open(modelset_path, "r") as f:
        json_data = json.load(f)
        for key, val in json_data.items():
            modelset_name = key
            modelset = OrderedDict(val)

    asyncio.run(main(modelset, scheme))
