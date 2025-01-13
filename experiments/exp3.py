from aiohttp import web
import os
import asyncio
import time
import sys
sys.path.append("..")
from init.init_tools import load_all_models_to_cpu_from_disk, init_deploy_stages_to_gpu_from_cpu
from global_data import global_config, global_var
from global_data.global_class import SingleInferenceRequest, WorkloadInferenceRequest
from client.client_workload import send_request_by_workload_with_submit_idx
from server_utils import inference_return_with_offload_thread, close_server
from collections import OrderedDict
import argparse
import json
from exp2 import generate_model_name_list
from azure_trace import Trace, preprocess_azure_v1_trace_sparse

EXP_PORT = 8083
log_file_pointer = None
modelset_name = None

offload_thread_list = []

rate_scale_list = [0.15, 0.16, 0.17, 0.18, 0.19]
cv_scale_list = [0.7, 1.0, 1.5, 1.9, 2.3]
slo_scale_list = [5, 5.5, 6, 6.5, 7, 7.5, 8]

gen_rate_list = [0.15, 0.16, 0.17, 0.18, 0.19, 0.17, 0.17, 0.17, 0.17]
gen_cv_list = [1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 1.5, 1.9, 2.3]
gen_seed_list = [29224, 37620, 680, 204380, 92836, 139100, 42988, 3179748, 2770340]
gen_seed_list = [i - 4 for i in gen_seed_list]
interval_seconds = 60

def print_log(message, end="\n"):
    global log_file_pointer
    print(message, end=end)
    log_file_pointer.write(message + end)


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


def get_workload_from_azure_replay(model_name_list, azure_replay):    
    replay_workload = azure_replay[model_name_list[0]].to_workload()
    model_name_list_len = len(model_name_list)
    for i in range(1, model_name_list_len):
        replay_workload += azure_replay[model_name_list[i]].to_workload()

    inf_req_list = []
    req_len = len(replay_workload.requests)
    for i in range(req_len):
        model_name = replay_workload.requests[i].model_name
        input_text = ""
        if model_name.find("bert") != -1:
            input_text = global_config.DEFAULT_INPUT_TEXT_BERT
        elif model_name.find("gpt2") != -1:
            input_text = global_config.DEFAULT_INPUT_TEXT_GPT2

        inf_req_list.append(SingleInferenceRequest(model_name, input_text))

    replay_workload.requests = inf_req_list

    return WorkloadInferenceRequest(replay_workload.requests, replay_workload.arrivals)


def generate_azure_v1_workload(model_name_list, gen_rate_list, gen_cv_list, gen_seed_list):
    workload_list = []
    azure_v1_dir = global_config.AZURE_V1_DIR
    azure_v1_name = global_config.AZURE_V1_NAME

    global interval_seconds

    if not os.path.exists(azure_v1_dir):
        preprocess_azure_v1_trace_sparse(os.path.dirname(azure_v1_dir), n_day=1)

    trace_azure_v1 = Trace(azure_v1_name, azure_v1_dir)

    for i in range(len(gen_rate_list)):
        replays = trace_azure_v1.replay(model_name_list,
                    model_mapping_strategy="stripe",
                    start_time="0.0.0",
                    end_time="0.0.1",
                    arrival_distribution="gamma",
                    interval_seconds=interval_seconds,
                    rate_scale_factor=gen_rate_list[i],
                    cv_scale_factor=gen_cv_list[i],
                    seed=gen_seed_list[i])
        workload = get_workload_from_azure_replay(model_name_list, replays)
        workload_list.append(workload)

    return workload_list


async def get_lat_list_all_workload(workload_list, gen_rate_list, gen_cv_list):
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
            "When rate_scale={}, cv_scale={}, lat_p90={:.4f}s, lat_p95={:.4f}s, lat_p99={:.4f}s".format(
                gen_rate_list[i], gen_cv_list[i], lat_p90, lat_p95, lat_p99
            )
        )

        for offload_thread in offload_thread_list:
            if offload_thread is not None:
                offload_thread.join()

    return e2e_lat_list_all_workload


async def test_exp3(exp_time, modelset, scheme):
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
    workload_list = generate_azure_v1_workload(
        model_name_list, gen_rate_list, gen_cv_list, gen_seed_list
    )

    print_log("For {},".format(modelset_name))
    server_runner, server_site = await start_server()

    global_var.is_system_running = True
    global_var.clear_global_var()
    models_info = await load_all_models_to_cpu_from_disk(model_name_list)
    
    global_config.set_deploy_scheme(deploy_scheme)
    print_log("deploy scheme: {}".format(deploy_scheme))

    init_deploy_stages_to_gpu_from_cpu(models_info)
    e2e_lat_list_all_workload = await get_lat_list_all_workload(workload_list, gen_rate_list, gen_cv_list)

    data_dict = {}
    data_dict["deploy_scheme"] = deploy_scheme
    data_dict["rate_scale_list"] = rate_scale_list
    data_dict["cv_scale_list"] = cv_scale_list
    data_dict["avg_e2e_lat_ready"] = [models_info[j]["avg_e2e_lat_ready"] for j in range(len(models_info))]
    data_dict["e2e_lat_list_all_workload"] = e2e_lat_list_all_workload

    await close_server(server_site, server_runner)
    global_var.clear_global_var()

    exp_res_save_dir = global_config.EXP_RES_DIR + "exp3/"
    exp3_modelset_data_path = exp_res_save_dir + "exp3_" + modelset_name + "_" + scheme + "_" + str(exp_time) + ".json"
    with open(exp3_modelset_data_path, "w") as write_f:
        json.dump(data_dict, write_f, indent=4, ensure_ascii=False)
    print_log("Experiment3 {} datas save path: {}".format(modelset_name, exp3_modelset_data_path))
    

async def main(modelset, scheme):
    exp_res_save_dir = global_config.EXP_RES_DIR + "exp3/"
    if not os.path.exists(exp_res_save_dir):
        os.makedirs(exp_res_save_dir)

    global log_file_pointer
    exp_time = int(time.time())
    exp3_log_file_path = exp_res_save_dir + "exp3_" + modelset_name + "_" + scheme + "_" + str(exp_time) + ".log"
    with open(exp3_log_file_path, "w") as f:
        log_file_pointer = f
        await test_exp3(exp_time, modelset, scheme)
        print_log("Experiment3 logs save path: {}\n".format(exp3_log_file_path))


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
