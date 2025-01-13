import time
from threading import Thread
from aiohttp import web
from global_data import global_config, global_var
from tools_inference.inference_func import inference_func_cotdm
from offload_tools import offload_func, offload_all_models_from_gpu_to_cpu


async def get_model_idx(model_name, slo=1.0):
    model_name_slo_idx_dict = global_var.model_name_slo_idx_dict

    min_req_idx = -1
    min_req_num = -1
    is_meet_slo = False
    for model_name_slo, idx_list in model_name_slo_idx_dict.items():
        if model_name != model_name_slo.name:
            continue

        if model_name_slo.slo <= slo:
            is_meet_slo = True

        if min_req_idx != -1 and is_meet_slo is True:
            continue

        for idx in idx_list:
            if min_req_num == -1 or global_var.model_request_num_list[idx] < min_req_num:
                min_req_num = global_var.model_request_num_list[idx]
                min_req_idx = idx

    if min_req_idx == -1:
        print('The model "{}" is not on the server.'.format(model_name))

    # print(min_req_idx)
    return min_req_idx


async def inference_by_deploy_scheme(input_text, model_idx):
    global_var.cur_request_num.incrementAndGet()
    global_var.cur_wait_offload_num.incrementAndGet()
    if global_config.DEBUG_MODE_SCHEDULER:
        print("global var cur_request_num increase:", global_var.cur_request_num.get())
        print("global var cur_wait_offload_num increase:", global_var.cur_wait_offload_num.get())

    inference_func = global_config.INFERENCE_FUNC

    use_gpu_list = global_var.model_use_gpu_lists[model_idx]
    encoded_input = global_var.tokenizer_lists[model_idx](input_text, return_tensors="pt").to(
        use_gpu_list[0]
    )
    global_var.model_request_num_list[model_idx] += 1
    global_var.is_model_using_lists[model_idx].wait()
    global_var.is_model_waiting_offload_lists[model_idx].set()
    result, inf_lat, ttft = inference_func(encoded_input, model_idx)
    # result, inf_lat, ttft = inference_func_cotdm(encoded_input, model_idx)

    if global_config.DEBUG_MODE_SCHEDULER:
        print("global var cur_request_num:", global_var.cur_request_num.get())
    if global_var.cur_request_num.decrementAndGet() == 0:
        if global_config.DEBUG_MODE_SCHEDULER:
            print("global var cur_request_num decrease:", global_var.cur_request_num.get())
        global_var.scheduler_inc_sem.release()

    global_var.model_request_num_list[model_idx] -= 1

    return result, inf_lat, ttft


async def inference_then_offload_sync(input_text, model_name, slo=1.0):
    model_idx = await get_model_idx(model_name, slo)
    if model_idx == -1:
        return None, None, None, None

    result, inf_lat, ttft = await inference_by_deploy_scheme(input_text, model_idx)

    offload_thread = Thread(target=offload_func, args=(model_idx,))
    offload_thread.start()

    offload_start_time = time.time()
    offload_thread.join()
    offload_end_time = time.time()

    offload_time = offload_end_time - offload_start_time

    return result, inf_lat, ttft, offload_time


async def inference_then_offload_async(input_text, model_name, slo=1.0):
    model_idx = await get_model_idx(model_name, slo)
    if model_idx == -1:
        return None, None, None

    result, inf_lat, ttft = await inference_by_deploy_scheme(input_text, model_idx)

    offload_thread = Thread(
        target=offload_func, args=(model_idx, global_config.DEFAULT_KEEP_ALIVE_TIME,)
    )
    offload_thread.start()

    return result, inf_lat, ttft


async def inference_return_with_offload_thread(input_text, model_name, slo=1.0):
    model_idx = await get_model_idx(model_name, slo)
    if model_idx == -1:
        return None, None, None, None

    result, inf_lat, ttft = await inference_by_deploy_scheme(input_text, model_idx)

    offload_thread = Thread(
        target=offload_func,
        args=(
            model_idx,
            global_config.DEFAULT_KEEP_ALIVE_TIME,
        ),
    )
    offload_thread.start()

    return result, inf_lat, ttft, offload_thread


async def handle_inference_with_offload_sync(request):
    data = await request.json()
    model_name = data.get("model_name", None)
    input_text = data.get("input_text", None)
    if model_name is None:
        return web.json_response({"text": "Model name is empty!", "inf_lat": -1, "ttft": -1, "offload_time": -1})

    if input_text is None:
        return web.json_response({"text": "Input is empty!", "inf_lat": -1, "ttft": -1, "offload_time": -1})

    result, inf_lat, ttft, offload_time = await inference_then_offload_sync(input_text, model_name)
    if result is None:
        return web.json_response({"text": "Inference failed!", "inf_lat": -1, "ttft": -1, "offload_time": -1})

    data = {"text": result, "inf_lat": inf_lat, "ttft": ttft, "offload_time": offload_time}
    return web.json_response(data)


async def handle_inference_with_offload_async(request):
    data = await request.json()
    model_name = data.get("model_name", None)
    input_text = data.get("input_text", None)
    if model_name is None:
        return web.json_response({"text": "Model name is empty!", "inf_lat": -1, "ttft": -1})

    if input_text is None:
        return web.json_response({"text": "Input is empty!", "inf_lat": -1, "ttft": -1})

    result, inf_lat, ttft = await inference_then_offload_async(input_text, model_name)
    if result is None:
        return web.json_response({"text": "Inference failed!", "inf_lat": -1, "ttft": -1})

    data = {"text": result, "inf_lat": inf_lat, "ttft": ttft}
    print('Inference model "{}" done.'.format(model_name))
    return web.json_response(data)
    

async def start_server_with_offload_sync(port=8080):
    app = web.Application()
    app.router.add_post("/inference", handle_inference_with_offload_sync)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()

    return runner, site


async def close_server(server_site, server_runner):
    global_var.is_system_running = False
    is_model_using_lists_len = len(global_var.is_model_using_lists)
    for i in range(is_model_using_lists_len):
        global_var.is_model_using_lists[i].wait()
    offload_all_models_from_gpu_to_cpu()
    await server_site.stop()
    await server_runner.cleanup()