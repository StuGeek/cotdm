import aiohttp
import asyncio
import argparse
import time
import sys
sys.path.append(".")
sys.path.append("..")
from global_data import global_config


async def send_single_inf_req(model_name, input_text, arrive_time=0, port=8080):
    await asyncio.sleep(arrive_time)
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:" + str(port) + "/inference",
            json={"model_name": model_name, "input_text": input_text},
        ) as response:
            res = await response.json(content_type="application/json", encoding="utf-8")
            end_time = time.time()
            e2e_lat = end_time - start_time
            if "offload_time" in res:
                e2e_lat -= res["offload_time"]
            inf_lat = res["inf_lat"]
            ttft = res["ttft"]
            # print("end to end start_time:", start_time)
            # print("end to end latency:", e2e_lat)
            # print("inference latency:", inf_lat)
            # print("extra time:", e2e_lat - inf_lat)
            # print(res)
    return res, inf_lat, ttft, e2e_lat


async def send_single_inf_req_with_submit_idx(
    model_name, input_text, submit_idx, arrive_time=0, port=8080
):
    await asyncio.sleep(arrive_time)
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:" + str(port) + "/inference",
            json={
                "model_name": model_name,
                "input_text": input_text,
                "submit_idx": submit_idx,
            },
        ) as response:
            res = await response.json(content_type="application/json", encoding="utf-8")
            end_time = time.time()
            e2e_lat = end_time - start_time
            if "offload_time" in res:
                e2e_lat -= res["offload_time"]
            text = res["text"]
            inf_lat = res["inf_lat"]
            ttft = res["ttft"]
            # print("end to end start_time:", start_time)
            # print("end to end latency:", e2e_lat)
            # print("inference latency:", inf_lat)
            # print("extra time:", e2e_lat - inf_lat)
            # print(res)
    return text, inf_lat, ttft, e2e_lat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="bert-large-uncased")
    args = parser.parse_args()
    model_name = args.name

    asyncio.run(send_single_inf_req(model_name, global_config.DEFAULT_INPUT_TEXT_BERT))
