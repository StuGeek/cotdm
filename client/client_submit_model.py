import aiohttp
import asyncio
import argparse


async def submit_model(model_name, port=8080):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://127.0.0.1:" + str(port) + "/submit", json={"model_name": model_name}
        ) as response:
            res = await response.text()
            print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="bert-large-uncased")
    args = parser.parse_args()
    model_name = args.name
    asyncio.run(submit_model(model_name))
