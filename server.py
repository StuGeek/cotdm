from aiohttp import web
from submit.submit_model_bert import submit_model_bert
from submit.submit_model_gpt2 import submit_model_gpt2
from init.init_tools import init_deploy
from server_utils import handle_inference_with_offload_async
from global_data import global_var

# import gc
# gc.disable()

model_name_list = ["bert-large-uncased", "gpt2-medium"]


async def handle_submit_model(request):
    data = await request.json()
    model_name = data.get("model_name", None)
    slo = data.get("slo", 1.0)

    try:
        if model_name is None:
            return web.Response(text="Model does not exists!")
        elif model_name.find("bert") == 0:
            await submit_model_bert(model_name, slo)
        elif model_name.find("gpt2") == 0:
            await submit_model_gpt2(model_name, slo)
        else:
            return web.Response(text=f"The model type of \"{model_name}\" is not supported.")
    except:
        return web.Response(text=f"Submit model \"{model_name}\" failed!")

    print(f"Successfully submitted model \"{model_name}\"!")
    return web.Response(text=f"Successfully submitted model \"{model_name}\"!")


async def start_server():
    await init_deploy(model_name_list)

    app = web.Application()
    app.add_routes([web.post("/submit", handle_submit_model)])
    app.add_routes([web.post("/inference", handle_inference_with_offload_async)])

    return app


if __name__ == "__main__":
    web.run_app(start_server())
    global_var.is_system_running = False
