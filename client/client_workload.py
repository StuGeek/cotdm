import asyncio
import sys
sys.path.append(".")
sys.path.append("..")
from client.client_single_inf_req import send_single_inf_req, send_single_inf_req_with_submit_idx
from global_data.global_class import SingleInferenceRequest, WorkloadInferenceRequest

async def send_request_by_workload(workload_req_list):
    task_list = []
    inf_req_list = workload_req_list.inf_req_list
    arri_time_list = workload_req_list.arri_time_list
    req_list_len = len(inf_req_list)
    for i in range(req_list_len):
        task = asyncio.create_task(
            send_single_inf_req(
                inf_req_list[i].model_name,
                inf_req_list[i].input_text,
                arri_time_list[i],
            )
        )
        task_list.append(task)

    results = await asyncio.gather(*task_list)
    e2e_lat_list = [result[3] for result in results]
    return e2e_lat_list

async def send_request_by_workload_with_submit_idx(workload_req_list, port=8080):
    task_list = []
    e2e_lat_list = []
    inf_req_list = workload_req_list.inf_req_list
    arri_time_list = workload_req_list.arri_time_list
    req_list_len = len(inf_req_list)
    for i in range(req_list_len):
        task = asyncio.create_task(
            send_single_inf_req_with_submit_idx(
                inf_req_list[i].model_name,
                inf_req_list[i].input_text,
                i,
                arri_time_list[i],
                port,
            )
        )
        task_list.append(task)

    results = await asyncio.gather(*task_list)
    e2e_lat_list = [result[3] for result in results]
    return e2e_lat_list

if __name__ == "__main__":
    DEFAULT_INPUT_TEXT_BERT = "The Emancipation Proclamation [MASK] votes for Republicans in rural New England and the upper Midwest, but cost votes in the Irish and German strongholds and in the lower Midwest, where many Southerners had lived for generations.[SEP]In the spring of 1863 Lincoln was sufficiently optimistic about upcoming military campaigns to think the end of the war could be near; the plans included attacks by Hooker on Lee north of Richmond, Rosecrans on Chattanooga, Grant on Vicksburg, and a naval assault on Charleston."
    DEFAULT_INPUT_TEXT_GPT2 = "You can now dive in the comfort of your Ford Explorer red lights, which let you take exquisite picture. Open the lens and aim to take a blurry window that shows a Ford Norris road that's not registered at Flickr. And, view each car in full view? You can use both of your Leica reflex sightlamoms, boosting your perceived resolution to 3160pfx when shooting in north-west California. (And that's right, driving always leaves a innocent minor site \u2014 like Flickr) Read more about runs in Oakland Auto Repair. The big goal: the drive-thru feuds where one manufacturer does just fine and the other manufacturer seems driven to get away with just fine.\n\nIn other words, don't get saved for good \u2013 captured photos and closed-circuit TV wins none of those kind of plicks.\n\nSeriously. In case there wasn't enough \u2014 you can try Huawei AimBit zoomless shooting below by using your camera with a desired zoom level. Again, that is, it means first thing in all the morning, then slowly losing the ability to view the final passage of the image when you look up. While my Ford Explorer is good, great and just plain fun, expecting to run the lights down, that bit of a little tweaking may prove scaring the hell out of you.\n\nLike this: Like Loading..."

    model_name_list = ["bert-large-uncased", "gpt2-medium"]
    arri_time_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    model_name_list_len = len(model_name_list)
    inf_req_list = []
    for i in range(len(arri_time_list)):
        if i % 2 == 0:
            req = SingleInferenceRequest(model_name_list[0], DEFAULT_INPUT_TEXT_BERT)
        else:
            req = SingleInferenceRequest(model_name_list[1], DEFAULT_INPUT_TEXT_GPT2)
        inf_req_list.append(req)

    workload_list = WorkloadInferenceRequest(inf_req_list, arri_time_list)
    asyncio.run(send_request_by_workload_with_submit_idx(workload_list))