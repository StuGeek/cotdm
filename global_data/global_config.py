import torch
from init.init_deploy_cotdm import deploy_single_model_cotdm
from init.init_deploy_ready import deploy_single_model_ready
from init.init_deploy_totoff import deploy_single_model_totoff
from init.init_deploy_strawman import deploy_single_model_strawman
from init.init_deploy_alpaserve import deploy_single_model_alpaserve
from init.init_deploy_deepplan import deploy_single_model_deepplan
from tools_inference.inference_func import (
    inference_func_cotdm,
    inference_func_ready,
    inference_func_totoff,
    inference_func_strawman,
    inference_func_alpaserve,
    inference_func_deepplan,
)

DATA_SAVE_DIR = "/data/yzk/"
MODEL_SAVE_DIR = DATA_SAVE_DIR + "splitted_model/"
MODEL_CONFIG_DIR = DATA_SAVE_DIR + "model_configs/"
EXP_RES_DIR = DATA_SAVE_DIR + "experiment_results/"

AZURE_V1_NAME = "azure_v1"
AZURE_V1_DIR = DATA_SAVE_DIR + "azurefunctions-dataset2019/azure_v1.pkl"

INIT_SLO_SCALE = 8

DEBUG_MODE_SCHEDULER = False

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# seq_len=512
DEFAULT_INPUT_TEXT_BERT = "The Emancipation Proclamation [MASK] votes for Republicans in rural New England and the upper Midwest, but cost votes in the Irish and German strongholds and in the lower Midwest, where many Southerners had lived for generations.[SEP]In the spring of 1863 Lincoln was sufficiently optimistic about upcoming military campaigns to think the end of the war could be near; the plans included attacks by Hooker on Lee north of Richmond, Rosecrans on Chattanooga, Grant on Vicksburg, and a naval assault on Charleston."
# seq_len=1237
DEFAULT_INPUT_TEXT_GPT2 = "You can now dive in the comfort of your Ford Explorer red lights, which let you take exquisite picture. Open the lens and aim to take a blurry window that shows a Ford Norris road that's not registered at Flickr. And, view each car in full view? You can use both of your Leica reflex sightlamoms, boosting your perceived resolution to 3160pfx when shooting in north-west California. (And that's right, driving always leaves a innocent minor site \u2014 like Flickr) Read more about runs in Oakland Auto Repair. The big goal: the drive-thru feuds where one manufacturer does just fine and the other manufacturer seems driven to get away with just fine.\n\nIn other words, don't get saved for good \u2013 captured photos and closed-circuit TV wins none of those kind of plicks.\n\nSeriously. In case there wasn't enough \u2014 you can try Huawei AimBit zoomless shooting below by using your camera with a desired zoom level. Again, that is, it means first thing in all the morning, then slowly losing the ability to view the final passage of the image when you look up. While my Ford Explorer is good, great and just plain fun, expecting to run the lights down, that bit of a little tweaking may prove scaring the hell out of you.\n\nLike this: Like Loading..."

DEFAULT_KEEP_ALIVE_TIME = 5

DEPLOY_ALG = "Half Mem"
DEPLOY_ALG_LIST = ["Min Deploy", "Half Mem"]

DEPLOY_SCHEME_DICT = {
    "Totally-Offload": 0,
    "Ready": 1,
    "Strawman": 2,
    "CoTDM": 3,
    "DeepPlan": 4,
    "AlpaServe": 5,
}

DEPLOY_SCHEME = "CoTDM"
INIT_DEPLOY_FUNC = deploy_single_model_cotdm
INFERENCE_FUNC = inference_func_cotdm


def set_deploy_scheme(deploy_scheme):
    global DEPLOY_SCHEME
    global INIT_DEPLOY_FUNC
    global INFERENCE_FUNC

    if deploy_scheme not in DEPLOY_SCHEME_DICT:
        raise ValueError(
            'The deploy scheme of "{}" is not supported.'.format(deploy_scheme)
        )
    elif deploy_scheme == "CoTDM":
        INIT_DEPLOY_FUNC = deploy_single_model_cotdm
        INFERENCE_FUNC = inference_func_cotdm
    elif deploy_scheme == "Ready":
        INIT_DEPLOY_FUNC = deploy_single_model_ready
        INFERENCE_FUNC = inference_func_ready
    elif deploy_scheme == "Totally-Offload":
        INIT_DEPLOY_FUNC = deploy_single_model_totoff
        INFERENCE_FUNC = inference_func_totoff
    elif deploy_scheme == "Strawman":
        INIT_DEPLOY_FUNC = deploy_single_model_strawman
        INFERENCE_FUNC = inference_func_strawman
    elif deploy_scheme == "AlpaServe":
        INIT_DEPLOY_FUNC = deploy_single_model_alpaserve
        INFERENCE_FUNC = inference_func_alpaserve
    elif deploy_scheme == "DeepPlan":
        INIT_DEPLOY_FUNC = deploy_single_model_deepplan
        INFERENCE_FUNC = inference_func_deepplan
    DEPLOY_SCHEME = deploy_scheme
