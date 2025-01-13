import numpy as np
import json
import os
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
from global_data import global_config
import exp1


def plot_fig_mean_lat(model_name_list, mean_lat_list_all_schemes):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    deploy_scheme_list = ["Totally-Offload", "Strawman", "DeepPlan", "CoTDM"]
    mean_lat_list = []
    for deploy_scheme in deploy_scheme_list:
        scheme_idx = deploy_scheme_dict[deploy_scheme]
        mean_lat_list.append(mean_lat_list_all_schemes[scheme_idx])

    x = np.arange(len(model_name_list))
    width = 0.15

    plt.grid(visible=True, axis="y", zorder=0)
    for i in range(len(mean_lat_list)):
        plt.bar(
            x - (1.5 - i) * width,
            mean_lat_list[i],
            width=width,
            edgecolor="black",
            zorder=100,
            label=deploy_scheme_list[i],
        )
    plt.xticks(x, labels=model_name_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Latency(s)", fontsize=14)
    plt.legend(bbox_to_anchor=(-0.1, 0.95, 1.05, 0.3), ncol=4, fontsize=9)
    plt.savefig("exp1_mean_lat.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp1_mean_lat.png", os.path.dirname(os.path.abspath(__file__)) + "/exp1_mean_lat.png"))


def plot_fig_thr(model_name_list, thr_list_all_schemes):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    deploy_scheme_list = ["Totally-Offload", "Strawman", "DeepPlan", "CoTDM"]
    thr_list = []
    for deploy_scheme in deploy_scheme_list:
        scheme_idx = deploy_scheme_dict[deploy_scheme]
        thr_list.append(thr_list_all_schemes[scheme_idx])

    x = np.arange(len(model_name_list))
    width = 0.15

    plt.grid(visible=True, axis="y", zorder=0)
    for i in range(len(thr_list)):
        plt.bar(
            x - (1.5 - i) * width,
            thr_list[i],
            width=width,
            edgecolor="black",
            zorder=100,
            label=deploy_scheme_list[i],
        )
    plt.xticks(x, labels=model_name_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Throughput(reqs/s)", fontsize=10)
    plt.legend(bbox_to_anchor=(-0.1, 0.95, 1.05, 0.3), ncol=4, fontsize=9)
    plt.savefig("exp1_thr.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp1_thr.png", os.path.dirname(os.path.abspath(__file__)) + "/exp1_thr.png"))


def plot_fig_nor_cost(model_name_list, cost_list_all_schemes):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    deploy_scheme_list = ["Strawman", "DeepPlan", "CoTDM"]
    scheme_idx_ready = deploy_scheme_dict["Ready"]
    cost_list_ready = cost_list_all_schemes[scheme_idx_ready]
    nor_cost_list = []
    for deploy_scheme in deploy_scheme_list:
        scheme_idx = deploy_scheme_dict[deploy_scheme]
        cost_list = cost_list_all_schemes[scheme_idx]
        nor_cost_list.append(
            [
                cost_list[i] / cost_list_ready[i]
                for i in range(len(cost_list))
            ]
        )

    x = np.arange(len(model_name_list))
    width = 0.15

    plt.grid(visible=True, axis="y", zorder=0)
    for i in range(len(nor_cost_list)):
        plt.bar(
            x - (1 - i) * width,
            nor_cost_list[i],
            width=width,
            edgecolor="black",
            zorder=100,
            label=deploy_scheme_list[i],
        )
    plt.xticks(x, labels=model_name_list, fontsize=10)
    plt.yticks(np.arange(0, 1, 0.2), fontsize=10)
    plt.ylabel("Normalized deployment cost", fontsize=9)
    plt.legend(bbox_to_anchor=(-0.1, 0.95, 0.91, 0.3), ncol=3, fontsize=9)
    plt.savefig("exp1_nor_cost.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp1_nor_cost.png", os.path.dirname(os.path.abspath(__file__)) + "/exp1_nor_cost.png"))


def plot_fig_exp1(model_name_list, mean_lat_list_all_schemes, thr_list_all_schemes, cost_list_all_schemes):
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (7, 2)

    plt.figure(1)
    plot_fig_mean_lat(model_name_list, mean_lat_list_all_schemes)
    plt.figure(2)
    plot_fig_thr(model_name_list, thr_list_all_schemes)
    plt.figure(3)
    plot_fig_nor_cost(model_name_list, cost_list_all_schemes)
    plt.show()


def print_perf_improve_exp1(model_name_list, mean_lat_list_all_schemes, thr_list_all_schemes, cost_list_all_schemes):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    scheme_idx_cotdm = deploy_scheme_dict["CoTDM"]

    mean_lat_list_cotdm = mean_lat_list_all_schemes[scheme_idx_cotdm]
    thr_list_cotdm = thr_list_all_schemes[scheme_idx_cotdm]
    cost_list_cotdm = cost_list_all_schemes[scheme_idx_cotdm]

    print("Performance improvement:")

    compare_deploy_scheme_list = ["Totally-Offload", "Strawman", "DeepPlan"]
    for scheme in compare_deploy_scheme_list:
        if scheme == "Totally-Offload":
            print("CoTDM vs Totally-Offload: Mean latency, Throughput")
        else:
            print(
                "CoTDM vs {}: Mean latency, Throughput, Normalized cost".format(scheme)
            )

        scheme_idx = deploy_scheme_dict[scheme]
        mean_lat_list = mean_lat_list_all_schemes[scheme_idx]
        thr_list = thr_list_all_schemes[scheme_idx]
        cost_list = cost_list_all_schemes[scheme_idx]

        mean_lat_compare_list = [
            (mean_lat_list[i] - mean_lat_list_cotdm[i]) / mean_lat_list[i]
            for i in range(len(mean_lat_list))
        ]
        thr_compare_list = [
            (thr_list_cotdm[i] - thr_list[i]) / thr_list[i]
            for i in range(len(thr_list))
        ]
        if scheme != "Totally-Offload":
            cost_compare_list = [
                (cost_list[i] - cost_list_cotdm[i]) / cost_list[i]
                for i in range(len(cost_list))
            ]

        for i in range(len(model_name_list)):
            print(
                "{}: {:.2f}%, {:.2f}%".format(
                    model_name_list[i],
                    mean_lat_compare_list[i] * 100,
                    thr_compare_list[i] * 100,
                ),
                end="",
            )
            if scheme != "Totally-Offload":
                print(", {:.2f}%".format(cost_compare_list[i] * 100), end="")
            print("")
        print(
                "Average: {:.2f}%, {:.2f}%".format(
                    sum(mean_lat_compare_list) / len(mean_lat_compare_list) * 100,
                    sum(thr_compare_list) / len(thr_compare_list) * 100,
                ),
                end="",
            )
        if scheme != "Totally-Offload":
            print(
                ", {:.2f}%".format(
                    sum(cost_compare_list) / len(cost_compare_list) * 100
                ),
                end="",
            )
        print("\n")


def get_latest_exp_res_file(exp_res_dir, file_name):
    latest_file_name = None    
    for filename in os.listdir(exp_res_dir):
        if filename.startswith(file_name) and filename.endswith(".json"):
            if latest_file_name is None or latest_file_name < filename:
                latest_file_name = filename

    if latest_file_name is None:
        return None
    
    latest_exp_res_file = exp_res_dir + latest_file_name
    return latest_exp_res_file


def beautify_model_name_list(model_name_list):
    new_model_name_list = []
    for i in range(len(model_name_list)):
        model_name = model_name_list[i]
        new_model_name = model_name
        if len(model_name) > 15:
            beautify_idx = model_name.rfind("-") + 1
            new_model_name = model_name[0:beautify_idx] + "\n" + model_name[beautify_idx:]
        new_model_name_list.append(new_model_name)

    return new_model_name_list


def main():
    exp1_res_dir = global_config.EXP_RES_DIR + "exp1/"
    exp_file_path = get_latest_exp_res_file(exp1_res_dir, "exp1")
    if exp_file_path is None:
        raise ValueError("Experiment1 doesn't have any result data file yet, please run \"python exp1.py\"!")
    else:
        print("Reading experiment1 result file: {}".format(exp_file_path))

    with open(exp_file_path, "r") as f:
        json_data = json.load(f)

    mean_lat_list_all_schemes = json_data["mean_lat_list_all_schemes"]
    thr_list_all_schemes = json_data["thr_list_all_schemes"]
    cost_list_all_schemes = json_data["cost_list_all_schemes"]

    print_perf_improve_exp1(exp1.model_name_list, mean_lat_list_all_schemes, thr_list_all_schemes, cost_list_all_schemes)
    model_name_list = beautify_model_name_list(exp1.model_name_list)
    plot_fig_exp1(model_name_list, mean_lat_list_all_schemes, thr_list_all_schemes, cost_list_all_schemes)


if __name__ == "__main__":
    main()
