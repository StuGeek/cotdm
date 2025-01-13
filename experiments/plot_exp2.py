import numpy as np
import json
import os
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
from global_data import global_config
from plot_exp1 import get_latest_exp_res_file


def get_goodput(e2e_lat_list, interval_time, slo_list):
    good_num = 0
    slo_list_len = len(slo_list)
    for i in range(len(e2e_lat_list)):
        if e2e_lat_list[i] <= slo_list[i % slo_list_len]:
            good_num += 1
    return good_num / interval_time


def get_sloatta(e2e_lat_list, slo_list):
    good_num = 0
    total_num = len(e2e_lat_list)
    slo_list_len = len(slo_list)
    for i in range(total_num):
        if e2e_lat_list[i] <= slo_list[i % slo_list_len]:
            good_num += 1
    return good_num / total_num * 100


def get_latp99_and_goodput_list(e2e_lat_list_all_workload, duration_list, slo_list):
    lat_p99_list_all_workload = []
    goodput_list_all_workload = []
    sloatta_list_all_workload = []

    for i in range(len(e2e_lat_list_all_workload)):
        e2e_lat_list = e2e_lat_list_all_workload[i]
        interval_time = duration_list[i] + e2e_lat_list[-1]

        sorted_e2e_lat_list = sorted(e2e_lat_list)
        lat_p99 = sorted_e2e_lat_list[int(0.99 * len(sorted_e2e_lat_list))]
        
        goodput = get_goodput(e2e_lat_list, interval_time, slo_list)
        sloatta = get_sloatta(e2e_lat_list, slo_list)

        lat_p99_list_all_workload.append(lat_p99)
        goodput_list_all_workload.append(goodput)
        sloatta_list_all_workload.append(sloatta)

    return lat_p99_list_all_workload, goodput_list_all_workload, sloatta_list_all_workload


def get_nor_latp99_and_goodput_list(e2e_lat_list_all_schemes, duration_list, slo_list):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    deploy_scheme_num = len(deploy_scheme_dict)
    scheme_idx_totoff = deploy_scheme_dict["Totally-Offload"]

    nor_lat_p99_list_all_schemes = [[] for i in range(deploy_scheme_num)]
    nor_goodput_list_all_schemes = [[] for i in range(deploy_scheme_num)]
    sloatta_list_all_schemes = [[] for i in range(deploy_scheme_num)]
    e2e_lat_list_all_workload_totoff = e2e_lat_list_all_schemes[scheme_idx_totoff]
    lat_p99_list_all_workload_totoff, goodput_list_all_workload_totoff, sloatta_list_all_workload_totoff = get_latp99_and_goodput_list(e2e_lat_list_all_workload_totoff, duration_list, slo_list)
    sloatta_list_all_schemes[scheme_idx_totoff] = sloatta_list_all_workload_totoff

    nor_deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan", "CoTDM"]
    for scheme in nor_deploy_scheme_list:
        scheme_idx = deploy_scheme_dict[scheme]
        e2e_lat_list_all_workload = e2e_lat_list_all_schemes[scheme_idx]
        lat_p99_list_all_workload, goodput_list_all_workload, sloatta_list_all_workload = get_latp99_and_goodput_list(e2e_lat_list_all_workload, duration_list, slo_list)

        nor_lat_p99_list = [lat_p99_list_all_workload[j] / lat_p99_list_all_workload_totoff[j] for j in range(len(lat_p99_list_all_workload))]
        nor_goodput_list = [goodput_list_all_workload[j] / goodput_list_all_workload_totoff[j] for j in range(len(goodput_list_all_workload))]
        
        nor_lat_p99_list_all_schemes[scheme_idx] = nor_lat_p99_list
        nor_goodput_list_all_schemes[scheme_idx] = nor_goodput_list
        sloatta_list_all_schemes[scheme_idx] = sloatta_list_all_workload

    return nor_lat_p99_list_all_schemes, nor_goodput_list_all_schemes, sloatta_list_all_schemes


def plot_fig_nor_lat_p99(nor_lat_p99_list_all_model_set, cv_list):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    model_set_num = len(nor_lat_p99_list_all_model_set)
    fig, axes = plt.subplots(1, model_set_num, figsize=(model_set_num * 4 + 1, 3))
    deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan", "CoTDM"]

    for i in range(model_set_num):
        ax = axes.flat[i]
        nor_lat_p99_list_all_cv = []

        for deploy_scheme in deploy_scheme_list:
            scheme_idx = deploy_scheme_dict[deploy_scheme]
            nor_lat_p99_list_all_cv.append(nor_lat_p99_list_all_model_set[i][scheme_idx])

        marker_list = ["o", "s", "v", "^"]
        for j in range(len(deploy_scheme_list)):
            ax.plot(
                cv_list,
                nor_lat_p99_list_all_cv[j],
                marker=marker_list[j],
                markersize=3,
                label=deploy_scheme_list[j],
            )

        ax.grid()
        ax.set_xlabel("Request fluctuation(cv)", fontsize=13)
        ax.set_ylabel("Normalized 99% tail latency", fontsize=10)
        ax.set_title("Model Set S" + str(i + 1), fontsize=13)

    plt.legend(bbox_to_anchor=(0.3, 1.45), ncol=4, fontsize=12, edgecolor='grey')
    fig.subplots_adjust(
        left=0.125, bottom=0.15, right=0.89, top=0.69, wspace=0.25, hspace=0.2
    )
    plt.savefig("exp2_nor_lat_p99.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp2_nor_lat_p99.png", os.path.dirname(os.path.abspath(__file__)) + "/exp2_nor_lat_p99.png"))


def plot_fig_nor_goodput(nor_goodput_list_all_model_set, cv_list):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    model_set_num = len(nor_goodput_list_all_model_set)
    fig, axes = plt.subplots(1, model_set_num, figsize=(model_set_num * 4 + 1, 3))
    deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan", "CoTDM"]

    for i in range(model_set_num):
        ax = axes.flat[i]
        nor_goodput_list_all_cv = []

        for deploy_scheme in deploy_scheme_list:
            scheme_idx = deploy_scheme_dict[deploy_scheme]
            nor_goodput_list_all_cv.append(nor_goodput_list_all_model_set[i][scheme_idx])

        marker_list = ["o", "s", "v", "^"]
        for j in range(len(deploy_scheme_list)):
            ax.plot(
                cv_list,
                nor_goodput_list_all_cv[j],
                marker=marker_list[j],
                markersize=3,
                label=deploy_scheme_list[j],
            )

        ax.grid()
        ax.set_xlabel("Request fluctuation(cv)", fontsize=13)
        ax.set_ylabel("Normalized goodput", fontsize=11)
        ax.set_yticks(np.arange(1.0, 1.5, 0.1))
        ax.set_title("Model Set S" + str(i + 1), fontsize=13)

    plt.legend(bbox_to_anchor=(0.3, 1.45), ncol=4, fontsize=12, edgecolor='grey')
    fig.subplots_adjust(
        left=0.125, bottom=0.15, right=0.89, top=0.69, wspace=0.25, hspace=0.2
    )
    plt.savefig("exp2_nor_goodput.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp2_nor_goodput.png", os.path.dirname(os.path.abspath(__file__)) + "/exp2_nor_goodput.png"))


def plot_fig_sloatta(sloatta_list_all_model_set, cv_list):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    model_set_num = len(sloatta_list_all_model_set)
    fig, axes = plt.subplots(1, model_set_num, figsize=(model_set_num * 4 + 1, 3))
    deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan", "CoTDM"]

    for i in range(model_set_num):
        ax = axes.flat[i]
        sloatta_list_all_cv = []

        for deploy_scheme in deploy_scheme_list:
            scheme_idx = deploy_scheme_dict[deploy_scheme]
            sloatta_list_all_cv.append(sloatta_list_all_model_set[i][scheme_idx])

        marker_list = ["o", "s", "v", "^"]
        for j in range(len(deploy_scheme_list)):
            ax.plot(
                cv_list,
                sloatta_list_all_cv[j],
                marker=marker_list[j],
                markersize=3,
                label=deploy_scheme_list[j],
            )

        ax.grid()
        ax.set_xlabel("Request fluctuation(cv)", fontsize=13)
        ax.set_ylabel("SLO Attainment(%)", fontsize=11)
        ax.set_title("Model Set S" + str(i + 1), fontsize=13)

    plt.legend(bbox_to_anchor=(0.3, 1.45), ncol=4, fontsize=12, edgecolor='grey')
    fig.subplots_adjust(
        left=0.125, bottom=0.15, right=0.89, top=0.69, wspace=0.25, hspace=0.2
    )
    plt.savefig("exp2_sloatta.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp2_sloatta.png", os.path.dirname(os.path.abspath(__file__)) + "/exp2_sloatta.png"))


def plot_fig_exp2(e2e_lat_list_all_model_set, cv_list, duration_list, slo_lists):
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (7, 2)

    nor_lat_p99_list_all_model_set = []
    nor_goodput_list_all_model_set = []
    sloatta_list_all_model_set = []

    for i in range(len(e2e_lat_list_all_model_set)):
        e2e_lat_list_all_schemes = e2e_lat_list_all_model_set[i]
        nor_lat_p99_list_all_schemes, nor_goodput_list_all_schemes, sloatta_list_all_schemes = (
            get_nor_latp99_and_goodput_list(e2e_lat_list_all_schemes, duration_list, slo_lists[i])
        )
        nor_lat_p99_list_all_model_set.append(nor_lat_p99_list_all_schemes)
        nor_goodput_list_all_model_set.append(nor_goodput_list_all_schemes)
        sloatta_list_all_model_set.append(sloatta_list_all_schemes)

    plt.figure(1)
    plot_fig_nor_lat_p99(nor_lat_p99_list_all_model_set, cv_list)
    plt.figure(2)
    plot_fig_nor_goodput(nor_goodput_list_all_model_set, cv_list)
    plt.figure(3)
    plot_fig_sloatta(sloatta_list_all_model_set, cv_list)
    plt.show()


def print_perf_improve_exp2(e2e_lat_list_all_model_set, cv_list, duration_list, slo_lists):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    scheme_idx_cotdm = deploy_scheme_dict["CoTDM"]

    print("Performance improvement:")

    for i in range(len(e2e_lat_list_all_model_set)):
        print("For modelset {},".format(i + 1))

        e2e_lat_list_all_schemes = e2e_lat_list_all_model_set[i]
        nor_lat_p99_list_all_schemes, nor_goodput_list_all_schemes, sloatta_list_all_schemes = get_nor_latp99_and_goodput_list(e2e_lat_list_all_schemes, duration_list, slo_lists[i])
        nor_lat_p99_list_all_workload_cotdm = nor_lat_p99_list_all_schemes[scheme_idx_cotdm]
        nor_goodput_list_all_workload_cotdm = nor_goodput_list_all_schemes[scheme_idx_cotdm]
        sloatta_list_all_workload_cotdm = sloatta_list_all_schemes[scheme_idx_cotdm]

        compare_deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan"]
        for scheme in compare_deploy_scheme_list:
            print("CoTDM vs {}: Normalized 99% tail latency, Normalized goodput, SLO attainment".format(scheme))
            scheme_idx = deploy_scheme_dict[scheme]
            nor_lat_p99_list_all_workload = nor_lat_p99_list_all_schemes[scheme_idx]
            nor_goodput_list_all_workload = nor_goodput_list_all_schemes[scheme_idx]
            sloatta_list_all_workload = sloatta_list_all_schemes[scheme_idx]
    
            compare_lat_p99_list = [(nor_lat_p99_list_all_workload[j] - nor_lat_p99_list_all_workload_cotdm[j]) / nor_lat_p99_list_all_workload[j] for j in range(len(nor_lat_p99_list_all_workload))]
            compare_goodput_list = [(nor_goodput_list_all_workload_cotdm[j] - nor_goodput_list_all_workload[j]) / nor_goodput_list_all_workload[j] for j in range(len(nor_goodput_list_all_workload))]
            compare_sloatta_list = [sloatta_list_all_workload_cotdm[j] - sloatta_list_all_workload[j] for j in range(len(sloatta_list_all_workload))]

            for j in range(len(compare_lat_p99_list)):
                print(
                    "cv={:.2f}, {:.2f}%, {:.2f}%, {:.2f}%".format(
                        cv_list[j], compare_lat_p99_list[j] * 100, compare_goodput_list[j] * 100, compare_sloatta_list[j]
                    )
                )

            avg_compare_lat_p99 = sum(compare_lat_p99_list) / len(compare_lat_p99_list)
            avg_compare_goodput = sum(compare_goodput_list) / len(compare_goodput_list)
            avg_compare_sloatta = sum(compare_sloatta_list) / len(compare_sloatta_list)
            print(
                "Average: {:.2f}%, {:.2f}%, {:.2f}%\n".format(
                    avg_compare_lat_p99 * 100,
                    avg_compare_goodput * 100,
                    avg_compare_sloatta,
                )
            )


def main():
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    deploy_scheme_num = len(deploy_scheme_dict)
    modelset_num = 3
    e2e_lat_list_all_model_set = []
    slo_lists = []
    exp2_res_dir = global_config.EXP_RES_DIR + "exp2/"
    deploy_scheme_list = ["totoff", "strawman", "alpaserve", "deepplan", "cotdm"]
    for i in range(modelset_num):
        e2e_lat_list_all_schemes = [[] for i in range(deploy_scheme_num)]

        for scheme in deploy_scheme_list:
            exp_file_path = get_latest_exp_res_file(exp2_res_dir, "exp2_modelset" + str(i + 1) + "_" + scheme)
            if not os.path.exists(exp_file_path):
                raise ValueError("{} doesn't exists!".format(exp_file_path))
            else:
                print("Running experiment2 data file path: {}".format(exp_file_path))

            with open(exp_file_path, "r") as f:
                json_data = json.load(f)

            deploy_scheme = json_data["deploy_scheme"]
            scheme_idx = deploy_scheme_dict[deploy_scheme]

            e2e_lat_list_all_workload = json_data["e2e_lat_list_all_workload"]
            e2e_lat_list_all_schemes[scheme_idx] = e2e_lat_list_all_workload

        e2e_lat_list_all_model_set.append(e2e_lat_list_all_schemes)

        cv_list = json_data["cv_list"]
        duration_list = json_data["duration_list"]
        avg_e2e_lat_ready = json_data["avg_e2e_lat_ready"]
        SLO_SCALE = 7
        slo_list = [round(lat * SLO_SCALE, 3) for lat in avg_e2e_lat_ready]
        slo_lists.append(slo_list)

    print_perf_improve_exp2(e2e_lat_list_all_model_set, cv_list, duration_list, slo_lists)
    plot_fig_exp2(e2e_lat_list_all_model_set, cv_list, duration_list, slo_lists)


if __name__ == "__main__":
    main()
