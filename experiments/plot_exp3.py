import numpy as np
import json
import os
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
from global_data import global_config
from plot_exp1 import get_latest_exp_res_file
from plot_exp2 import get_sloatta
import exp3
from exp3 import rate_scale_list, cv_scale_list, slo_scale_list

SLO_SCALE = 5

def get_sloatta_list(e2e_lat_list_all_workload, slo_list):
    slo_attainment_list_all_workload = []

    for e2e_lat_list in e2e_lat_list_all_workload:
        slo_attainment = get_sloatta(e2e_lat_list, slo_list)
        slo_attainment_list_all_workload.append(slo_attainment)

    return slo_attainment_list_all_workload


def get_sloatta_list_by_slo_scale(e2e_lat_list_all_workload, slo_list, slo_scale_list):
    slo_attainment_list_all_workload = []

    for slo_scale in slo_scale_list:
        slo_attainment = get_sloatta(e2e_lat_list_all_workload[0], [slo * slo_scale / SLO_SCALE for slo in slo_list])
        slo_attainment_list_all_workload.append(slo_attainment)

    return slo_attainment_list_all_workload


def plot_fig_rate_scale_slo_atta(rate_scale_list, rate_scale_e2e_lat_list_all_model_set, slo_lists):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    model_set_num = len(rate_scale_e2e_lat_list_all_model_set)
    fig, axes = plt.subplots(1, model_set_num, figsize=(model_set_num * 4 + 1, 3))
    deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan", "CoTDM"]

    for i in range(model_set_num):
        ax = axes.flat[i]
        rate_scale_slo_atta_all_cv_list = []

        for deploy_scheme in deploy_scheme_list:
            scheme_idx = deploy_scheme_dict[deploy_scheme]
            rate_scale_slo_atta_all_cv = get_sloatta_list(rate_scale_e2e_lat_list_all_model_set[i][scheme_idx], slo_lists[i])
            rate_scale_slo_atta_all_cv_list.append(rate_scale_slo_atta_all_cv)

        marker_list = ["o", "s", "v", "^"]
        for j in range(len(deploy_scheme_list)):
            ax.plot(
                rate_scale_list,
                rate_scale_slo_atta_all_cv_list[j],
                marker=marker_list[j],
                markersize=3,
                label=deploy_scheme_list[j],
            )

        ax.grid()
        ax.set_xlabel("Rate Scale", fontsize=13)
        ax.set_ylabel("SLO Attainment(%)", fontsize=10)
        ax.set_yticks(np.arange(86, 102, 2))
        ax.set_title("Model Set S" + str(i + 1), fontsize=13)

    plt.legend(bbox_to_anchor=(0.3, 1.45), ncol=4, fontsize=12, edgecolor='grey')
    fig.subplots_adjust(
        left=0.125, bottom=0.15, right=0.89, top=0.69, wspace=0.25, hspace=0.2
    )
    plt.savefig("exp3_rate_scale_sloatta.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp3_rate_scale_sloatta.png", os.path.dirname(os.path.abspath(__file__)) + "/exp3_rate_scale_sloatta.png"))


def plot_fig_cv_scale_slo_atta(cv_scale_list, cv_scale_e2e_lat_list_all_model_set, slo_lists):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    model_set_num = len(cv_scale_e2e_lat_list_all_model_set)
    fig, axes = plt.subplots(1, model_set_num, figsize=(model_set_num * 4 + 1, 3))
    deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan", "CoTDM"]

    for i in range(model_set_num):
        ax = axes.flat[i]
        cv_scale_slo_atta_all_cv_list = []

        for deploy_scheme in deploy_scheme_list:
            scheme_idx = deploy_scheme_dict[deploy_scheme]
            cv_scale_slo_atta_all_cv = get_sloatta_list(cv_scale_e2e_lat_list_all_model_set[i][scheme_idx], slo_lists[i])
            cv_scale_slo_atta_all_cv_list.append(cv_scale_slo_atta_all_cv)


        marker_list = ["o", "s", "v", "^"]
        for j in range(len(deploy_scheme_list)):
            ax.plot(
                cv_scale_list,
                cv_scale_slo_atta_all_cv_list[j],
                marker=marker_list[j],
                markersize=3,
                label=deploy_scheme_list[j],
            )

        ax.grid()
        ax.set_xlabel("CV Scale", fontsize=13)
        ax.set_ylabel("SLO Attainment(%)", fontsize=11)
        ax.set_xticks(np.arange(0.5, 3, 0.5))
        if i == 2:
            ax.set_yticks(np.arange(70, 102, 5))
        else:
            ax.set_yticks(np.arange(65, 102, 5))
        ax.set_title("Model Set S" + str(i + 1), fontsize=13)

    plt.legend(bbox_to_anchor=(0.3, 1.45), ncol=4, fontsize=12, edgecolor='grey')
    fig.subplots_adjust(
        left=0.125, bottom=0.15, right=0.89, top=0.69, wspace=0.25, hspace=0.2
    )
    plt.savefig("exp3_cv_scale_sloatta.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp3_cv_scale_sloatta.png", os.path.dirname(os.path.abspath(__file__)) + "/exp3_cv_scale_sloatta.png"))


def plot_fig_slo_scale_slo_atta(slo_scale_list, slo_scale_e2e_lat_list_all_model_set, slo_lists):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    model_set_num = len(slo_scale_e2e_lat_list_all_model_set)
    fig, axes = plt.subplots(1, model_set_num, figsize=(model_set_num * 4 + 1, 3))
    deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan", "CoTDM"]

    for i in range(model_set_num):
        ax = axes.flat[i]
        slo_scale_slo_atta_all_cv_list = []

        for deploy_scheme in deploy_scheme_list:
            scheme_idx = deploy_scheme_dict[deploy_scheme]
            slo_scale_slo_atta_all_cv = get_sloatta_list_by_slo_scale(slo_scale_e2e_lat_list_all_model_set[i][scheme_idx], slo_lists[i], slo_scale_list)
            slo_scale_slo_atta_all_cv_list.append(slo_scale_slo_atta_all_cv)


        marker_list = ["o", "s", "v", "^"]
        for j in range(len(deploy_scheme_list)):
            ax.plot(
                slo_scale_list,
                slo_scale_slo_atta_all_cv_list[j],
                marker=marker_list[j],
                markersize=3,
                label=deploy_scheme_list[j],
            )

        ax.grid()
        ax.set_xlabel("SLO Scale", fontsize=13)
        ax.set_ylabel("SLO Attainment(%)", fontsize=11)
        ax.set_xticks(np.arange(5, 8.5, 0.5))
        if i == 2:
            ax.set_yticks(np.arange(85, 101, 3))
        else:
            ax.set_yticks(np.arange(90, 101, 2))
        ax.set_title("Model Set S" + str(i + 1), fontsize=13)

    plt.legend(bbox_to_anchor=(0.3, 1.45), ncol=4, fontsize=12, edgecolor='grey')
    fig.subplots_adjust(
        left=0.125, bottom=0.15, right=0.89, top=0.69, wspace=0.25, hspace=0.2
    )
    plt.savefig("exp3_slo_scale_sloatta.png", bbox_inches="tight")
    print("Figure {} save path: {}".format("exp3_slo_scale_sloatta.png", os.path.dirname(os.path.abspath(__file__)) + "/exp3_slo_scale_sloatta.png"))


def plot_fig_exp3(e2e_lat_list_all_model_set, rate_scale_list, cv_scale_list, slo_scale_list, slo_lists):
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (7, 2)

    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    rate_scale_e2e_lat_list_all_model_set = []
    cv_scale_e2e_lat_list_all_model_set = []
    slo_scale_e2e_lat_list_all_model_set = []

    for i in range(len(e2e_lat_list_all_model_set)):
        e2e_lat_list_all_schemes = e2e_lat_list_all_model_set[i]

        rate_scale_e2e_lat_list_all_schemes = []
        cv_scale_e2e_lat_list_all_schemes = []
        slo_scale_e2e_lat_list_all_schemes = []

        for scheme in deploy_scheme_dict:
            scheme_idx = deploy_scheme_dict[scheme]
            e2e_lat_list_all_workload = e2e_lat_list_all_schemes[scheme_idx]

            if e2e_lat_list_all_workload == []:
                rate_scale_e2e_lat_list_all_workload = []
                cv_scale_e2e_lat_list_all_workload = []
                slo_scale_e2e_lat_list_all_workload = []
            else:
                rate_scale_e2e_lat_list_all_workload = e2e_lat_list_all_workload[0:5]
                cv_scale_e2e_lat_list_all_workload = [e2e_lat_list_all_workload[5]] + [e2e_lat_list_all_workload[2]] + e2e_lat_list_all_workload[6:]
                slo_scale_e2e_lat_list_all_workload = [e2e_lat_list_all_workload[2]]

            rate_scale_e2e_lat_list_all_schemes.append(rate_scale_e2e_lat_list_all_workload)
            cv_scale_e2e_lat_list_all_schemes.append(cv_scale_e2e_lat_list_all_workload)
            slo_scale_e2e_lat_list_all_schemes.append(slo_scale_e2e_lat_list_all_workload)

        rate_scale_e2e_lat_list_all_model_set.append(rate_scale_e2e_lat_list_all_schemes)
        cv_scale_e2e_lat_list_all_model_set.append(cv_scale_e2e_lat_list_all_schemes)
        slo_scale_e2e_lat_list_all_model_set.append(slo_scale_e2e_lat_list_all_schemes)

    plt.figure(1)
    plot_fig_rate_scale_slo_atta(rate_scale_list, rate_scale_e2e_lat_list_all_model_set, slo_lists)
    plt.figure(2)
    plot_fig_cv_scale_slo_atta(cv_scale_list, cv_scale_e2e_lat_list_all_model_set, slo_lists)
    plt.figure(3)
    plot_fig_slo_scale_slo_atta(slo_scale_list, slo_scale_e2e_lat_list_all_model_set, slo_lists)
    plt.show()


def print_perf_improve_exp3(e2e_lat_list_all_model_set, slo_lists):
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    scheme_idx_cotdm = deploy_scheme_dict["CoTDM"]

    print("Performance improvement:")

    for i in range(len(e2e_lat_list_all_model_set)):
        print("For modelset {},".format(i + 1))

        e2e_lat_list_all_schemes = e2e_lat_list_all_model_set[i]
        e2e_lat_list_all_workload_cotdm = e2e_lat_list_all_schemes[scheme_idx_cotdm]

        rate_scale_e2e_lat_list_all_workload_cotdm = e2e_lat_list_all_workload_cotdm[0:5]
        cv_scale_e2e_lat_list_all_workload_cotdm = [e2e_lat_list_all_workload_cotdm[5]] + [e2e_lat_list_all_workload_cotdm[2]] + e2e_lat_list_all_workload_cotdm[6:]
        slo_scale_e2e_lat_list_all_workload_cotdm = [e2e_lat_list_all_workload_cotdm[2]]

        rate_scale_slo_atta_list_all_workload_cotdm = get_sloatta_list(rate_scale_e2e_lat_list_all_workload_cotdm, slo_lists[i])
        cv_scale_slo_atta_list_all_workload_cotdm = get_sloatta_list(cv_scale_e2e_lat_list_all_workload_cotdm, slo_lists[i])
        slo_scale_slo_atta_list_all_workload_cotdm = get_sloatta_list_by_slo_scale(slo_scale_e2e_lat_list_all_workload_cotdm, slo_lists[i], slo_scale_list)

        compare_deploy_scheme_list = ["Strawman", "AlpaServe", "DeepPlan"]
        for scheme in compare_deploy_scheme_list:
            print("CoTDM vs {}: SLO attainment".format(scheme))
            scheme_idx = deploy_scheme_dict[scheme]

            e2e_lat_list_all_workload = e2e_lat_list_all_schemes[scheme_idx]
            rate_scale_e2e_lat_list_all_workload = e2e_lat_list_all_workload[0:5]
            cv_scale_e2e_lat_list_all_workload = [e2e_lat_list_all_workload[5]] + [e2e_lat_list_all_workload[2]] + e2e_lat_list_all_workload[6:]
            slo_scale_e2e_lat_list_all_workload = [e2e_lat_list_all_workload[2]]

            rate_scale_slo_atta_list_all_workload = get_sloatta_list(rate_scale_e2e_lat_list_all_workload, slo_lists[i])
            cv_scale_slo_atta_list_all_workload = get_sloatta_list(cv_scale_e2e_lat_list_all_workload, slo_lists[i])
            slo_scale_slo_atta_list_all_workload = get_sloatta_list_by_slo_scale(slo_scale_e2e_lat_list_all_workload, slo_lists[i], slo_scale_list)

            compare_rate_scale_slo_atta_list = [(rate_scale_slo_atta_list_all_workload_cotdm[j] - rate_scale_slo_atta_list_all_workload[j]) / rate_scale_slo_atta_list_all_workload[j] for j in range(len(rate_scale_slo_atta_list_all_workload))]
            compare_cv_scale_slo_atta_list = [(cv_scale_slo_atta_list_all_workload_cotdm[j] - cv_scale_slo_atta_list_all_workload[j]) / cv_scale_slo_atta_list_all_workload[j] for j in range(len(cv_scale_slo_atta_list_all_workload))]
            compare_slo_scale_slo_atta_list = [(slo_scale_slo_atta_list_all_workload_cotdm[j] - slo_scale_slo_atta_list_all_workload[j]) / slo_scale_slo_atta_list_all_workload[j] for j in range(len(slo_scale_slo_atta_list_all_workload))]

            for j in range(len(compare_rate_scale_slo_atta_list)):
                print(
                    "rate_scale={:.2f}, {:.2f}%".format(
                        rate_scale_list[j], compare_rate_scale_slo_atta_list[j] * 100
                    )
                )

            for j in range(len(compare_cv_scale_slo_atta_list)):
                print(
                    "cv_scale={:.2f}, {:.2f}%".format(
                        cv_scale_list[j], compare_cv_scale_slo_atta_list[j] * 100
                    )
                )

            for j in range(len(compare_slo_scale_slo_atta_list)):
                print(
                    "slo_scale={:.2f}, {:.2f}%".format(
                        slo_scale_list[j], compare_slo_scale_slo_atta_list[j] * 100
                    )
                )

            avg_compare_rate_scale = sum(compare_rate_scale_slo_atta_list) / len(compare_rate_scale_slo_atta_list)
            avg_compare_cv_scale = sum(compare_cv_scale_slo_atta_list) / len(compare_cv_scale_slo_atta_list)
            avg_compare_slo_scale = sum(compare_slo_scale_slo_atta_list) / len(compare_slo_scale_slo_atta_list)
            print(
                "Average: rate_scale {:.2f}%, cv_scale {:.2f}%, slo_scale {:.2f}%\n".format(
                    avg_compare_rate_scale * 100,
                    avg_compare_cv_scale * 100,
                    avg_compare_slo_scale * 100,
                )
            )


def main():
    deploy_scheme_dict = global_config.DEPLOY_SCHEME_DICT
    deploy_scheme_num = len(deploy_scheme_dict)
    modelset_num = 3
    e2e_lat_list_all_model_set = []
    slo_lists = []
    exp3_res_dir = global_config.EXP_RES_DIR + "exp3/"
    deploy_scheme_list = ["strawman", "alpaserve", "deepplan", "cotdm"]
    for i in range(modelset_num):
        e2e_lat_list_all_schemes = [[] for i in range(deploy_scheme_num)]

        for scheme in deploy_scheme_list:
            exp_file_path = get_latest_exp_res_file(exp3_res_dir, "exp3_modelset" + str(i + 1) + "_" + scheme)
            if not os.path.exists(exp_file_path):
                raise ValueError("{} doesn't exists!".format(exp_file_path))
            else:
                print("Running experiment3 data file path: {}".format(exp_file_path))

            with open(exp_file_path, "r") as f:
                json_data = json.load(f)

            deploy_scheme = json_data["deploy_scheme"]
            scheme_idx = deploy_scheme_dict[deploy_scheme]

            e2e_lat_list_all_workload = json_data["e2e_lat_list_all_workload"]
            e2e_lat_list_all_schemes[scheme_idx] = e2e_lat_list_all_workload

        e2e_lat_list_all_model_set.append(e2e_lat_list_all_schemes)
        
        rate_scale_list = json_data["rate_scale_list"]
        cv_scale_list = json_data["cv_scale_list"]
        slo_scale_list = exp3.slo_scale_list
        avg_e2e_lat_ready = json_data["avg_e2e_lat_ready"]
        slo_list = [round(lat * SLO_SCALE, 3) for lat in avg_e2e_lat_ready]
        slo_lists.append(slo_list)
    
    print_perf_improve_exp3(e2e_lat_list_all_model_set, slo_lists)
    plot_fig_exp3(e2e_lat_list_all_model_set, rate_scale_list, cv_scale_list, slo_scale_list, slo_lists)


if __name__ == "__main__":
    main()
