import argparse
import os

import regex as re

from gurobi_import import *
from utilities import *

SCALE_PW = 1
# PW_ARR = [3600, 30000, 60000]
# PW_ARR = [600, 5000, 10000]
PW_ARR = [30000]
TOTAL_TEST_FOR_EACH_SET = 5

IS_LOCAL = False


def read_input(folder_name, p_sub_floder=""):
    p = np.loadtxt(os.path.join(folder_name + p_sub_floder, "pdis.txt"))
    label = np.loadtxt(os.path.join(folder_name, "label"), dtype=int)
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"))
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"))
    return ml, cl, label, p


def metrics(label, partition):
    nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
    ari = adjusted_rand_score(label, partition)
    acc = calculate_acc(np.asarray(label), np.asarray(partition))
    return nmi, ari, acc


def get_stats_from_data(dataset_folder, aft_pred_folder):
    stats = dict()
    # for pairwise_factor in range(start_idx, end_idx, 1):
    for pairwise_factor in PW_ARR:
        print("PW", pairwise_factor)
        stats[pairwise_factor] = {
            "onmi": [],
            "oacc": [],
            "o#violates": [],
            "nmi": [],
            "acc": [],
            "#violates": [],
            "per_change": [],
            "pos_change": [],
            "time": [],
        }
        pairwise_num = pairwise_factor * SCALE_PW
        folder_prefix_path = dataset_folder + str(pairwise_num) + "/"
        for testid in range(TOTAL_TEST_FOR_EACH_SET):
            test_folder = folder_prefix_path + "test" + str(testid).zfill(2)
            if aft_pred_folder == "/post-dcc":
                bft_pred_folder = "/dcc"
            elif aft_pred_folder == "/dcc-pw-bal-post":
                bft_pred_folder = "/dcc-pw-bal"
            else:
                bft_pred_folder = ""
            ml, cl, label, pdis = read_input(test_folder, bft_pred_folder)
            bfr_pred = np.argmax(pdis, axis=1)
            if os.path.exists(test_folder + aft_pred_folder + "/pred.txt"):
                aft_pred = np.loadtxt(test_folder + aft_pred_folder + "/pred.txt")
            elif os.path.exists(test_folder + aft_pred_folder + "/label.txt"):
                aft_pred = np.loadtxt(test_folder + aft_pred_folder + "/label.txt")
            else:
                aft_p = np.loadtxt(test_folder + aft_pred_folder + "/pdis.txt")
                aft_pred = np.argmax(aft_p, axis=1)
            # if re.search("dcc$", aft_pred_folder):
            aft_pred = convert_label(label, aft_pred)
            bfr_pred = convert_label(label, bfr_pred)

            nmi, ari, acc = metrics(label, bfr_pred)

            if IS_LOCAL:
                conflict_points = get_point_violates(ml, cl, bfr_pred)
                sub_label = []
                sub_bfr_pred = []
                for p in conflict_points:
                    sub_label.append(label[p])
                    sub_bfr_pred.append(bfr_pred[p])
                nmi, ari, acc = metrics(sub_label, sub_bfr_pred)
            stats[pairwise_factor]["onmi"].append(nmi)
            stats[pairwise_factor]["oacc"].append(acc)
            stats[pairwise_factor]["o#violates"].append(get_num_violates(ml, cl, bfr_pred))

            nmi, ari, acc = metrics(label, aft_pred)
            if aft_pred_folder == "/dcc":
                print("NMI - ACC", nmi, acc)
                print(label[:20])
                print(bfr_pred[:20])
                print(aft_pred[:20])
                print(get_cluster_size(aft_pred))

            if IS_LOCAL:
                conflict_points = get_point_violates(ml, cl, bfr_pred)
                sub_label = []
                sub_afr_pred = []
                for p in conflict_points:
                    sub_label.append(label[p])
                    sub_afr_pred.append(aft_pred[p])
                nmi, ari, acc = metrics(sub_label, sub_afr_pred)
            stats[pairwise_factor]["nmi"].append(nmi)
            stats[pairwise_factor]["acc"].append(acc)
            stats[pairwise_factor]["#violates"].append(get_num_violates(ml, cl, aft_pred))
            if re.search("^\/cop", aft_pred_folder):
                stats[pairwise_factor]["per_change"].append("-")
                # stats[pairwise_factor]["pos_change"].append("-")
                # continue
            n = len(bfr_pred)
            per_change = get_num_change(bfr_pred, aft_pred)
            pos_change = get_pos_change(label, bfr_pred, aft_pred)
            stats[pairwise_factor]["per_change"].append(per_change)
            stats[pairwise_factor]["pos_change"].append(pos_change)
    return stats


keep_import()

parser = argparse.ArgumentParser(description='Pairwise constraints for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
args = parser.parse_args()

dataset_folder = "/home/henry/codes/deep_constrained_clustering/experiments/generating-constraints/" + args.data + "-pw/"

methods = [
    {"folder": "/post",
     "name": "PW"},
    {"folder": "/post-car",
     "name": "PW + CS"},
    {"folder": "/all",
     "name": "PW + CS + NC"},
]
caption = "Experimental results of combining constraints on "

for method in methods:
    method["stats"] = get_stats_from_data(dataset_folder, method["folder"])
stats_cop = get_stats_from_data(dataset_folder, "/cop")


# for i in range(start_idx, end_idx, 1):
def get_values(stats, key1, key2, key3, key4):
    ans = []
    for i in PW_ARR:
        ans.append(get_mean_and_std(stats[i][key1]))
        ans.append(get_mean_and_std(stats[i][key2]))
        if type(key3) == str:
            ans.append(get_mean_and_std(stats[i][key3]))
        else:
            ans.append(key3)
        if type(key4) == str:
            ans.append(get_mean_and_std(stats[i][key4]))
        else:
            ans.append(key4)
    return ans


first_row = ["No constraint", get_mean_and_std(stats_cop[PW_ARR[0]]["onmi"])[0],
             get_mean_and_std(stats_cop[PW_ARR[0]]["oacc"])[0], "-", "-"]
exported_table = [first_row]
for i in PW_ARR:
    for j, method in enumerate(methods):
        if j == 0:
            s = "\hline \multirow{" + str(len(methods)) + "}{*}{" + str(i) + "}"
        else:
            s = ""
        row = [method["name"],
               get_mean_and_std(method["stats"][i]["nmi"]),
               get_mean_and_std(method["stats"][i]["acc"]),
               get_mean_and_std(method["stats"][i]["per_change"]),
               get_mean_and_std(method["stats"][i]["pos_change"])
               ]
        exported_table.append(row)

print(exported_table)
with open("report-combine-" + args.data + ".tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["Method", "NMI", "ACC", "Changes (\%)", "Positive changes (\%)"],
                                  caption=caption + args.data,
                                  ref="combine-" + args.data.lower()))
