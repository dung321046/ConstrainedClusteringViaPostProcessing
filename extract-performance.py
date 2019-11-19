import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from gurobi_import import *
from utilities import *

SCALE_PW = 1
#PW_ARR = [3600, 30000, 60000]
PW_ARR = [600, 5000, 10000]
TOTAL_TEST_FOR_EACH_SET = 5


def read_copkmeans(folder_name):
    stats = np.loadtxt(os.path.join(folder_name, "cop/stat.txt"))
    predict = np.loadtxt(os.path.join(folder_name, "cop/label.txt"))
    return stats[0], stats[1], stats[2], predict


def read_input(folder_name, is_dcc=False):
    n, k, nML, nCL = np.loadtxt(os.path.join(folder_name, "main_var.txt"), dtype=int)
    if is_dcc:
        p = np.loadtxt(os.path.join(folder_name, "dcc/pdis.txt"))
    else:
        p = np.loadtxt(os.path.join(folder_name, "pdis.txt"))
    label = np.loadtxt(os.path.join(folder_name, "label"), dtype=int)
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"))
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"))
    return n, k, p, ml, cl, label


def metrics(label, partition):
    nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
    ari = adjusted_rand_score(label, partition)
    acc = calculate_acc(np.asarray(label), np.asarray(partition))
    return nmi, ari, acc


def metrics(label, partition):
    nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
    ari = adjusted_rand_score(label, partition)
    acc = calculate_acc(np.asarray(label), np.asarray(partition))
    return nmi, ari, acc


def run_experiment(dataset_folder, is_dcc):
    stats = dict()
    # for pairwise_factor in range(start_idx, end_idx, 1):
    for pairwise_factor in PW_ARR:
        print("PW", pairwise_factor)
        stats[pairwise_factor] = {
            "onmi": [],
            "oacc": [],
            "nmi": [],
            "acc": [],
            "#violates": [],
            "per_change": [],
            "time": [],
        }
        pairwise_num = pairwise_factor * SCALE_PW
        folder_prefix_path = dataset_folder + str(pairwise_num) + "/"
        for testid in range(TOTAL_TEST_FOR_EACH_SET):
            n, k, p, ml, cl, label = read_input(folder_prefix_path + "test" + str(testid).zfill(2), is_dcc)
            time_running, nmi, acc, y_pred = read_copkmeans(folder_prefix_path + "test" + str(testid).zfill(2))
            if len(y_pred) == 0:
                y_pred = np.argmax(p, axis=1)
            nmi, ari, acc = metrics(label, y_pred)
            stats[pairwise_factor]["nmi"].append(nmi)
            stats[pairwise_factor]["acc"].append(acc)
            stats[pairwise_factor]["time"].append(time_running)
            stats[pairwise_factor]["#violates"].append(get_num_violates(ml, cl, y_pred))
    return stats


def run_stat(dataset_folder):
    stats = dict()
    # for pairwise_factor in range(start_idx, end_idx, 1):
    for pairwise_factor in PW_ARR:
        print("PW", pairwise_factor)
        stats[pairwise_factor] = {
            "onmi": [],
            "oacc": [],
            "nmi": [],
            "acc": [],
            "#violates": [],
            "per_change": [],
            "time": [],
        }
        pairwise_num = pairwise_factor * SCALE_PW
        folder_prefix_path = dataset_folder + str(pairwise_num) + "/"
        for testid in range(TOTAL_TEST_FOR_EACH_SET):
            s = np.loadtxt(os.path.join(folder_prefix_path + "test" + str(testid).zfill(2), "dcc/stat.txt"))
            if len(s) != 2:
                print("Error stats DCC", pairwise_factor, TOTAL_TEST_FOR_EACH_SET)
                print(s)
            stats[pairwise_factor]["time"].append(s[0])
    return stats


keep_import()

parser = argparse.ArgumentParser(description='Pairwise constraints for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
args = parser.parse_args()

dataset_folder = "/home/henry/codes/deep_constrained_clustering/experiments/generating-constraints/" + args.data + "-pw/"

stats_cop = run_experiment(dataset_folder, False)

stats_dcc = run_stat(dataset_folder)


# for i in range(start_idx, end_idx, 1):
def get_values(stats, key1, key2, key3):
    ans = []
    for i in PW_ARR:
        ans.append(get_mean_and_std(stats[i][key1])[0])
        ans.append(get_mean_and_std(stats[i][key2])[0])
        if type(key3) == str:
            ans.append(get_mean_and_std(stats[i][key3]))
        else:
            ans.append(key3)
    return ans


exported_table = []
column_stats = ["COP Kmeans"]
column_stats.extend(get_values(stats_cop, "nmi", "acc", "#violates"))
exported_table.append(column_stats)

print(exported_table)
with open("report-vertical-pw-copkmeans-" + args.data + ".tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["", "NMI", "ACC", "\#Unsatisfied", "NMI", "ACC", "\#Unsatisfied", "NMI", "ACC",
                                   "\#Unsatisfied"],
                                  caption="Experimental results of pairwise constraints on " + args.data + " using COP-Kmeans"))
exported_table = []
# for i in range(start_idx, end_idx, 1):
for i in PW_ARR:
    exported_table.append([args.data, i * SCALE_PW,
                           get_mean_and_std(stats_cop[i]["time"]),
                           get_mean_and_std(stats_dcc[i]["time"])])

with open("report-pw-perform-" + args.data + ".tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["\#Pairwise", "COP Kmeans", "DCC"],
                                  caption="Performance comparision on " + args.data))
