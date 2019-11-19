import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from gurobi_import import *
from utilities import *

# SCALE_PW = 2000
# SCALE_PW = 12000
SCALE_PW = 1
start_idx = 5
end_idx = 6
PW_ARR = [30000, 60000]
# PW_ARR = [30000, 60000]
# PW_ARR = [600, 5000, 10000]
# Maximum test = 20
TOTAL_TEST_FOR_EACH_SET = 5


def clustering_car_pw(n, k, dis_matrix, ml, cl, lower, upper):
    model = Model("Clustering with cardinality and pairwise constraints")
    model.setParam('OutputFlag', False)
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=-dis_matrix[i, j], vtype="B", name="x[%s,%s]" % (i, j))
    for i in range(n):
        coef = []
        var = []
        for j in range(k):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), "=", 1, name="1-cluster-for-point[%s]" % i)
    for i in range(len(ml)):
        for j in range(k):
            u, v = ml[i][0], ml[i][1]
            model.addConstr(x[u, j], "=", x[v, j], name="ML[%s]Cluster[%s]" % (i, j))
    for i in range(len(cl)):
        for j in range(k):
            u, v = cl[i][0], cl[i][1]
            model.addConstr(LinExpr([1, 1], [x[u, j], x[v, j]]), "<=", 1, name="CL[%s]Cluster[%s]" % (i, j))
    for j in range(k):
        coef = []
        var = []
        for i in range(n):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), ">=", lower, name="lower-car[%s]" % j)
        model.addConstr(LinExpr(coef, var), "<=", upper, name="upper-car[%s]" % j)

    model.update()
    model.__data = x
    return model


def read_input(folder_name, is_dcc=False):
    n, k, nML, nCL = np.loadtxt(os.path.join(folder_name, "main_var.txt"), dtype=int)
    if is_dcc:
        p = np.loadtxt(os.path.join(folder_name, "dcc-pw-bal/pdis.txt"))
    else:
        p = np.loadtxt(os.path.join(folder_name, "pdis.txt"))
    label = np.loadtxt(os.path.join(folder_name, "label"))
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"))
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"))
    return n, k, p, ml, cl, label


def metrics(label, partition):
    nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
    ari = adjusted_rand_score(label, partition)
    acc = calculate_acc(np.asarray(label), np.asarray(partition))
    return nmi, ari, acc


def run_model(folder_name, is_dcc):
    n, k, p, ml, cl, label = read_input(folder_name, is_dcc)
    counter = dict()
    for l in label:
        if l in counter:
            counter[l] += 1
        else:
            counter[l] = 1
    # cluster_size = int(n / k)
    label_car = get_cluster_size(label)
    min_car = min(label_car)
    max_car = max(label_car)
    # Assuming that we know the label already
    y_pred = np.argmax(p, axis=1)
    if is_dcc:
        print("Acc:", calculate_acc(label, y_pred))
        print("NMI:", normalized_mutual_info_score(label, y_pred))
    model = clustering_car_pw(n, k, p, ml, cl, min_car, max_car)
    # model.write('clustering-car-pw.lp')
    model.optimize()
    if model.SolCount == 0:
        print("No solution found, status %d" % model.Status)
        return metrics(label, y_pred), (NINF, NINF, NINF), get_num_violates(ml, cl, y_pred), 0.0, model.Runtime
    c = model.__data
    partition = extract_cluster_id(n, c, k)
    # print(partition)
    per_change = 0.0
    for i in range(n):
        if partition[i] != y_pred[i]:
            per_change += 1.0

    per_change = 100.0 * per_change / n
    return metrics(label, y_pred), metrics(label, partition), get_num_violates(ml, cl,
                                                                               y_pred), per_change, model.Runtime, partition


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
            test_folder = folder_prefix_path + "test" + str(testid).zfill(2)
            orignal_scores, scores, num_vio, per_change, time_running, pred = run_model(test_folder, is_dcc)

            nmi, ari, acc = orignal_scores
            stats[pairwise_factor]["onmi"].append(nmi)
            stats[pairwise_factor]["oacc"].append(acc)

            nmi, ari, acc = scores
            stats[pairwise_factor]["nmi"].append(nmi)
            stats[pairwise_factor]["acc"].append(acc)

            stats[pairwise_factor]["#violates"].append(num_vio)
            stats[pairwise_factor]["per_change"].append(per_change)
            stats[pairwise_factor]["time"].append(time_running)
            if is_dcc:
                if os.path.exists(test_folder + "/dcc-pw-bal-post"):
                    continue
                if not os.path.exists(test_folder + "/dcc-pw-bal-post"):
                    os.makedirs(test_folder + "/dcc-pw-bal-post")
                np.savetxt(test_folder + "/dcc-pw-bal-post/pred.txt", pred, fmt='%s')
                local_stat = np.asarray([time_running, acc, nmi])
                np.savetxt(test_folder + "/dcc-pw-bal-post/stat.txt", local_stat, fmt='%s')
                continue
            if os.path.exists(test_folder + "/post-car"):
                continue
            if not os.path.exists(test_folder + "/post-car"):
                os.makedirs(test_folder + "/post-car")
            np.savetxt(test_folder + "/post-car/pred.txt", pred, fmt='%s')
            local_stat = np.asarray([time_running, acc, nmi])
            np.savetxt(test_folder + "/post-car/stat.txt", local_stat, fmt='%s')
    return stats


keep_import()

parser = argparse.ArgumentParser(description='Pairwise constraints for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
args = parser.parse_args()

dataset_folder = "/home/henry/codes/deep_constrained_clustering/experiments/generating-constraints/" + args.data + "-pw/"

stats = run_experiment(dataset_folder, False)
stats_dcc = run_experiment(dataset_folder, True)
exported_table = []
# for i in range(start_idx, end_idx, 1):
for i in PW_ARR:
    exported_table.append([i * SCALE_PW,
                           get_mean_and_std(stats[i]["nmi"]),
                           get_mean_and_std(stats[i]["acc"]),
                           get_mean_and_std(stats[i]["#violates"]),
                           get_mean_and_std(stats[i]["per_change"]),
                           get_mean_and_std(stats_dcc[i]["onmi"]),
                           get_mean_and_std(stats_dcc[i]["oacc"]),
                           get_mean_and_std(stats_dcc[i]["nmi"]),
                           get_mean_and_std(stats_dcc[i]["acc"]),
                           get_mean_and_std(stats_dcc[i]["#violates"]),
                           get_mean_and_std(stats_dcc[i]["per_change"]),
                           get_mean_and_std(stats[i]["time"]),
                           get_mean_and_std(stats_dcc[i]["time"]),
                           ])

with open("report-pw-" + args.data + ".tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["\#Pairwise", "NMI-IDEC-Our", "ACC", "\#Unsatisfied", "\% Relabel",
                                   "NMI-DCC", "ACC", "NMI-DCC-Our", "ACC", "\#Unsatisfied", "\% Relabel", "Time1  (s)",
                                   "Time2 (s)"
                                   ],
                                  caption="Experimental results of pairwise constraints on " + args.data))


# for i in range(start_idx, end_idx, 1):
def get_values(stats, key1, key2, key3):
    ans = []
    for i in PW_ARR:
        ans.append(get_mean_and_std(stats[i][key1]))
        ans.append(get_mean_and_std(stats[i][key2]))
        print(key3)
        print(stats[i])
        if type(key3) == str:
            ans.append(get_mean_and_std(stats[i][key3]))
        else:
            ans.append(key3)
    return ans


exported_table = []
column_stats = ["IDEC"]
column_stats.extend(get_values(stats, "onmi", "oacc", "#violates"))
exported_table.append(column_stats)

column_stats = ["IDEC-Post"]
column_stats.extend(get_values(stats, "nmi", "acc", 0))
exported_table.append(column_stats)

column_stats = ["DCC"]
column_stats.extend(get_values(stats_dcc, "onmi", "oacc", "#violates"))
exported_table.append(column_stats)

column_stats = ["DCC-Post"]
column_stats.extend(get_values(stats_dcc, "nmi", "acc", 0))
exported_table.append(column_stats)

print(exported_table)
with open("report-vertical-pw-car-" + args.data + ".tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["", "NMI", "ACC", "\#Unsatisfied", "NMI", "ACC", "\#Unsatisfied", "NMI", "ACC",
                                   "\#Unsatisfied"],
                                  caption="Experimental results of pairwise-balance constraints on " + args.data))
