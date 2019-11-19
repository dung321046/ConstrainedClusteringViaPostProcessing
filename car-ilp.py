import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from gurobi_import import *
from utilities import *

SCALE_PW = 1
start_idx = 0
end_idx = 1
# PW_ARR = [500, 700, 1000, 1200, 1500, 1700, 1900]
PW_ARR = [60000]
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
        p = np.loadtxt(os.path.join(folder_name, "dcc/pdis.txt"))
    else:
        p = np.loadtxt(os.path.join(folder_name, "pdis.txt"))
    label = np.loadtxt(os.path.join(folder_name, "label"))
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"))
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"))
    car = np.loadtxt(os.path.join(folder_name, "cardinality.txt"))
    return n, k, p, ml, cl, label, car


def metrics(label, partition):
    nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
    ari = adjusted_rand_score(label, partition)
    acc = calculate_acc(np.asarray(label), np.asarray(partition))
    return nmi, ari, acc


def count_size(y):
    size_dict = dict()
    for lb in y:
        if lb not in size_dict:
            size_dict[lb] = 1
        else:
            size_dict[lb] += 1
    return size_dict


def get_car_stat(car, car_min, car_max):
    num_violates = 0
    largest = 0
    for c in car:
        if c < car_min:
            largest = max(largest, car_min - c)
            num_violates += car_min - c
        if c > car_max:
            num_violates += c - car_max
            largest = max(largest, c - car_max)
    return num_violates, largest


def run_model(folder_name, is_dcc):
    n, k, p, ml, cl, label, car = read_input(folder_name, is_dcc)
    # Assuming that we know the label already
    y_pred = np.argmax(p, axis=1)
    print("Init:", count_size(y_pred))
    if is_dcc:
        print("Acc:", calculate_acc(label, y_pred))
        print("NMI:", normalized_mutual_info_score(label, y_pred))
    ml_size = int(len(ml) / 2)
    cl_size = int(len(cl) / 2)
    if ml_size + cl_size < n:
        cl_size += 1
    # ml = ml[:ml_size]
    # cl = cl[:cl_size]
    ml = []
    cl = []
    min_car = min(car)
    max_car = max(car)
    print(min_car, max_car)
    bft_car = get_cluster_size(y_pred)
    num_car_violates, largest_diff = get_car_stat(bft_car, min_car, max_car)
    model = clustering_car_pw(n, k, p, ml, cl, min_car, max_car)
    # model.write('clustering-car-pw.lp')
    model.optimize()
    if model.SolCount == 0:
        print("No solution found, status %d" % model.Status)
        return metrics(label, y_pred), (NINF, NINF, NINF), get_num_violates(ml, cl,
                                                                            y_pred), 0.0, model.Runtime, num_car_violates, largest_diff
    c = model.__data
    partition = extract_cluster_id(n, c, k)
    print("ILP:", count_size(partition))
    # print(partition)
    per_change = get_num_change(y_pred, partition)
    pos_change = get_pos_change(label, y_pred, partition)
    return metrics(label, y_pred), metrics(label, partition), get_num_violates(ml, cl,
                                                                               y_pred), per_change, pos_change, model.Runtime, num_car_violates, largest_diff


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
            "pos_change": [],
            "is_better": [],
            "time": [],
            "#car_vios": [],
            "car_dff": [],
        }
        pairwise_num = pairwise_factor * SCALE_PW
        folder_prefix_path = dataset_folder + args.data + "-" + str(pairwise_num) + "/"
        for testid in range(TOTAL_TEST_FOR_EACH_SET):
            orignal_scores, scores, num_vio, per_change, pos_change, time_running, car_vios, car_dff = run_model(
                folder_prefix_path + "test" + str(testid).zfill(2), is_dcc)

            stats[pairwise_factor]["#car_vios"].append(car_vios)
            stats[pairwise_factor]["car_dff"].append(car_dff)

            nmi, ari, old_acc = orignal_scores
            stats[pairwise_factor]["onmi"].append(nmi)
            stats[pairwise_factor]["oacc"].append(old_acc)
            print("Init  :", orignal_scores)
            print("ILP   :", scores)
            nmi, ari, acc = scores
            stats[pairwise_factor]["nmi"].append(nmi)
            stats[pairwise_factor]["acc"].append(acc)

            if old_acc > acc:
                stats[pairwise_factor]["is_better"].append(0)
            else:
                stats[pairwise_factor]["is_better"].append(1)
            stats[pairwise_factor]["#violates"].append(num_vio)
            stats[pairwise_factor]["per_change"].append(per_change)
            stats[pairwise_factor]["pos_change"].append(pos_change)
            stats[pairwise_factor]["time"].append(time_running)
    return stats


keep_import()

parser = argparse.ArgumentParser(description='Pairwise constraints for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
args = parser.parse_args()

dataset_folder = "/home/henry/codes/ILP/" + args.data + "-car-pw/"

stats = run_experiment(dataset_folder, False)
# pw_stats = run_experiment(dataset_folder, True)
exported_table = []
# for i in range(start_idx, end_idx, 1):
for i in PW_ARR:
    print("better", stats[i]["is_better"])
    exported_table.append([i * SCALE_PW,
                           # get_mean_and_std(stats[i]["onmi"]),
                           # get_mean_and_std(stats[i]["oacc"]),
                           get_mean_and_std(stats[i]["nmi"]),
                           get_mean_and_std(stats[i]["acc"]),
                           # get_mean_and_std(stats[i]["#violates"]),
                           get_mean_and_std(stats[i]["per_change"]),
                           get_mean_and_std(stats[i]["pos_change"]),
                           # get_mean_and_std(stats[i]["is_better"]),
                           # get_mean_and_std(stats[i]["#car_vios"]),
                           # get_mean_and_std(stats[i]["car_dff"]),
                           get_mean_and_std(stats[i]["time"]),
                           ])

with open("report-car-pw-" + args.data + ".tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["\#Size", "NMI", "ACC", "Number of changes", "Positive changes", "time"
                                      #, "\#Violates", "Max cluster violates", "time"
                                   ],
                                  caption="Experimental results of cluster size constraints on " + args.data))
