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


def clustering_car_pw(n, k, dis_matrix, ml, cl, car, car_error):
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
        lower = int(car * (1 - car_error))
        upper = int(car * (1 + car_error))
        model.addConstr(LinExpr(coef, var), ">=", lower, name="lower-car[%s]" % j)
        model.addConstr(LinExpr(coef, var), "<=", upper, name="upper-car[%s]" % j)

    model.update()
    model.__data = x
    return model


def read_input(folder_name):
    n, k, nML, nCL = np.loadtxt(os.path.join(folder_name, "main_var.txt"), dtype=int)
    p = np.loadtxt(folder_name + "/post-car/post-car-idec/pdis.txt")
    label = np.loadtxt(os.path.join(folder_name, "label"))
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"))
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"))
    return n, k, p, ml, cl, label


def metrics(label, partition):
    nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
    ari = adjusted_rand_score(label, partition)
    acc = calculate_acc(np.asarray(label), np.asarray(partition))
    return nmi, ari, acc


def run_model(folder_name):
    n, k, p, ml, cl, label = read_input(folder_name)
    counter = dict()
    for l in label:
        if l in counter:
            counter[l] += 1
        else:
            counter[l] = 1
    cluster_size = int(n / k)
    # Assuming that we know the label already
    y_pred = np.argmax(p, axis=1)
    model = clustering_car_pw(n, k, p, ml, cl, cluster_size, 0.0)
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


def run_experiment(dataset_folder):
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
            orignal_scores, scores, num_vio, per_change, time_running, pred = run_model(test_folder)

            nmi, ari, acc = orignal_scores
            stats[pairwise_factor]["onmi"].append(nmi)
            stats[pairwise_factor]["oacc"].append(acc)

            nmi, ari, acc = scores
            stats[pairwise_factor]["nmi"].append(nmi)
            stats[pairwise_factor]["acc"].append(acc)

            stats[pairwise_factor]["#violates"].append(num_vio)
            stats[pairwise_factor]["per_change"].append(per_change)
            stats[pairwise_factor]["time"].append(time_running)
            if os.path.exists(test_folder + "/cidec-car-post"):
                continue
            os.makedirs(test_folder + "/cidec-car-post")
            np.savetxt(test_folder + "/cidec-car-post/pred.txt", pred, fmt='%s')
            local_stat = np.asarray([time_running, acc, nmi])
            np.savetxt(test_folder + "/cidec-car-post/stat.txt", local_stat, fmt='%s')
    return stats


keep_import()

parser = argparse.ArgumentParser(description='Pairwise constraints for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
args = parser.parse_args()

dataset_folder = "/home/henry/codes/deep_constrained_clustering/experiments/generating-constraints/" + args.data + "-pw/"

stats = run_experiment(dataset_folder)
