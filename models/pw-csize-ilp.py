import argparse
import os

from models.gurobi_import import *
from models.utilities import *

SCALE_PW = 1

PW_ARR = [3600]
TOTAL_TEST_FOR_EACH_SET = 1


def clustering_car_pw(n, k, dis_matrix, ml, cl, lower, upper):
    model = Model("Clustering with cardinality and pairwise constraints")
    model.setParam('OutputFlag', False)
    # Adding main parameter and objective function: X * Cluster-Allocation-Matrix
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=-dis_matrix[i, j], vtype="B", name="x[%s,%s]" % (i, j))
    # Constraint for non-empty cluster. But not need for large dataset or exist cluster size constraints.
    # for i in range(k):
    #     coef = []
    #     var = []
    #     for j in range(n):
    #         coef.append(1)
    #         var.append(x[j, i])
    #     model.addConstr(LinExpr(coef, var), ">=", 1, name="At least 1 instances for cluster[%s]" % i)
    # Constraint: each point only belongs to 1 cluster.
    for i in range(n):
        coef = []
        var = []
        for j in range(k):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), "=", 1, name="1-cluster-for-point[%s]" % i)
    # Pairwise constraints
    for i in range(len(ml)):
        for j in range(k):
            u, v = ml[i][0], ml[i][1]
            model.addConstr(x[u, j], "=", x[v, j], name="ML[%s]Cluster[%s]" % (i, j))
    for i in range(len(cl)):
        for j in range(k):
            u, v = cl[i][0], cl[i][1]
            model.addConstr(LinExpr([1, 1], [x[u, j], x[v, j]]), "<=", 1, name="CL[%s]Cluster[%s]" % (i, j))
    # Cluster size constraints
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
    return n, k, p, ml, cl, label


def metrics(label, partition):
    nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
    ari = adjusted_rand_score(label, partition)
    acc = calculate_acc(np.asarray(label), np.asarray(partition))
    return nmi, ari, acc


def run_model(folder_name, is_dcc):
    n, k, p, ml, cl, label = read_input(folder_name, is_dcc)

    # Finding bounding for cluster size constraints
    if args.csize == "True":
        label_car = get_cluster_size(label)
        min_car = min(label_car)
        max_car = max(label_car)
    else:
        min_car = 1
        max_car = n
    # Clustering of IDEC or DCC
    clustering = np.argmax(p, axis=1)
    model = clustering_car_pw(n, k, p, ml, cl, min_car, max_car)
    # model.write('clustering-car-pw.lp')
    model.optimize()
    if model.SolCount == 0:
        print("No solution found, status %d" % model.Status)
        return metrics(label, clustering), (NINF, NINF, NINF), get_num_violates(ml, cl, clustering), 0.0, model.Runtime
    c = model.__data
    # New clustering from ILP
    partition = extract_cluster_id(n, c, k)
    per_change = 0.0
    for i in range(n):
        if partition[i] != clustering[i]:
            per_change += 1.0

    per_change = 100.0 * per_change / n
    return metrics(label, clustering), metrics(label, partition), get_num_violates(ml, cl,
                                                                                   clustering), per_change, model.Runtime, partition


def run_experiment(dataset_folder, is_dcc):
    stats = dict()
    for pairwise_factor in PW_ARR:
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
                os.makedirs(test_folder + "/dcc-post", exist_ok=True)
                np.savetxt(test_folder + "/dcc-post/pred.txt", pred, fmt='%s')
                local_stat = np.asarray([time_running, acc, nmi])
                np.savetxt(test_folder + "/dcc-post/stat.txt", local_stat, fmt='%s')
                continue
            os.makedirs(test_folder + "/post", exist_ok=True)
            np.savetxt(test_folder + "/post/pred.txt", pred, fmt='%s')
            local_stat = np.asarray([time_running, acc, nmi])
            np.savetxt(test_folder + "/post/stat.txt", local_stat, fmt='%s')
    return stats


keep_import()

parser = argparse.ArgumentParser(description='Pairwise and Cluster-size constraints for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
parser.add_argument('--csize', type=str, default="True", metavar='N',
                    help='option to add cluster size constraints (True, False)')
args = parser.parse_args()

dataset_folder = "./sample_test/" + args.data + "/"
stats = run_experiment(dataset_folder, False)
stats_dcc = run_experiment(dataset_folder, True)

print(stats)
print(stats_dcc)
