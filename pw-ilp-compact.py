import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from gurobi_import import *
from utilities import *

# SCALE_PW = 2000
SCALE_PW = 12000

start_idx = 1
end_idx = 2

# Maximum test = 20
TOTAL_TEST_FOR_EACH_SET = 10


def clustering_car_pw(n, k, dis_matrix, ml, cl):
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

    model.update()
    model.__data = x
    return model


def read_input(folder_name):
    n, k, nML, nCL = np.loadtxt(os.path.join(folder_name, "main_var.txt"), dtype=int)
    p = np.loadtxt(os.path.join(folder_name, "pdis.txt"))
    label = np.loadtxt(os.path.join(folder_name, "label"))
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"))
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"))
    return n, k, p, ml, cl, label


keep_import()

parser = argparse.ArgumentParser(description='Pairwise constraints for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
args = parser.parse_args()
for COMPACT in [0, 0.7]:
    S_COMPACT = str(int(COMPACT * 100))
    dataset_folder = "/home/henry/codes/ILP/" + args.data + "-pw-" + S_COMPACT + "/"

    nmi_arr = dict()
    ari_arr = dict()
    violated_arr = dict()
    acc_arr = dict()
    car_error_arr = dict()
    percentage_change = dict()
    times = dict()
    find_permutation = False
    permutation = [2, 9, 6, 5, 1, 3, 0, 7, 8, 4]
    for pairwise_factor in range(start_idx, end_idx, 1):
        nmi_arr[pairwise_factor] = []
        ari_arr[pairwise_factor] = []
        acc_arr[pairwise_factor] = []
        times[pairwise_factor] = []
        percentage_change[pairwise_factor] = []
        violated_arr[pairwise_factor] = []

        pairwise_num = pairwise_factor * SCALE_PW
        folder_prefix_path = dataset_folder + str(pairwise_num) + "/"
        for testid in range(TOTAL_TEST_FOR_EACH_SET):
            n, k, p, ml, cl, label = read_input(folder_name=folder_prefix_path + "test" + str(testid).zfill(2))
            # Assuming that we know the label already
            y_pred = np.argmax(p, axis=1)
            # if not find_permutation:
            #     permutation, _ = find_best_permutation(label, y_pred, k)
            #     find_permutation = True
            #     print("Best permutaion:", permutation)
            violated_arr[pairwise_factor].append(get_num_violates(ml, cl, y_pred))

            model = clustering_car_pw(n, k, p, ml, cl)
            # model.write('clustering-car-pw.lp')
            model.optimize()
            times[pairwise_factor].append(model.Runtime)
            if model.SolCount == 0:
                print("No solution found, status %d" % model.Status)
                nmi_arr[pairwise_factor].append(NINF)
                ari_arr[pairwise_factor].append(NINF)
                acc_arr[pairwise_factor].append(NINF)
            else:
                c = model.__data
                partition = extract_cluster_id(n, c, k)
                # print(partition)
                per_change = 0.0
                for i in range(n):
                    if partition[i] != y_pred[i]:
                        per_change += 1.0

                per_change = per_change / n
                nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
                ari = adjusted_rand_score(label, partition)
                nmi_arr[pairwise_factor].append(nmi)
                ari_arr[pairwise_factor].append(ari)
                percentage_change[pairwise_factor].append(per_change)
                acc_arr[pairwise_factor].append(get_acc_with_permu(label, partition, permutation))

    exported_table = []
    for i in range(start_idx, end_idx, 1):
        exported_table.append([i * SCALE_PW,
                               get_mean_and_std(nmi_arr[i]),
                               get_mean_and_std(acc_arr[i]),
                               get_mean_and_std(violated_arr[i]),
                               get_mean_and_std(percentage_change[i]),
                               get_mean_and_std(times[i])])

    with open("report-pw-" + S_COMPACT + ".tex", "w") as file:
        file.write(arr_to_table_latex(exported_table,
                                      ["\#Pairwise", "NMI", "ACC", "\#Unsatisfied", "\% Relabel", "Time Running(s)"],
                                      caption="Experimental results of pairwise constraints on " + args.data + " with compact = " + S_COMPACT))
