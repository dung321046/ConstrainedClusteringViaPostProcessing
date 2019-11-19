import argparse
import os

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from gurobi_import import *
from utilities import *

# SCALE_PW = 2000
SCALE_PW = 12000

start_idx = 5
end_idx = 6

# Maximum test = 20
TOTAL_TEST_FOR_EACH_SET = 3


def clustering_car_pw(n, k, dis_matrix, ml, cl):
    model = Model("Clustering with cardinality and pairwise constraints")
    model.setParam('OutputFlag', False)
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=-dis_matrix[i][j], vtype="B", name="x[%s,%s]" % (i, j))
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


def convert_edges(vertex, new_id_dict, edge):
    edge[0] = int(edge[0])
    edge[1] = int(edge[1])
    if edge[0] not in vertex:
        new_id_dict[edge[0]] = len(vertex)
        vertex.add(edge[0])
    if edge[1] not in vertex:
        new_id_dict[edge[1]] = len(vertex)
        vertex.add(edge[1])

    return (new_id_dict[edge[0]], new_id_dict[edge[1]])


def read_input(folder_name):
    n, k, nML, nCL = np.loadtxt(os.path.join(folder_name, "main_var.txt"), dtype=int)
    p = np.loadtxt(os.path.join(folder_name, "pdis.txt"))
    label = np.loadtxt(os.path.join(folder_name, "label"))
    label = list(label)
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"))
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"))
    vertex = set()
    new_id_dict = dict()
    new_ml = []
    for i in range(len(ml)):
        new_ml.append(convert_edges(vertex, new_id_dict, ml[i]))
    new_cl = []
    for i in range(len(cl)):
        new_cl.append(convert_edges(vertex, new_id_dict, cl[i]))
    new_p = []
    fixed_label = []
    fixed_pred = []
    m = len(vertex)
    new_label = []
    for i in vertex:
        new_label.append(label[int(i)])
    for id in range(n):
        if id in vertex:
            new_p.append(p[id])
        else:
            fixed_label.append(label[id])
            fixed_pred.append(np.argmax(p[id]))
    return n, k, new_p, new_ml, new_cl, new_label, m, fixed_label, fixed_pred


keep_import()

parser = argparse.ArgumentParser(description='Pairwise constraints for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
args = parser.parse_args()

dataset_folder = "/home/henry/codes/ILP/" + args.data + "-pw/"

nmi_arr = dict()
ari_arr = dict()
violated_arr = dict()
acc_arr = dict()
car_error_arr = dict()
percentage_change = dict()
times = dict()
for pairwise_factor in range(start_idx, end_idx, 1):
    nmi_arr[pairwise_factor] = []
    ari_arr[pairwise_factor] = []
    acc_arr[pairwise_factor] = []
    times[pairwise_factor] = []
    percentage_change[pairwise_factor] = []
    violated_arr[pairwise_factor] = []

    pairwise_num = pairwise_factor * SCALE_PW
    folder_prefix_path = dataset_folder + args.data + "-" + str(pairwise_num) + "/"
    for testid in range(TOTAL_TEST_FOR_EACH_SET):
        n, k, p, ml, cl, label, m, fixed_label, fixed_pred = read_input(
            folder_name=folder_prefix_path + "test" + str(testid).zfill(2))
        # Assuming that we know the label already
        y_pred = np.argmax(p, axis=1)
        violated_arr[pairwise_factor].append(get_num_violates(ml, cl, y_pred))

        model = clustering_car_pw(m, k, p, ml, cl)
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
            partition = extract_cluster_id(m, c, k)
            # print(partition)
            per_change = 0.0
            for i in range(m):
                if partition[i] != y_pred[i]:
                    per_change += 1.0

            per_change = per_change / n
            label.extend(fixed_label)
            partition.extend(fixed_pred)
            partition = np.asarray(partition)
            label = np.asarray(label)
            nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
            ari = adjusted_rand_score(label, partition)
            nmi_arr[pairwise_factor].append(nmi)
            ari_arr[pairwise_factor].append(ari)
            percentage_change[pairwise_factor].append(per_change)
            acc_arr[pairwise_factor].append(calculate_acc(label, partition))

exported_table = []
for i in range(start_idx, end_idx, 1):
    exported_table.append([i * SCALE_PW,
                           get_mean_and_std(nmi_arr[i]),
                           get_mean_and_std(acc_arr[i]),
                           get_mean_and_std(violated_arr[i]),
                           get_mean_and_std(percentage_change[i]),
                           get_mean_and_std(times[i])])

with open("report-pw-" + args.data + ".tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["\#Pairwise", "NMI", "ACC", "\#Unsatisfied", "\% Relabel", "Time Running(s)"],
                                  caption="Experimental results of pairwise constraints on " + args.data))
