import argparse
import json
import os
import time

from gurobi_import import *
from utilities import *

NUMC_ARR = [3, 10000]


# NUMC_ARR = [139, 279]


def clustering_exclusion_list(n, k, dis_matrix, exclusion_detail):
    model = Model("Clustering with exclusion constraints")
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
    total_point = {}
    for const_id, exclusion in enumerate(exclusion_detail):
        coef2 = []
        var2 = []
        for i in range(k):
            total_point[const_id, i] = model.addVar(vtype=GRB.BINARY, name="x[%s,%s]" % (const_id, i))
            for row in exclusion['rows']:
                model.addConstr(LinExpr([1, -1], [total_point[const_id, i], x[row, i]]), ">=", 0,
                                name="lower-bound[%s] %s %s" % (const_id, i, row))
            coef = [-1]
            var = [total_point[const_id, i]]
            for row in exclusion['rows']:
                coef.append(1)
                var.append(x[row, i])
            model.addConstr(LinExpr(coef, var), ">=", 0, name="upper-bound[%s] %s" % (const_id, i))
            # Add to sum equation
            coef2.append(1)
            var2.append(total_point[const_id, i])
        model.addConstr(LinExpr(coef2, var2), "<=", 2, name="numclass-feature %s" % const_id)
    model.update()
    model.__data = x
    return model


def read_input(folder_name):
    n, k, nel = np.loadtxt(os.path.join(folder_name, "main_var.txt"), dtype=int)
    p = np.loadtxt(os.path.join(folder_name, "pdis.txt"))
    label = np.loadtxt(os.path.join(folder_name, "label"), dtype=int)
    with open(os.path.join(folder_name, "numclass-feature.json"), 'r') as f:
        exclusion_detail = json.load(f)
    return n, k, p, label, exclusion_detail


def counting_class(y, exclusion):
    b = set()
    for row in exclusion['rows']:
        b.add(y[row])
    return len(b)


keep_import()
parser = argparse.ArgumentParser(description='Num-class constraints for Reuters')
args = parser.parse_args()
args.data = "Reuters"
dataset_folder = "/home/henry/codes/deep_constrained_clustering/experiments/generating-constraints/" + args.data + "-numclass-set/"
nmi_arr = dict()
ari_arr = dict()
acc_arr = dict()
times = dict()
perchange_arr = dict()
violated_con = dict()
size_var = dict()
poschange_arr = dict()
find_permutation = False
permutation = []
ARR = [5, 10]
for exclusion_num in ARR:
    nmi_arr[exclusion_num] = []
    ari_arr[exclusion_num] = []
    acc_arr[exclusion_num] = []
    times[exclusion_num] = []
    violated_con[exclusion_num] = []
    perchange_arr[exclusion_num] = []
    poschange_arr[exclusion_num] = []
    size_var[exclusion_num] = []
    # folder_prefix_path = dataset_folder + args.data + "-" + str(exclusion_num) + "/"
    folder_prefix_path = dataset_folder + args.data + "-" + str(3) + "/"
    start_time = time.time()
    print(exclusion_num)
    for testid in range(5):
        print("Test id:", testid)
        n, k, p, label, exclusion_detail = read_input(
            folder_name=folder_prefix_path + "test" + str(testid).zfill(2))
        # exclusion_detail = np.take(exclusion_detail, np.random.permutation(len(exclusion_detail))[:exclusion_num])
        exclusion_detail = exclusion_detail[:exclusion_num]
        y_pred = np.argmax(p, axis=1)
        total_unsat_feature = 0
        total_clauses = 0
        for exclusion in exclusion_detail:
            # print(exclusion["feature"])
            total_clauses += len(exclusion["rows"])
            unsat_clause = counting_class(y_pred, exclusion)
            if unsat_clause > exclusion['class-number']:
                total_unsat_feature += 1
                # print(total_unsat_feature)
            #size_var[exclusion_num].append(len(exclusion["rows"]))
        size_var[exclusion_num].append(total_clauses)
        violated_con[exclusion_num].append(total_unsat_feature)
        model = clustering_exclusion_list(n, k, p, exclusion_detail)
        model.optimize()
        times[exclusion_num].append(model.Runtime)
        if model.SolCount == 0:
            print("No solution found, status %d" % model.Status)
            nmi_arr[exclusion_num].append(NINF)
            ari_arr[exclusion_num].append(NINF)
            acc_arr[exclusion_num].append(NINF)
            perchange_arr[exclusion_num].append(0)
        else:
            c = model.__data
            partition = extract_cluster_id(n, c, k)
            nmi, ari, acc = metrics(label, partition)
            print(nmi, acc)
            nmi_arr[exclusion_num].append(nmi)
            ari_arr[exclusion_num].append(ari)
            acc_arr[exclusion_num].append(acc)
            perchange_arr[exclusion_num].append(get_num_change(y_pred, partition))
nmi, ari, acc = metrics(label, y_pred)
exported_table = [["0", nmi, acc, "-", "-", "-", "-"]]

for num_constraints in ARR:
    exported_table.append([num_constraints, get_mean_and_std(nmi_arr[num_constraints]),
                           get_mean_and_std(acc_arr[num_constraints]),
                           get_mean_and_std(size_var[num_constraints]),
                           get_mean_and_std(violated_con[num_constraints]),
                           get_mean_and_std(perchange_arr[num_constraints]),
                           get_mean_and_std(times[num_constraints])
                           ])

with open("report-classnum.tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["\#Constraints", "NMI", "ACC", "\#Clauses", "\#Unsat Constraints", "Change (\%)",
                                   "Times"],
                                  caption="Experimental results of class numbers constraints on " + args.data,
                                  ref="num-class"))
