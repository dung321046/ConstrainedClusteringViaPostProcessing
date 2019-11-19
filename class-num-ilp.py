import argparse
import json
import os
import time

from gurobi_import import *
from utilities import *

NUMC_ARR = [2000]

# NUMC_ARR = [139, 279]
import random


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
        choice = random.randint(0, 2)
        if choice == 0:
            model.addConstr(LinExpr(coef2, var2), "=", exclusion['class-number'], name="numclass-feature %s" % const_id)
        elif choice == 1:
            model.addConstr(LinExpr(coef2, var2), ">=", exclusion['class-number'],
                            name="numclass-feature %s" % const_id)
        else:
            model.addConstr(LinExpr(coef2, var2), "<=", exclusion['class-number'],
                            name="numclass-feature %s" % const_id)
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
parser = argparse.ArgumentParser(description='Exclusion for MNIST, Fashion and Reuters')
parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
args = parser.parse_args()

dataset_folder = "/home/henry/codes/deep_constrained_clustering/experiments/generating-constraints/" + args.data + "-numclass/"
nmi_arr = dict()
ari_arr = dict()
acc_arr = dict()
times = dict()
violated_con = dict()
find_permutation = False
permutation = []

for exclusion_num in NUMC_ARR:
    nmi_arr[exclusion_num] = []
    ari_arr[exclusion_num] = []
    acc_arr[exclusion_num] = []
    times[exclusion_num] = []
    violated_con[exclusion_num] = []
    folder_prefix_path = dataset_folder + args.data + "-" + str(exclusion_num) + "/"
    start_time = time.time()
    print(exclusion_num)
    for testid in range(1):
        print("Test id:", testid)
        n, k, p, label, exclusion_detail = read_input(
            folder_name=folder_prefix_path + "test" + str(testid).zfill(2))
        y_pred = np.argmax(p, axis=1)
        total_unsat_feature = 0
        for exclusion in exclusion_detail:
            unsat_clause = counting_class(y_pred, exclusion)
            if unsat_clause != exclusion['class-number']:
                total_unsat_feature += 1
        violated_con[exclusion_num].append(total_unsat_feature)
        model = clustering_exclusion_list(n, k, p, exclusion_detail)
        model.write('clustering-exclusion-list.lp')
        model.optimize()
        times[exclusion_num].append(model.Runtime)
        if model.SolCount == 0:
            print("No solution found, status %d" % model.Status)
            nmi_arr[exclusion_num].append(NINF)
            ari_arr[exclusion_num].append(NINF)
            acc_arr[exclusion_num].append(NINF)
        else:
            c = model.__data
            partition = extract_cluster_id(n, c, k)
            nmi, ari, acc = metrics(label, partition)
            nmi_arr[exclusion_num].append(nmi)
            ari_arr[exclusion_num].append(ari)
            acc_arr[exclusion_num].append(acc)

nmi, ari, acc = metrics(label, y_pred)
exported_table = [["0", nmi, acc, "-", "-"]]

for num_constraints in NUMC_ARR:
    exported_table.append([num_constraints, get_mean_and_std(nmi_arr[num_constraints]),
                           get_mean_and_std(acc_arr[num_constraints]),
                           get_mean_and_std(violated_con[num_constraints]),
                           get_mean_and_std(times[num_constraints])])

with open("report-classnum.tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["\#Constraints", "NMI", "ACC", "\#Unsat Constraints", "Runtime(s)"],
                                  caption="Experimental results of class numbers constraints on " + args.data,
                                  ref="num-class"))
