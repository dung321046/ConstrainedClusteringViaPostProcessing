import json
import os

from models.gurobi_import import *
from models.utilities import *


def clustering_exclusion_list(n, k, dis_matrix, exclusion_detail):
    model = Model("Clustering with exclusion constraints")
    model.setParam('OutputFlag', False)
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=-dis_matrix[i, j], vtype="B", name="x[%s,%s]" % (i, j))
    # Constraint for non-empty cluster but no need for large dataset or exist cluster size constraints.
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

nmi_arr = dict()
ari_arr = dict()
acc_arr = dict()
times = dict()
perchange_arr = dict()
violated_con = dict()
size_var = dict()
poschange_arr = dict()

ARR = [5, 10]
for num_constraints in ARR:
    nmi_arr[num_constraints] = []
    ari_arr[num_constraints] = []
    acc_arr[num_constraints] = []
    times[num_constraints] = []
    violated_con[num_constraints] = []
    perchange_arr[num_constraints] = []
    poschange_arr[num_constraints] = []
    size_var[num_constraints] = []
    folder_prefix_path = "./sample_test/Reuters-set3/"

    for testid in range(5):
        n, k, p, label, exclusion_detail = read_input(
            folder_name=folder_prefix_path + "test" + str(testid).zfill(2))
        exclusion_detail = exclusion_detail[:num_constraints]
        init_clustering = np.argmax(p, axis=1)
        total_unsat_feature = 0
        total_clauses = 0
        for exclusion in exclusion_detail:
            total_clauses += len(exclusion["rows"])
            num_of_cluster = counting_class(init_clustering, exclusion)
            if num_of_cluster > exclusion['class-number']:
                total_unsat_feature += 1

        size_var[num_constraints].append(total_clauses)
        violated_con[num_constraints].append(total_unsat_feature)
        model = clustering_exclusion_list(n, k, p, exclusion_detail)
        model.optimize()
        times[num_constraints].append(model.Runtime)
        if model.SolCount == 0:
            print("No solution found, status %d" % model.Status)
            nmi_arr[num_constraints].append(NINF)
            ari_arr[num_constraints].append(NINF)
            acc_arr[num_constraints].append(NINF)
            perchange_arr[num_constraints].append(0)
        else:
            c = model.__data
            clustering = extract_cluster_id(n, c, k)
            nmi, ari, acc = metrics(label, clustering)
            nmi_arr[num_constraints].append(nmi)
            ari_arr[num_constraints].append(ari)
            acc_arr[num_constraints].append(acc)
            perchange_arr[num_constraints].append(get_num_change(init_clustering, clustering))

nmi, ari, acc = metrics(label, init_clustering)
exported_table = [["0", nmi, acc, "-", "-", "-", "-"]]

for num_constraints in ARR:
    exported_table.append([num_constraints, get_mean_and_std(nmi_arr[num_constraints]),
                           get_mean_and_std(acc_arr[num_constraints]),
                           get_mean_and_std(size_var[num_constraints]),
                           get_mean_and_std(violated_con[num_constraints]),
                           get_mean_and_std(perchange_arr[num_constraints]),
                           get_mean_and_std(times[num_constraints])
                           ])

with open("report-attribute-level.tex", "w") as file:
    file.write(arr_to_table_latex(exported_table,
                                  ["\#Constraints", "NMI", "ACC", "\#Clauses", "\#Unsat Constraints", "Change (\%)",
                                   "Times"],
                                  caption="Experimental results of class numbers constraints on Reuters",
                                  ref="attr-level"))
