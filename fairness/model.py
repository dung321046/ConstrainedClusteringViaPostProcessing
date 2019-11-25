from models.gurobi_import import *

keep_import()

import numpy as np

ALPHA = 0.5


def clustering_fairness(n, k, allocation_matrix, ml_details, neighbor_details):
    '''

    :param n: number of instances
    :param k: number of clusters
    :param allocation_matrix: A[ij] is the likelihood of assigning: instance X_i to cluster C_j
    :param ml_details: list of pairs: Two instances in each pair (u,v) must have same neighbor -> X_uj = X_vj for all
    j \in [1,K] if ALPHA >= 0.5
    :param neighbor_details: array of n elements, each element N_i is the list of all neighbor of the instance X_i
    :return: Optimized model of ILP
    '''
    model = Model("Clustering with fairness constraints")
    model.setParam('OutputFlag', False)
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=-allocation_matrix[i, j], vtype="B", name="x[%s,%s]" % (i, j))
    for i in range(n):
        coef = []
        var = []
        for j in range(k):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), "=", 1, name="1-cluster-for-point[%s]" % i)
    for i in range(k):
        coef = []
        var = []
        for j in range(n):
            coef.append(1)
            var.append(x[j, i])
        model.addConstr(LinExpr(coef, var), ">=", 1, name="at-least-1-point-for-cluster[%s]" % i)
    is_presentation = np.full(n, True)
    for ml in ml_details:
        for i in range(len(ml) - 1):
            is_presentation[ml[i + 1]] = False
            for j in range(k):
                u, v = ml[i], ml[i + 1]
                model.addConstr(x[u, j], "=", x[v, j], name="ML[%s]Cluster[%s]" % (i, j))
    for const_id, neighbor in enumerate(neighbor_details):
        if not is_presentation[const_id]:
            continue
        for cluster_id in range(k):
            coef2 = [-len(neighbor) * ALPHA]
            var2 = [x[const_id, cluster_id]]
            for row in neighbor:
                # if not is_presentation[row]:
                #     continue
                coef2.append(1)
                var2.append(x[row, cluster_id])
            model.addConstr(LinExpr(coef2, var2), ">=", 0, name="fairness %s on %s" % (const_id, cluster_id))
    model.update()
    model.__data = x
    return model


def clustering_group_fairness(n, k, allocation_matrix, protected_lower_groups, lower_ratios, protected_upper_groups,
                              upper_ratios):
    """

    :param n: number of instances
    :param k: number of clusters
    :param allocation_matrix: A[ij] is the likelihood of assigning: instance X_i to cluster C_j
    :param protected_lower_groups:  groups that we want to have a lower bound
    :param lower_ratios: lower bound of ratio: each protected group over cluster size for all cluster
    :param protected_upper_groups: groups that we want to have a upper bound
    :param upper_ratios: upper bound of ratio: each protected group over cluster size for all cluster
    :return:  Optimized model of ILP
    """
    model = Model("Clustering with fairness constraints")
    model.setParam('OutputFlag', False)
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=-allocation_matrix[i, j], vtype="B", name="x[%s,%s]" % (i, j))
    for i in range(n):
        coef = []
        var = []
        for j in range(k):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), "=", 1, name="1-cluster-for-point[%s]" % i)
    for i in range(k):
        coef = []
        var = []
        for j in range(n):
            coef.append(1)
            var.append(x[j, i])
        model.addConstr(LinExpr(coef, var), ">=", 1, name="at-least-1-point-for-cluster[%s]" % i)
    for t, g in enumerate(protected_lower_groups):
        alpha = lower_ratios[t]
        for h in range(k):
            coef = []
            var = []
            for i in range(n):
                var.append(x[i, h])
                if i in g:
                    coef.append(1 - alpha)
                else:
                    coef.append(-alpha)
            model.addConstr(LinExpr(coef, var), ">=", 0, name="lower-ratio-for-group[%s]-cluster[%s] " % (t, h))
    for t, g in enumerate(protected_upper_groups):
        beta = upper_ratios[t]
        for h in range(k):
            coef = []
            var = []
            for i in range(n):
                var.append(x[i, h])
                if i in g:
                    coef.append(1 - beta)
                else:
                    coef.append(-beta)
            model.addConstr(LinExpr(coef, var), "<=", 0, name="upper-ratio-for-group[%s]-cluster[%s] " % (t, h))
    model.update()
    model.__data = x
    return model


def clustering_combine_fairness(n, k, allocation_matrix, protected_lower_groups, lower_ratios, protected_upper_groups,
                                upper_ratios, ml_details, neighbor_details):
    model = Model("Clustering with fairness constraints")
    model.setParam('OutputFlag', False)
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=-allocation_matrix[i, j], vtype="B", name="x[%s,%s]" % (i, j))
    for i in range(n):
        coef = []
        var = []
        for j in range(k):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), "=", 1, name="1-cluster-for-point[%s]" % i)
    for i in range(k):
        coef = []
        var = []
        for j in range(n):
            coef.append(1)
            var.append(x[j, i])
        model.addConstr(LinExpr(coef, var), ">=", 1, name="at-least-1-point-for-cluster[%s]" % i)
    for t, g in enumerate(protected_lower_groups):
        lower_bound = lower_ratios[t]
        for h in range(k):
            coef = []
            var = []
            for i in range(n):
                var.append(x[i, h])
                if i in g:
                    coef.append(1 - lower_bound)
                else:
                    coef.append(-lower_bound)
            model.addConstr(LinExpr(coef, var), ">=", 0, name="lower-ratio-for-group[%s]-cluster[%s] " % (t, h))
    for t, g in enumerate(protected_upper_groups):
        upper_bound = upper_ratios[t]
        for h in range(k):
            coef = []
            var = []
            for i in range(n):
                var.append(x[i, h])
                if i in g:
                    coef.append(1 - upper_bound)
                else:
                    coef.append(-upper_bound)
            model.addConstr(LinExpr(coef, var), "<=", 0, name="upper-ratio-for-group[%s]-cluster[%s] " % (t, h))
    is_presentation = np.full(n, True)
    for ml in ml_details:
        for i in range(len(ml) - 1):
            is_presentation[ml[i + 1]] = False
            for j in range(k):
                u, v = ml[i], ml[i + 1]
                model.addConstr(x[u, j], "=", x[v, j], name="ML[%s]Cluster[%s]" % (i, j))
    for const_id, neighbor in enumerate(neighbor_details):
        if not is_presentation[const_id]:
            continue
        for cluster_id in range(k):
            coef2 = [-len(neighbor) * ALPHA]
            var2 = [x[const_id, cluster_id]]
            for row in neighbor:
                coef2.append(1)
                var2.append(x[row, cluster_id])
            model.addConstr(LinExpr(coef2, var2), ">=", 0, name="fairness %s on %s" % (const_id, cluster_id))
    model.update()
    model.__data = x
    return model
