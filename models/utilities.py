import statistics

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

NINF = -100000


def element_to_str(e):
    if type(e) == list:
        if len(e) == 1:
            e = e[0]
        elif len(e) == 0:
            return "-"
    if type(e) == int:
        if e == NINF:
            return "NaN"
        return "{:d}".format(e)
    if type(e) == str:
        return e
    if type(e) == tuple and len(e) == 2:
        if abs(e[1]) < 0.0001:
            if abs(e[0]) < 0.0001:
                return "0"
            return "{:.4f}".format(e[0])
        return "{:.4f} $\\pm$ {:.4f}".format(e[0], e[1])
    return "{:.4f}".format(e)


def arr_to_table_latex(arr, column_names, caption="", ref=""):
    string_file = "\\begin{table}[h]\n"
    string_file += "\\caption{" + caption + "}\\label{tab:" + ref + "}\n " + r"\resizebox{\columnwidth}!{" + "\n\\begin{tabular}{ |"
    for i in range(len(arr[0])):
        string_file += " r |"
    string_file += "}\n \\hline\n"
    for i in range(len(arr[0]) - 1):
        string_file += column_names[i] + " &"
    string_file += column_names[-1] + " \\\\ \n \\hline \n"
    for i in range(len(arr)):
        for j in range(len(arr[i]) - 1):
            string_file += element_to_str(arr[i][j]) + " & "
        string_file += element_to_str(arr[i][-1]) + "\\\\ \n  \n"
    string_file += "\\hline\n\\end{tabular}\n}\n"

    string_file += "\n\\end{table}"
    return string_file


def get_mean_and_std(arr):
    if len(arr) == 0:
        return -1000000, 0.0
    if len(arr) == 1:
        return arr[0], 0.0
    if type(arr[0]) == str:
        return arr[0]
    return statistics.mean(arr), np.std(np.asarray(arr))


def get_num_violates(ml, cl, y):
    ans = 0
    for i in range(len(ml)):
        u, v = int(ml[i][0]), int(ml[i][1])
        if y[u] != y[v]:
            ans += 1
    for i in range(len(cl)):
        u, v = int(cl[i][0]), int(cl[i][1])
        if y[u] == y[v]:
            ans += 1
    return ans


def get_point_violates(ml, cl, y):
    ans = set()
    for i in range(len(ml)):
        u, v = int(ml[i][0]), int(ml[i][1])
        if y[u] != y[v]:
            ans.add(u)
            ans.add(v)
    for i in range(len(cl)):
        u, v = int(cl[i][0]), int(cl[i][1])
        if y[u] == y[v]:
            ans.add(u)
            ans.add(v)
    return list(ans)


def get_error(true_v, v):
    return 1.0 * abs(true_v - v) / true_v


def count(ys, y_value):
    ans = 0
    for y in ys:
        if y == y_value:
            ans += 1
    return ans


def get_acc(label, pred):
    n = len(label)
    acc = 0.0
    for i in range(n):
        if label[i] == pred[i]:
            acc += 1.0
    return acc / n


def get_acc_with_permu(label, pred, permu):
    n = len(label)
    acc = 0.0
    for i in range(n):
        if label[i] == permu[pred[i]]:
            acc += 1.0
    return acc / n


def find_best_permutation(label, pred, k):
    from itertools import permutations
    a = np.full(k, 0)
    for i in range(k):
        a[i] = i
    best_acc = 0.0
    best_permu = []
    freq_dict = []
    for i in range(k):
        freq_dict.append(dict())
    for i, p in enumerate(pred):
        if label[i] not in freq_dict[p]:
            freq_dict[p][label[i]] = 1
        else:
            freq_dict[p][label[i]] += 1
    possible_permu = []
    for i in range(k):
        possible_permu.append(set())
    for i in range(k):
        for j in range(k):
            if j in freq_dict[i]:
                if freq_dict[i][j] > 5:
                    possible_permu[i].add(j)
            if len(freq_dict[i]) == 0:
                if j in freq_dict[i]:
                    possible_permu[i].add(j)
    # print(possible_permu)
    for pernumtation in permutations(a):
        possible = True
        for i, v in enumerate(pernumtation):
            if v not in possible_permu[i]:
                possible = False
                break
        if not possible:
            continue
        new_pre = []
        for p in pred:
            new_pre.append(pernumtation[p])
        acc = get_acc(label, new_pre)
        if acc > best_acc:
            best_acc = acc
            best_permu = pernumtation[:]
    return best_permu, best_acc


def extract_cluster_id(n, c, k):
    ans = []
    for i in range(n):
        is_add = False
        for j in range(k):
            if c[i, j].X == 1:
                ans.append(j)
                is_add = True
        if not is_add:
            for j in range(k):
                if abs(c[i, j].X - 1) < 0.00001:
                    ans.append(j)
                    break
    return ans


def calculate_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def convert_label(y_true, y_pred):
    """

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        the label of y_pred based on y_true
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    mapping_dict = dict()
    for i, j in ind:
        mapping_dict[i] = j
    ans = []
    for y in y_pred:
        ans.append(mapping_dict[y])
    return ans


def get_cluster_size(y):
    cluster_size_dict = dict()
    for i in y:
        if i not in cluster_size_dict:
            cluster_size_dict[i] = 0
        cluster_size_dict[i] += 1
    return list(cluster_size_dict.values())


def get_percent_change(y_bft, y_aft):
    per_change = 0.0
    n = len(y_bft)
    for i in range(n):
        if y_bft[i] != y_aft[i]:
            per_change += 1.0
    return 100.0 * per_change / n


def get_num_change(y_bft, y_aft):
    per_change = 0
    n = len(y_bft)
    for i in range(n):
        if y_bft[i] != y_aft[i]:
            per_change += 1
    return per_change


def get_pos_change(label, y_bft, y_aft):
    pos_change = 0
    n = len(y_bft)
    for i in range(n):
        if y_bft[i] != y_aft[i] and y_aft[i] == label[i]:
            pos_change += 1
    return pos_change


def metrics(label, partition):
    nmi = normalized_mutual_info_score(label, partition, average_method="arithmetic")
    ari = adjusted_rand_score(label, partition)
    acc = calculate_acc(np.asarray(label), np.asarray(partition))
    return nmi, ari, acc


def get_cluster_center(dis_space, pred):
    n = len(dis_space)
    f = len(dis_space[0])
    clusters = dict()
    for i in range(n):
        if pred[i] not in clusters:
            clusters[pred[i]] = [i]
        else:
            clusters[pred[i]].append(i)
    ans = dict()
    for key in clusters.keys():
        ans[key] = np.full(f, 0.0)
        for i in clusters[key]:
            ans[key] += dis_space[i]
        ans[key] /= len(clusters[key])
    ans = [a for a in ans.values()]
    return ans


import torch


def soft_assign(z, mu, alpha=1.0):
    z = torch.tensor(z)
    mu = torch.tensor(mu)
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - mu) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q


def l2_distance(point1, point2):
    # print(type(point1))
    # print(type(point2))
    # print(point1.shape)
    # print(point2.shape)
    return np.linalg.norm(point1 - point2)
    # return sum([(float(i)-float(j))**2 for (i, j) in zip(point1, point2)])
