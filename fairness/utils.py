import statistics

import numpy as np
import torch
from sklearn import preprocessing

NINF = -1000000000


def normalized(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return x_scaled


def soft_assign(z, mu, alpha=1.0):
    z = torch.tensor(z)
    mu = torch.tensor(mu)
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - mu) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q


def target_distribution(q):
    p = q ** 2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p


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


def l2_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def get_within_cluster_sum(centers, x, clusters):
    sum = 0
    for i, p in enumerate(x):
        sum += l2_distance(p, centers[clusters[i]])
    return sum


def get_max_split(x, Y):
    max_split = 100000000.0
    for i in range(len(Y)):
        for j in range(i):
            if Y[i] != Y[j]:
                max_split = min(l2_distance(x[i], x[j]), max_split)
    return max_split


def get_num_same_class(neighbor, clusterings, cluster_id):
    ans = 0
    for n in neighbor:
        if clusterings[n] == cluster_id:
            ans += 1
    return ans


def get_class_distribution(neighbor, clusterings):
    ans = dict()
    for n in neighbor:
        if clusterings[n] not in ans:
            ans[clusterings[n]] = 1
        else:
            ans[clusterings[n]] += 1
    return ans


def get_violates_and_ratios(neighbors, y_pred):
    ratios = []
    num_violates = 0
    for i in range(len(neighbors)):
        if len(neighbors[i]) == 0:
            continue
        num_same_class = get_num_same_class(neighbors[i], y_pred, y_pred[i])
        ratios.append(num_same_class / len(neighbors[i]))
        if num_same_class < len(neighbors[i]) * 0.5:
            num_violates += 1

    return ratios, num_violates


def get_major_votes(neighbors, y_pred):
    y_change = []
    for i in range(len(neighbors)):
        if len(neighbors[i]) == 0:
            y_change.append(y_pred[i])
            continue
        t = get_class_distribution(neighbors[i], y_pred)
        import operator
        sorted_x = sorted(t.items(), key=operator.itemgetter(1))
        y_change.append(sorted_x[-1][0])
    return y_change


def get_num_change(y_bft, y_aft):
    per_change = 0
    n = len(y_bft)
    for i in range(n):
        if y_bft[i] != y_aft[i]:
            per_change += 1
    return per_change


def get_dis_matrix(dis_space):
    N = len(dis_space)
    dis_matrix = np.full((N, N), 0)
    for i in range(N):
        for j in range(N):
            dis_matrix[i][j] = l2_distance(dis_space[i], dis_space[j])
    return dis_matrix


def get_ncut(sim_matrix, pred):
    n = len(sim_matrix[0])
    clusters = dict()
    for i in range(n):
        if pred[i] not in clusters:
            clusters[pred[i]] = [i]
        else:
            clusters[pred[i]].append(i)
    ans = 0.0
    for key in clusters.keys():
        vol = 0.0
        link = 0.0
        others = []
        for i in range(n):
            if pred[i] != key:
                others.append(i)
        for i in clusters[key]:
            vol += sum(sim_matrix[i])
            for j in others:
                link += sim_matrix[i][j]
        ans += 0.5 * link / vol
    return ans


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


def take_subsample(x, k):
    n = len(x)
    idx = np.random.permutation(n)[:k]
    return idx, np.take(x, idx, axis=0)


def create_neighbor(cat):
    group = dict()
    for i, c in enumerate(cat):
        if (c[1], c[2]) not in group:
            group[(c[1], c[2])] = [i]
            continue
        group[(c[1], c[2])].append(i)

    neighbors = []
    for i in range(len(cat)):
        neighbors.append([])
    same_cat_groups = []
    for g in group.values():
        sub_scg = dict()
        for i in g:
            if cat[i][0] not in sub_scg:
                sub_scg[cat[i][0]] = [i]
            else:
                sub_scg[cat[i][0]].append(i)
            neighbor = []
            for j in g:
                if i != j and abs(cat[i][0] - cat[j][0]) <= 2:
                    neighbor.append(j)
            neighbors[i] = neighbor
        for sg in sub_scg.values():
            if len(sg) > 1:
                same_cat_groups.append(sg)
    return neighbors, same_cat_groups


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


def get_mean_and_std(arr):
    if len(arr) == 0:
        return NINF, 0.0
    if len(arr) == 1:
        return arr[0], 0.0
    if type(arr[0]) == str:
        return arr[0]
    return statistics.mean(arr), np.std(np.asarray(arr))


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
        return "{:.2f} $\\pm$ {:.2f}".format(e[0], e[1])
    return "{:.4f}".format(e)


def arr_to_table_latex(arr, column_names, caption="", ref=""):
    string_file = "\\begin{table}[h]\n"
    string_file += "\\caption{" + caption + "}\n"
    string_file += "\\label{tab:" + ref + "}" + r"\resizebox{\columnwidth}!{" + "\n\\begin{tabular}{ |"
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
    string_file += "\\hline\n\\end{tabular}\n}\n\\end{table}"

    return string_file


def convert_sim_matrix(n, list_weight):
    sim_matrix = np.full((n, n), 0.0)
    k = 0

    for edge in list_weight[0]:
        k += 1
        if k < 10:
            print("+", edge)
    print("----", len(list_weight))


def freqs_by_cluster(clustering, att, k):
    freq = []
    for i in range(k):
        freq.append({0: 0, 1: 0})
    for i, c in enumerate(clustering):
        freq[c][att[i]] += 1
    return freq


def ratios_by_cluster(clustering, att, k, selected_att=1):
    freqs = freqs_by_cluster(clustering, att, k)
    ans = []
    for freq in freqs:
        ans.append(1.0 * freq[selected_att] / (freq[0] + freq[1]))
    return ans


def to_str_ratios(ratios):
    for i, ratio in enumerate(ratios):
        print("(" + str(i + 1) + ",", ratio, ")")


def get_cluster_violates(ratios, alpha, beta):
    ans = 0
    for r in ratios:
        if r < alpha or r > beta:
            ans += 1

    return ans

