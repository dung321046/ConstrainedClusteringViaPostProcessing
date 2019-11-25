import json
import os

import pandas as pd

from fairness.utils import *

female_ratio_lower_bound = 0.3215
female_ratio_upper_bound = 0.3415

original_data = pd.read_csv(
    "adult.full",
    names=[
        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],
    sep=r'\s*,\s*',
    engine='python',
    na_values="?")
N = len(original_data)
dis_space = original_data[["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]]

dis_space = np.asarray(dis_space)
dis_space = normalized(dis_space)

with open(os.path.join("neighbor.json"), 'r') as f:
    nb_detail = json.load(f)

sex_att = np.asarray(original_data[["Sex"]])
bin_att = []
female = []
for id, i in enumerate(sex_att):
    if i == "Male":
        bin_att.append(0)
    elif i == "Female":
        female.append(id)
        bin_att.append(1)
    else:
        raise Exception("missing gender", i)


def run_model(N, k, dis_matrix, p_l_group, lower_ratios, p_u_group, upper_ratios):
    from fairness.model import clustering_group_fairness

    model = clustering_group_fairness(N, k, dis_matrix, p_l_group, lower_ratios, p_u_group, upper_ratios)
    # model.write('clustering-fairness.lp')
    model.optimize()
    print("Run:", model.Runtime)
    if model.SolCount == 0:
        print("No solution found, status %d" % model.Status)
    else:
        c = model.__data
        partition = extract_cluster_id(N, c, k)
        ratios, num_violates = get_violates_and_ratios(nb_detail, partition)
    return {"time": model.Runtime, "WCS": get_within_cluster_sum(kmeans.cluster_centers_, dis_space, partition),
            "violates": num_violates}


import timeit

total_test = []
for test_id in range(10):
    test_info = dict()
    from sklearn.cluster import KMeans

    k = 5
    kmeans = KMeans(k, n_init=20, random_state=test_id)

    start = timeit.default_timer()
    km_clusters = kmeans.fit_predict(dis_space)
    test_info["time"] = (float)(timeit.default_timer() - start)
    print(to_str_ratios(ratios_by_cluster(km_clusters, bin_att, k)))

    cluster_dis_matrix = np.full((N, k), 0.0)
    for i in range(N):
        for j in range(k):
            cluster_dis_matrix[i][j] = -l2_distance(dis_space[i], kmeans.cluster_centers_[j])

    test_info["WCS"] = get_within_cluster_sum(kmeans.cluster_centers_, dis_space, km_clusters)
    test_info["alpha"] = min(ratios_by_cluster(km_clusters, bin_att, k))
    test_info["beta"] = max(ratios_by_cluster(km_clusters, bin_att, k))
    # Notes: should we use p or q. Q is more straight forward

    q = soft_assign(dis_space, kmeans.cluster_centers_, alpha=1.0)
    q = np.log(q)
    test_info["ILP-WCS"] = run_model(N, k, cluster_dis_matrix, [female], [female_ratio_lower_bound], [female],
                                     [female_ratio_upper_bound])
    test_info["ILP-Prob"] = run_model(N, k, q, [female], [female_ratio_lower_bound], [female],
                                      [female_ratio_upper_bound])
    total_test.append(test_info)

with open("result-group.json", 'w') as f:
    json.dump(total_test, f)
