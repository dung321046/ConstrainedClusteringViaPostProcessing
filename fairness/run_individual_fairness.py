import json
import os
import timeit

import pandas as pd

from fairness.utils import *

alpha = 0.3215
beta = 0.3415


def run_model(N, k, obj_matrix):
    from fairness.model import clustering_fairness

    model = clustering_fairness(N, k, obj_matrix, mustlink_detail, nb_detail)
    # model = clustering_fairness(N, k, cluster_dis_matrix, mustlink_detail, nb_detail)

    # model.write('clustering-fairness.lp')
    model.optimize()
    ans = {"time": model.Runtime}
    if model.SolCount == 0:
        print("No solution found, status %d" % model.Status)
        ans["WCS"] = NINF
    else:
        c = model.__data
        partition = extract_cluster_id(N, c, k)
        # ans["label"] = partition
        ans["WCS"] = get_within_cluster_sum(kmeans.cluster_centers_, dis_space, partition)
        ratios = ratios_by_cluster(partition, bin_att, k)
        ans["alpha"] = min(ratios)
        ans["beta"] = max(ratios)
        ans["cluster-violates"] = get_cluster_violates(ratios, alpha, beta)
        # print(to_str_ratios(ratios_by_cluster(partition, bin_att, k)))
        print("Constrained WCS:", ans["WCS"])
        ratios, num_violates = get_violates_and_ratios(nb_detail, partition)
        print("Num violates:", num_violates)
    return ans


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

with open(os.path.join("same-neighbor.json"), 'r') as f:
    mustlink_detail = json.load(f)

# Group fairness detail
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

total_test = []
for i in range(10):
    test_info = dict()
    from sklearn.cluster import KMeans

    k = 5
    kmeans = KMeans(k, n_init=20, random_state=i)
    start = timeit.default_timer()
    km_clusters = kmeans.fit_predict(dis_space)
    test_info["time"] = (float)(timeit.default_timer() - start)
    cluster_dis_matrix = np.full((N, k), 0.0)
    for i in range(N):
        for j in range(k):
            cluster_dis_matrix[i][j] = -l2_distance(dis_space[i], kmeans.cluster_centers_[j])

    test_info["WCS"] = get_within_cluster_sum(kmeans.cluster_centers_, dis_space, km_clusters)
    print("WCS:", test_info["WCS"])

    ratios, num_violates = get_violates_and_ratios(nb_detail, km_clusters)
    test_info["violates"] = num_violates
    female_ratio_by_clusters = ratios_by_cluster(km_clusters, bin_att, k)
    test_info["cluster-violates"] = get_cluster_violates(female_ratio_by_clusters, alpha, beta)
    test_info["alpha"] = min(female_ratio_by_clusters)
    test_info["beta"] = max(female_ratio_by_clusters)
    print("Num violates:", num_violates)

    # Notes: should we use p or q. Q is more straight forward

    q = soft_assign(dis_space, kmeans.cluster_centers_, alpha=1.0)
    q = np.log(q)
    # p = target_distribution(q)
    test_info["ILP-WCS"] = run_model(N, k, cluster_dis_matrix)
    test_info["ILP-Prob"] = run_model(N, k, q)
    # Most-vote
    y_pred = kmeans.labels_
    partition = []
    start = timeit.default_timer()
    for iter in range(10):
        partition = get_major_votes(nb_detail, y_pred)
        y_pred = partition
    time_running = (float)(timeit.default_timer() - start)
    ratios, num_violates = get_violates_and_ratios(nb_detail, partition)
    f_ratios = ratios_by_cluster(partition, bin_att, k)
    test_info["Most-vote"] = {"time": time_running,
                              "WCS": get_within_cluster_sum(kmeans.cluster_centers_, dis_space, partition),
                              "violates": num_violates,
                              "alpha": min(f_ratios),
                              "beta": max(f_ratios),
                              "cluster-violates": get_cluster_violates(f_ratios, alpha, beta)}
    # Add to total test
    total_test.append(test_info)

with open("result.json", 'w') as f:
    json.dump(total_test, f)
