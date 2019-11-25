import json
import os
import timeit

import pandas as pd

from fairness.utils import *

female_ratio_lower_bound = 0.3215
female_ratio_upper_bound = 0.3415

# Number of iterations for most-vote
M = 10


def run_model(N, k, obj_matrix):
    '''
    :param N:
    :param k:
    :param obj_matrix:
    :return:
    '''
    from fairness.model import clustering_fairness

    model = clustering_fairness(N, k, obj_matrix, mustlink_detail, nb_detail)
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
        ans["cluster-violates"] = get_cluster_violates(ratios, female_ratio_lower_bound, female_ratio_upper_bound)
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
    neg_dis_to_centroid_matrix = np.full((N, k), 0.0)
    for i in range(N):
        for j in range(k):
            neg_dis_to_centroid_matrix[i][j] = -l2_distance(dis_space[i], kmeans.cluster_centers_[j])

    test_info["WCS"] = get_within_cluster_sum(kmeans.cluster_centers_, dis_space, km_clusters)
    print("WCS:", test_info["WCS"])

    ratios, num_violates = get_violates_and_ratios(nb_detail, km_clusters)
    test_info["violates"] = num_violates
    female_ratio_by_clusters = ratios_by_cluster(km_clusters, bin_att, k)
    test_info["cluster-violates"] = get_cluster_violates(female_ratio_by_clusters, female_ratio_lower_bound,
                                                         female_ratio_upper_bound)
    test_info["alpha"] = min(female_ratio_by_clusters)
    test_info["beta"] = max(female_ratio_by_clusters)
    print("Num violates:", num_violates)

    q = soft_assign(dis_space, kmeans.cluster_centers_, alpha=1.0)
    q = np.log(q)
    test_info["ILP-WCS"] = run_model(N, k, neg_dis_to_centroid_matrix)
    test_info["ILP-Prob"] = run_model(N, k, q)
    # Most-vote
    y_pred = kmeans.labels_
    partition = []
    start = timeit.default_timer()
    for iter in range(M):
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
                              "cluster-violates": get_cluster_violates(f_ratios, female_ratio_lower_bound,
                                                                       female_ratio_upper_bound)}
    # Add to total test
    total_test.append(test_info)

with open("result.json", 'w') as f:
    json.dump(total_test, f)
