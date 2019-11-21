import pandas as pd

from fairness.utils import *

original_data = pd.read_csv("adult.full",
                            names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                                   "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                                   "Hours per week", "Country", "Target"],
                            sep=r'\s*,\s*',
                            engine='python',
                            na_values="?")

N = len(original_data)

print("Total:", N)
dis_space = original_data[["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]]

dis_space = np.asarray(dis_space)
dis_space = normalized(dis_space)

from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(k, n_init=20, random_state=0)
y_pred = kmeans.fit_predict(dis_space)

print("1:", kmeans.cluster_centers_)
print("2:", get_cluster_center(dis_space, y_pred))
print("WCS:", get_within_cluster_sum(kmeans.cluster_centers_, dis_space, y_pred))
cat = np.asarray(original_data[["Age", "Education", "Occupation"]])

neighbors, same_neighbor_groups = create_neighbor(cat)
print("Same neighbor group:", same_neighbor_groups)
import os
import json

with open(os.path.join("neighbor.json"), 'w') as fp:
    json.dump(neighbors, fp)

with open(os.path.join("same-neighbor.json"), 'w') as fp:
    json.dump(same_neighbor_groups, fp)

ratios = []
num_violates = 0
sizes = []
zero_nb = 0
for i in range(N):
    sizes.append(len(neighbors[i]))
    if len(neighbors[i]) == 0:
        zero_nb += 1
        print(cat[i])
        continue
    num_same_class = get_num_same_class(neighbors[i], y_pred, y_pred[i])
    ratios.append(num_same_class / len(neighbors[i]))
    if num_same_class < len(neighbors[i]) * 0.5:
        num_violates += 1

print("Size:", N)
print("#Violates:", num_violates)
print("Zero nb:", zero_nb)

import matplotlib.pyplot as plt

plt.hist(ratios, bins=50)
plt.xlabel('Same class ratios')
plt.ylabel('Number of instances')
plt.show()
