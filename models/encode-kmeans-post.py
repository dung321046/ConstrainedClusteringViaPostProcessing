import argparse
import timeit

from lib_deep_clustering.datasets import MNIST, FashionMNIST, Reuters
from lib_deep_clustering.dec_with_pw import IDEC
from models.gurobi_import import *
from models.utilities import *

keep_import()


def build_dis(x, centers):
    n = len(x)
    k = len(centers)
    dis_matrix = []
    for i in range(n):
        row = []
        for j in range(k):
            row.append(l2_distance(x[i], centers[j]))
        dis_matrix.append(row)
    return dis_matrix


def adjust_dis_mtrx(x, centers, changed_clusters):
    n = len(x)
    dis_matrix = []
    for i in range(n):
        row = []
        for j in changed_clusters:
            row.append(l2_distance(x[i], centers[j]))
        dis_matrix.append(row)
    return dis_matrix


def read_ml_cl(folder_name):
    ml = np.loadtxt(os.path.join(folder_name, "ml.txt"), dtype=int)
    cl = np.loadtxt(os.path.join(folder_name, "cl.txt"), dtype=int)
    return ml, cl


def clustering_pw(n, k, dis_matrix, ml, cl):
    model = Model("Clustering with pairwise constraints")
    model.setParam('OutputFlag', False)
    x = {}
    for i in range(n):
        for j in range(k):
            x[i, j] = model.addVar(obj=dis_matrix[i][j], vtype="B", name="x[%s,%s]" % (i, j))
    for i in range(n):
        coef = []
        var = []
        for j in range(k):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), "=", 1, name="1-cluster-for-point[%s]" % i)
    for i in range(len(ml)):
        for j in range(k):
            u, v = ml[i][0], ml[i][1]
            model.addConstr(x[u, j], "=", x[v, j], name="ML[%s]Cluster[%s]" % (i, j))
    for i in range(len(cl)):
        for j in range(k):
            u, v = cl[i][0], cl[i][1]
            model.addConstr(LinExpr([1, 1], [x[u, j], x[v, j]]), "<=", 1, name="CL[%s]Cluster[%s]" % (i, j))
    model.update()
    model.__data = x
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairwise MNIST Example')
    parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
    args = parser.parse_args()
    args.pretrain = "./model/idec_mnist.pt"
    mnist_train = MNIST('./model_weight/mnist', train=True, download=True)
    X = mnist_train.train_data
    Y = mnist_train.train_labels
    k = 10
    idec = IDEC(input_dim=784, z_dim=10, n_clusters=10,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    if args.data == "Fashion":
        args.pretrain = "./model/idec_fashion.pt"
        fashionmnist_train = FashionMNIST('./model_weight/fashion_mnist', train=True, download=True)
        X = fashionmnist_train.train_data
        Y = fashionmnist_train.train_labels
    elif args.data == "Reuters":
        args.pretrain = "./model_weight/idec_reuters.pt"
        reuters_train = Reuters('./dataset/reuters', train=True, download=True)
        X = reuters_train.train_data
        Y = reuters_train.train_labels
        k = 4
        idec = IDEC(input_dim=2000, z_dim=10, n_clusters=4,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    # Take the encoding of IDEC as input
    idec.load_model(args.pretrain)
    latent = idec.encodeBatch(X)
    X = np.asarray(latent)

    Y = np.asarray(Y)
    N = len(Y)
    folder_name = "./sample_test/" + args.data
    import os
    from sklearn.cluster import KMeans

    for pairwise_factor in [6]:
        pairwise_num = int(pairwise_factor * N / 100)
        stat_pw = []
        for test in range(1):
            # Read pairwise
            test_folder = folder_name + str(pairwise_num) + "/test" + str(test).zfill(2)
            must_link, cannot_link = read_ml_cl(test_folder)

            start = timeit.default_timer()
            kmeans = KMeans(k, n_init=20, random_state=1)
            clustering = np.asarray(kmeans.fit_predict(X))
            freq = np.full(k, 0)
            for i in clustering:
                freq[i] += 1
            centers = np.asarray(kmeans.cluster_centers_)
            dis_matrix = build_dis(X, centers)
            for epoch_id in range(300):
                model = clustering_pw(N, k, dis_matrix, must_link, cannot_link)
                # model.write('clustering-car-pw.lp')
                model.optimize()
                if model.SolCount == 0:
                    print("No solution found, status %d" % model.Status)
                    break
                c = model.__data
                partition = np.asarray(extract_cluster_id(N, c, k))
                per_change = get_percent_change(clustering, partition)
                print("Iter %2s - Acc: %.3f - Change: %.3f" % (epoch_id, calculate_acc(Y, partition), per_change))
                if per_change < 0.001:
                    break
                for i, center in enumerate(centers):
                    center *= freq[i]
                changed_clts = set()
                for i in range(k):
                    changed_clts.add(i)
                for i in range(N):
                    if clustering[i] != partition[i]:
                        changed_clts.discard(clustering[i])
                        changed_clts.discard(partition[i])
                        centers[clustering[i]] -= X[i]
                        centers[partition[i]] += X[i]
                freq = np.full(k, 0)
                for i in partition:
                    freq[i] += 1
                for i in range(k):
                    centers[i] = centers[i] / freq[i]
                clustering = partition
                adjust_dis_mtrx(X, centers, changed_clts)
            time_running = (float)(timeit.default_timer() - start)
            print("Finish COP-KMeans!")

            clusters = np.asarray(partition)
            os.makedirs(test_folder + "/encode-kmeans-post", exist_ok=True)
            # Save the clustering result
            np.savetxt(test_folder + "/encode-kmeans-post/label.txt", clusters, fmt='%s')
            # Save runtime, acc and nmi
            _acc = calculate_acc(Y, clusters)
            _nmi = normalized_mutual_info_score(Y, clusters, average_method="arithmetic")
            local_stat = np.asarray([time_running, _acc, _nmi])
            np.savetxt(test_folder + "/encode-kmeans-post/stat.txt", local_stat, fmt='%.6f')
