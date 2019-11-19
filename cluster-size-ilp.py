import argparse

from sklearn.cluster import KMeans

from gurobi_import import *
from lib.datasets import MNIST, FashionMNIST, Reuters
from lib.dec_with_pw import IDEC
from utilities import *

keep_import()


def clustering_car_pw(n, k, dis_matrix, ml, cl, lower, upper):
    model = Model("Clustering with cardinality and pairwise constraints")
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

    for j in range(k):
        coef = []
        var = []
        for i in range(n):
            coef.append(1)
            var.append(x[i, j])
        model.addConstr(LinExpr(coef, var), ">=", lower, name="lower-car[%s]" % j)
        model.addConstr(LinExpr(coef, var), "<=", upper, name="upper-car[%s]" % j)
    model.update()
    model.__data = x
    return model


from scipy.spatial import distance


def build_dis(x, centers):
    n = len(x)
    k = len(centers)
    dis_matrix = []
    for i in range(n):
        row = []
        for j in range(k):
            row.append(distance.euclidean(x[i], centers[j]))
        dis_matrix.append(row)
    return dis_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pairwise MNIST Example')
    parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
    parser.add_argument('--pretrain', type=str, default="./model/idec_mnist.pt", metavar='N',
                        help='directory for pre-trained weights')
    args = parser.parse_args()
    k = 10
    idec = None
    if args.data == "Fashion":
        fashionmnist_train = FashionMNIST('./dataset/fashion_mnist', train=True, download=True)
        X = fashionmnist_train.train_data
        y = fashionmnist_train.train_labels
        args.pretrain = "./model/idec_fashion.pt"
        idec = IDEC(input_dim=784, z_dim=10, n_clusters=10,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    elif args.data == "Reuters":
        reuters_train = Reuters('./dataset/reuters', train=True, download=False)
        args.pretrain = "./model/idec_reuters.pt"
        X = reuters_train.train_data
        y = reuters_train.train_labels
        k = 4
        idec = IDEC(input_dim=2000, z_dim=10, n_clusters=4,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    elif args.data == "MNIST":
        mnist_train = MNIST('./dataset/mnist', train=True, download=False)
        X = mnist_train.train_data
        y = mnist_train.train_labels
        args.pretrain = "./model/idec_mnist.pt"
        idec = IDEC(input_dim=784, z_dim=10, n_clusters=10,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    idec.load_model(args.pretrain)
    print(type(X), X.shape)
    latent = idec.encodeBatch(X)
    X = np.asarray(latent)
    # X = np.asarray(X)
    feature_size = len(X[0])
    y = np.asarray(y)
    n = len(y)
    true_cluster_sizes = get_cluster_size(y)
    min_car = min(true_cluster_sizes)
    max_car = max(true_cluster_sizes)
    # min_car = 0
    # max_car = 100000000
    kmeans = KMeans(k, n_init=20)
    y_pred = kmeans.fit_predict(X)
    centers = np.asarray(kmeans.cluster_centers_)
    for epoch_id in range(300):
        dis_matrix = build_dis(X, centers)
        model = clustering_car_pw(n, k, dis_matrix, [], [], min_car, max_car)
        # model.write('clustering-car-pw.lp')
        model.optimize()
        if model.SolCount == 0:
            print("No solution found, status %d" % model.Status)
            break
        c = model.__data
        partition = np.asarray(extract_cluster_id(n, c, k))
        per_change = get_percent_change(y_pred, partition)
        print("Acc:", calculate_acc(y, partition))
        print("Change:", per_change)
        if per_change < 0.001:
            break
        y_pred = partition
        new_centers = []
        cluster_sizes = dict()
        for i in range(k):
            new_centers.append(np.full(feature_size, 0.0))
        for i in range(n):
            new_centers[partition[i]] += X[i]
            if partition[i] not in cluster_sizes:
                cluster_sizes[partition[i]] = 0.0
            cluster_sizes[partition[i]] += 1.0
        # print(cluster_sizes)
        # print(new_centers[:3])
        distance_center = 0.0
        for i in range(k):
            new_centers[i] = new_centers[i] / cluster_sizes[i]
            distance_center += distance.euclidean(new_centers[i], centers[i])
        print("Distance:", distance_center)
        centers = np.asarray(new_centers)

    print("Final eval:", metrics(y, y_pred))
