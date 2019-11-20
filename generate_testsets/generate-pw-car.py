import argparse
import os
import random

import numpy as np

from lib_deep_clustering.datasets import MNIST, FashionMNIST, Reuters
from lib_deep_clustering.dec_with_pw import IDEC
from lib_deep_clustering.utils import generate_random_pair

if __name__ == "__main__":
    np.random.seed(270693)
    random.seed(270693)
    parser = argparse.ArgumentParser(description='Generating PW constraints with compact factor')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--update-interval', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--pretrain', type=str, default="../model/idec_mnist.pt", metavar='N',
                        help='directory for pre-trained weights')
    parser.add_argument('--data', type=str, default="MNIST", metavar='N', help='dataset(MNIST, Fashion, Reuters)')
    parser.add_argument('--use_pretrain', type=bool, default=True)
    args = parser.parse_args()
    # Set parameters
    ml_penalty, cl_penalty = 0.1, 1
    k = 10
    if args.data == "Fashion":
        fashionmnist_train = FashionMNIST('../dataset/fashion_mnist', train=True, download=True)
        X = fashionmnist_train.train_data
        y = fashionmnist_train.train_labels
        args.pretrain = "../model/idec_fashion.pt"
        ml_penalty = 1
        idec = IDEC(input_dim=784, z_dim=10, n_clusters=10,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    elif args.data == "Reuters":
        reuters_train = Reuters('../dataset/reuters', train=True, download=False)
        X = reuters_train.train_data
        y = reuters_train.train_labels
        args.pretrain = "../model/idec_reuters.pt"
        k = 4
        idec = IDEC(input_dim=2000, z_dim=10, n_clusters=4,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    else:
        mnist_train = MNIST('../dataset/mnist', train=True, download=False)
        X = mnist_train.train_data
        y = mnist_train.train_labels
        args.pretrain = "../model/idec_mnist.pt"
        idec = IDEC(input_dim=784, z_dim=10, n_clusters=10,
                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)
    if args.use_pretrain:
        idec.load_model(args.pretrain)
    y = np.asarray(y)
    N = len(y)
    print("Test size:", N)
    latent = idec.encodeBatch(X)
    q = idec.soft_assign(latent)
    p = idec.target_distribution(q).data.cpu().numpy()
    logp = np.empty([len(p), len(p[0])])
    for i in range(len(p)):
        for j in range(len(p[i])):
            logp[i][j] = np.log(p[i][j])
    pred = np.argmax(logp, axis=1)

    folder_name = args.data
    os.mkdir(folder_name)
    for pairwise_factor in [6, 50, 100]:
        pairwise_num = int(pairwise_factor * N / 100)
        print(pairwise_num)
        folder_prefix_path = folder_name + "/" + str(pairwise_num) + "/"
        os.mkdir(folder_prefix_path)
        folder_prefix_path += "test"
        for seed in range(10):
            np.random.seed(270693 + seed)
            random.seed(270693 + seed)
            folder_path = folder_prefix_path + str(seed).zfill(2)
            os.mkdir(folder_path)
            car = np.zeros(k, dtype=int)
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                car[u] = c
            ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(y, pairwise_num)
            main_var = np.array([N, k, len(ml_ind1), len(cl_ind1)])
            np.savetxt(os.path.join(folder_path, "ml.txt"), np.column_stack((ml_ind1, ml_ind2)), fmt='%s')
            np.savetxt(os.path.join(folder_path, "cl.txt"), np.column_stack((cl_ind1, cl_ind2)), fmt='%s')
            np.savetxt(os.path.join(folder_path, "main_var.txt"), main_var, fmt='%s')
            np.savetxt(os.path.join(folder_path, "pdis.txt"), logp, fmt='%.6f')
            np.savetxt(os.path.join(folder_path, "cardinality.txt"), car, fmt='%s')
            np.savetxt(os.path.join(folder_path, "label"), y, fmt='%s')
