import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.autograd import Variable
from torch.nn import Parameter

from lib.utils import acc

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2)


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class IDEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
                 encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0, alpha=1., gamma=0.1):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(z)
        xrecon = self._dec(h)
        # compute q -> NxK
        q = self.soft_assign(z)
        return z, q, xrecon

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def encodeBatch(self, X, batch_size=256):
        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        kldloss = kld(p, q)
        return self.gamma * kldloss

    def recon_loss(self, x, xrecon):
        recon_loss = torch.mean((xrecon - x) ** 2)
        return recon_loss

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return cl_loss

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def predict(self, X, y):
        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     self.cuda()
        latent = self.encodeBatch(X)
        q = self.soft_assign(latent)

        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        y = y.data.cpu().numpy()
        if y is not None:
            print("acc: %.5f, nmi: %.5f" % (
                acc(y, y_pred), normalized_mutual_info_score(y, y_pred, average_method="arithmetic")))
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred, average_method="arithmetic")
        return final_acc, final_nmi

    def fit(self, stat, ml_list, cl_list, ml_ind1, ml_ind2, cl_ind1, cl_ind2, ml_p, cl_p, X,
            y=None, lr=0.001, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3):
        '''X: tensor data'''
        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     self.cuda()
        print("=====Training IDEC=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = self.encodeBatch(X)
        y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        y_pred_last = y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            y = y.cpu().numpy()
            # print("Kmeans acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        ml_num_batch = int(math.ceil(1.0 * ml_ind1.shape[0] / batch_size))
        cl_num_batch = int(math.ceil(1.0 * cl_ind1.shape[0] / batch_size))
        cl_num = cl_ind1.shape[0]
        ml_num = ml_ind1.shape[0]
        last_acc, last_nmi, final_epoch = 0, 0, 0
        update_ml = 1
        update_cl = 1
        last_delta = 0.0
        last_nmi = 1.0
        arr_acc = []
        tor_acc = 0.0005
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
                    num_vcon = 0
                    for i in range(len(ml_list[0])):
                        if y_pred[ml_list[0][i]] != y_pred[ml_list[1][i]]:
                            num_vcon += 1
                    for i in range(len(cl_list[0])):
                        if y_pred[cl_list[0][i]] == y_pred[cl_list[1][i]]:
                            num_vcon += 1
                    last_acc = acc(y, y_pred)
                    last_nmi = normalized_mutual_info_score(y, y_pred, average_method="arithmetic")
                    arr_acc.append(last_acc)
                    if len(arr_acc) > 4:
                        arr_acc = arr_acc[1:]
                    final_epoch = epoch
                    print("NMI -  ACC - #violated constraints")
                    stat.append((last_nmi, last_acc, num_vcon, last_delta))
                    print("%.5f\t%.5f\t%d\n" % (last_nmi, last_acc, num_vcon))
                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                y_pred_last = y_pred
                if max(arr_acc) - min(arr_acc) < tor_acc and len(arr_acc) == 4:
                    print("Not improve acc. Stopping training. Last Acc:", arr_acc)
                    break
                if epoch > 20:
                    update_interval = 2
                    tor_acc = 0.005
                elif epoch > 60:
                    update_interval = 1
                    # select the best result
                    if max(arr_acc) - min(arr_acc) < 0.01 and last_acc > max(arr_acc) - 0.00001:
                        print("Not improve acc. Stopping training. Last Acc:", arr_acc)
                        break

                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break
                print("Delta label:", delta_label)
            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                z, qbatch, xrecon = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.recon_loss(inputs, xrecon)
                loss = cluster_loss + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))
            ml_loss = 0.0
            if epoch % update_ml == 0:
                for ml_batch_idx in range(ml_num_batch):
                    px1 = X[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    px2 = X[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    # pbatch1 = p[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    # pbatch2 = p[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    loss = (ml_p * self.pairwise_loss(q1, q2, "ML") + self.recon_loss(inputs1, xr1) + self.recon_loss(
                        inputs2, xr2))
                    # 0.1 for mnist/reyuters, 1 for fashion, the parameters are tuned via grid search on validation set
                    ml_loss += loss.data
                    loss.backward()
                    optimizer.step()

            cl_loss = 0.0
            if epoch % update_cl == 0:
                for cl_batch_idx in range(cl_num_batch):
                    px1 = X[cl_ind1[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    px2 = X[cl_ind2[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    # pbatch1 = p[cl_ind1[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    # pbatch2 = p[cl_ind2[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    loss = cl_p * self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if ml_num_batch > 0 and cl_num_batch > 0:
                print("Pairwise Total:", round(float(ml_loss.cpu()), 2) + float(cl_loss.cpu()), "ML loss",
                      float(ml_loss.cpu()), "CL loss:", float(cl_loss.cpu()))
        return last_acc, last_nmi, final_epoch
