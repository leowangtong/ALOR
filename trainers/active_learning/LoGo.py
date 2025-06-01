import copy
import math
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from .AL import AL


class LoGo(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.pred = []

    def get_grad_features(self):
        self.pred = []
        self.model.eval()
        nLab = self.n_class
        if 'rn50' in self.cfg.MODEL.BACKBONE.NAME:
            embDim = 1024  # rn50
        else:
            embDim = 512  # rn101/vitb16/vitb32
        embedding = torch.zeros([len(self.U_index), embDim])

        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            for i, batch in enumerate(unlabeled_loader):
                inputs = batch["img"].to(self.device)
                out, features = self.model(inputs, get_feature=True)
                batchProbs = torch.nn.functional.softmax(out, dim=1).data
                maxInds = torch.argmax(batchProbs, 1)

                for j in range(len(inputs)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[i * self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE + j] = features[j].clone() * (
                                        1 - batchProbs[j][c])

        return torch.Tensor(embedding)

    def predict_prob(self, unlabel_idxs):
        self.model.eval()
        probs = torch.zeros([len(unlabel_idxs), self.n_class])
        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            for i, batch in enumerate(unlabeled_loader):
                inputs = batch["img"].to(self.device)
                out, features = self.model(inputs, get_feature=True)
                for j in range(len(inputs)):
                    for a in range(len(unlabel_idxs)):
                        if i * self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE + j == unlabel_idxs[a]:
                            probs[a] = torch.nn.functional.softmax(out, dim=1)[j].cpu().data
        return probs

    def query(self, n_query):
        unlabel_idxs = np.array(self.U_index)

        # cluster with uncertain samples by local net
        embedding = self.get_grad_features()

        print("Macro Step: K-Means EM algorithm with local-only model")
        kmeans = KMeans(n_clusters=n_query)
        kmeans.fit(embedding)
        cluster_pred = kmeans.labels_

        cluster_dict = {i: [] for i in range(n_query)}
        for u_idx, c in zip(unlabel_idxs, cluster_pred):
            cluster_dict[c].append(u_idx)

        print("Micro Step: 1 step of EM algorithm with global model")
        # query with uncertain samples by global net via predefined cluster
        query_idx = []
        for c_i in cluster_dict.keys():
            cluster_idxs = np.array(cluster_dict[c_i])

            probs = self.predict_prob(cluster_idxs)
            log_probs = torch.log(probs)

            # inf to zero
            log_probs[log_probs == float('-inf')] = 0
            log_probs[log_probs == float('inf')] = 0

            U = (probs * log_probs).sum(1)
            U = U.numpy()

            try:
                chosen = np.argsort(U)[0]
                query_idx.append(cluster_idxs[chosen])
            except:
                # IndexError: index 0 is out of bounds for axis 0 with size 0 with ConvergenceWarning
                continue

        query_idx = list(set(query_idx))

        # sometimes k-means clustering output smaller amount of centroids due to convergence errors
        if len(query_idx) != n_query:
            print('cluster centroids number is different from the number of query budget')

            num = math.ceil((n_query - len(query_idx)) / len(np.unique(cluster_pred)))
            idx, skip = 0, []

            query_idx = set(query_idx)
            U_dict = {c_i: None for c_i in cluster_dict.keys()}
            while len(query_idx) < n_query:
                for c_i in cluster_dict.keys():
                    if c_i in skip: continue

                    cluster_idxs = np.array(cluster_dict[c_i])

                    if len(cluster_idxs) < idx + 1:
                        skip.append(c_i)
                    else:
                        if U_dict[c_i] is None:
                            # store uncertainty
                            probs = self.predict_prob(cluster_idxs)
                            log_probs = torch.log(probs)

                            log_probs[log_probs == float('-inf')] = 0
                            log_probs[log_probs == float('inf')] = 0

                            U = (probs * log_probs).sum(1)
                            U = U.numpy()
                            U_dict[c_i] = deepcopy(U)
                        else:
                            U = U_dict[c_i]

                        chosen = np.argsort(U)[idx + 1:idx + 1 + num]
                        try:
                            query_idx = query_idx.union(set(cluster_idxs[chosen]))
                        except TypeError:
                            query_idx = query_idx.union(set([cluster_idxs[chosen]]))
                idx += num

            query_idx = list(query_idx)[:n_query]
        return query_idx
