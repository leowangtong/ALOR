import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from .AL import AL


class BADGE(AL):
    def __init__(self, model, unlabeled_dst, ul, logit_scale, best_head, args, n_class, device, **kwargs):
        super().__init__(model, unlabeled_dst, n_class, **kwargs)
        self.device = device 
        self.pred = []
        self.ul = ul
        self.logit_scale = logit_scale
        self.classifier_head = best_head
        self.args = args

    def get_grad_features(self):
        self.pred = []
        self.model.eval()
        if 'rn50' in self.args.model_cfg:
            embDim = 1024 #rn50
        else:
            embDim = 512 # rn101/vitb16/vitb32
        num_unlabeled = len(self.ul)
        grad_embeddings = torch.zeros([num_unlabeled, embDim * self.n_class])
        with torch.no_grad():
            for i, data in enumerate(self.unlabeled_dst):
                inputs = data[0].to(self.device)
                features = self.model.encode_image(inputs)
                out = self.classifier_head(features) * self.logit_scale.exp()

                batchProbs = torch.nn.functional.softmax(out, dim=1).data
                maxInds = torch.argmax(batchProbs, 1)
                # _, preds = torch.max(out.data, 1)
                self.pred.append(maxInds.detach().cpu())

                for j in range(len(inputs)):
                    for c in range(self.n_class):
                        if c == maxInds[j]:
                            grad_embeddings[i * 128 + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                        1 - batchProbs[j][c])
                        else:
                            grad_embeddings[i * 128 + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                        -1 * batchProbs[j][c])
        return grad_embeddings.cpu().numpy()

    # kmeans ++ initialization
    def k_means_plus_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if len(mu) % 100 == 0:
                print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def select(self, n_query, **kwargs):
        unlabeled_features = self.get_grad_features()
        selected_indices = self.k_means_plus_centers(X=unlabeled_features, K=n_query)
        scores = list(np.ones(len(selected_indices))) # equally assign 1 (meaningless)

        Q_index = [idx for idx in selected_indices]
        return Q_index
    
    def select_by_filter(self, n_query, **kwargs):
        unlabeled_features = self.get_grad_features()
        self.pred = torch.cat(self.pred)
        
        pred_idx = []
        ret_idx = []
        Q_index = self.k_means_plus_centers(X=unlabeled_features, K=10*n_query)
        
        
        for q in Q_index:
            if int(self.pred[q]) not in pred_idx:
                pred_idx.append(int(self.pred[q]))
                ret_idx.append(q)
                
        if len(pred_idx) == self.n_class:
            ret_idx = [idx for idx in ret_idx]
            print(f"pred idx(all the classes): {pred_idx}")
            return ret_idx, None
        
        print("Fail to get all the classes!!!")
        for q in Q_index:
            if len(ret_idx) == self.n_class:
                ret_idx = [idx for idx in ret_idx]
                print(f"pred idx: {pred_idx}")
                return ret_idx, None
            
            if q not in ret_idx:
                pred_idx.append(int(self.pred[q]))
                ret_idx.append(q)
                
        raise EnvironmentError
                
                