import torch
import numpy as np
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from .AL import AL
from .entropy import Entropy


class TFS(AL):
    def __init__(self, model, unlabeled_dst, logit_scale, best_head, n_class, labeled, all_dict, device, **kwargs):
        super().__init__(model, unlabeled_dst, n_class, **kwargs)
        self.device = device
        self.labeled = labeled
        self.logit_scale = logit_scale
        self.classifier_head = best_head
        self.model = model
        self.all_dict = all_dict

    def select(self, n_query, **kwargs):
        num_dict = {}
        for i in range(n_query):
            num_dict[str(i)] = 0
        for i in self.labeled:
            num_dict[i.split(' ')[1]] += 1
        next_round = []
        selector = Entropy(self.model, self.unlabeled_dst, self.logit_scale, self.classifier_head, self.n_class, device='cuda')
        np_entropy = selector.rank_uncertainty()
        dict_entropy = {}
        for i in range(n_query):
            dict_entropy[str(i)] = np.array([])
            for j in self.all_dict[str(i)]:
                dict_entropy[str(i)] = np.append(dict_entropy[str(i)], np.array(np_entropy[j]))
        for i in range(n_query):
            ranking = sorted(num_dict.items(), key=lambda item: item[1])
            for k in range(n_query):
                for j in range(len(self.all_dict[ranking[k][0]])):
                    x = dict_entropy[ranking[k][0]]
                    sampled_data = self.all_dict[ranking[k][0]][np.argsort(x)[j]]
                    if sampled_data not in next_round:
                        next_round.append(sampled_data)
                        num_dict[ranking[k][0]] += 1
                        break
                if len(next_round) - 1 == i:
                    break
        return next_round

