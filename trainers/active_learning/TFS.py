import torch
import numpy as np
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from .AL import AL
from .entropy import Entropy


class TFS(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, dataset, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.dataset = dataset

    def select(self, n_query, **kwargs):
        all_dict = {}
        num_dict = {}
        for i in range(n_query):
            num_dict[str(i)] = 0
            all_dict[str(i)] = []
        for i in self.dataset._train_x:
            num_dict[str(i.label)] += 1
        unlabeled_set = torch.utils.data.Subset(self.unlabeled_dst, self.U_index)
        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=unlabeled_set,
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
                    all_dict[str(int(maxInds[j].cpu()))].append(self.U_index[i * self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE + j])
        len_alldict = 0
        for m in range(n_query):
            len_alldict += len(all_dict[str(m)])
        idx = []
        selector = Entropy(self.cfg, self.model, self.unlabeled_dst, self.U_index, n_query, self.device)
        np_entropy = selector.rank_uncertainty(self.unlabeled_dst)
        dict_entropy = {}
        for m in range(n_query):
            dict_entropy[str(m)] = np.array([])
            for n in all_dict[str(m)]:
                dict_entropy[str(m)] = np.append(dict_entropy[str(m)], np.array(np_entropy[n]))
        for m in range(n_query):
            bottom_100 = sorted(num_dict.items(), key=lambda item: item[1])
            for k in range(n_query):
                for j in range(len(all_dict[bottom_100[k][0]])):
                    x = dict_entropy[bottom_100[k][0]]
                    sampled_data = all_dict[bottom_100[k][0]][np.argsort(x)[j]]
                    if sampled_data in self.U_index and sampled_data not in idx:
                        idx.append(sampled_data)
                        num_dict[bottom_100[k][0]] += 1
                        break
                if len(idx) - 1 == m:
                    break
        return idx

