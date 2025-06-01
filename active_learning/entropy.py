import torch
import numpy as np
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from .AL import AL

class Entropy(AL):
    def __init__(self, model, unlabeled_dst, logit_scale, best_head, n_class, device, **kwargs):
        super().__init__(model, unlabeled_dst, n_class, **kwargs)
        self.device = device
        self.logit_scale = logit_scale
        self.classifier_head = best_head
        
    def run(self, n_query):
        scores = self.rank_uncertainty()
        selection_result = np.argsort(scores)[:n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        self.model.eval()
        with torch.no_grad():
            scores = np.array([])
            
            print("| Calculating uncertainty of Unlabeled set")
            for inputs, labels, tokenized_text, source in self.unlabeled_dst:
                inputs = inputs.to(self.device)
                
                features = self.model.encode_image(inputs)
                preds = self.classifier_head(features) * self.logit_scale.exp()
                preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                scores = np.append(scores, entropys)

        return scores

    def select(self, n_query, **kwargs):
        selected_indices, scores = self.run(n_query)
        Q_index = [idx for idx in selected_indices]
        return Q_index
