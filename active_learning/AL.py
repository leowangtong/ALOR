import torch

class AL(object):
    def __init__(self, model, unlabeled_dst, n_class, **kwargs):
        self.unlabeled_dst = unlabeled_dst
        self.n_class = n_class
        self.model = model
        self.index = []

    def select(self, **kwargs):
        return