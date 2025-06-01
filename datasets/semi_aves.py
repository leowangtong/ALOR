import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from datasets.oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class semi_aves(DatasetBase):
    dataset_dir = "semi_aves"

    def __init__(self, cfg):
        self.lab2cname_file = os.path.join("./data/semi_aves/semi_aves_metrics-LAION400M.json")

        train, val, test, retrieval = self.read_data()

        num_shots = cfg.DATASET.NUM_SHOTS

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test, retrieval = OxfordPets.subsample_classes(train, val, test, retrieval, subsample=subsample)

        super().__init__(train_x=train, retrieval=retrieval, val=val, test=test)

    def read_data(self):
        def _collate(im, y, c):
            items = []
            item = Datum(impath=im, label=y, classname=c)  # convert to 0-based label
            items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test, retrieval = [], [], [], []

        with open('./data/semi_aves/ltrain.txt', 'r') as inpu:
            ltrains = inpu.readlines()
            inpu.close()
        for ltrain in ltrains:
            im = './data/semi_aves/' + ltrain.split(' ')[0]
            label = ltrain.split(' ')[1]
            cname = lab2cname[label]['name']
            train.extend(_collate(im, int(label), cname))


        with open('./data/semi_aves/val.txt', 'r') as inpu:
            vals = inpu.readlines()
            inpu.close()
        for lval in vals:
            im = './data/semi_aves/' + lval.split(' ')[0]
            label = lval.split(' ')[1]
            cname = lab2cname[label]['name']
            val.extend(_collate(im, int(label), cname))

        with open('./data/semi_aves/test.txt', 'r') as inpu:
            tests = inpu.readlines()
            inpu.close()
        for ltest in tests:
            im = './data/semi_aves/' + ltest.split(' ')[0]
            label = ltest.split(' ')[1]
            cname = lab2cname[label]['name']
            test.extend(_collate(im, int(label), cname))

        with open('./data/semi_aves/retrieved_data.txt', 'r') as inpu:
            retrievals = inpu.readlines()
            inpu.close()
        for lretrieval in retrievals:
            im = './data/semi_aves/' + lretrieval.split(' ')[0]
            label = lretrieval.split(' ')[1]
            cname = lab2cname[label]['name']
            retrieval.extend(_collate(im, int(label), cname))

        return train, val, test, retrieval
