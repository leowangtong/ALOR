import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing,read_json

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class Food101(DatasetBase):

    dataset_dir = "Food101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.lab2cname_file = os.path.join("./data/food101/food101_metrics-LAION400M.json")

        train, val, test, retrieval = self.read_data()
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

        with open('./data/food101/ltrain.txt', 'r') as inpu:
            ltrains = inpu.readlines()
            inpu.close()
        for ltrain in ltrains:
            im = './data/food101/' + ltrain.split(' ')[0]
            label = ltrain.split(' ')[1]
            cname = lab2cname[label]['name']
            train.extend(_collate(im, int(label), cname))


        with open('./data/food101/val.txt', 'r') as inpu:
            vals = inpu.readlines()
            inpu.close()
        for lval in vals:
            im = './data/food101/' + lval.split(' ')[0]
            label = lval.split(' ')[1]
            cname = lab2cname[label]['name']
            val.extend(_collate(im, int(label), cname))

        with open('./data/food101/test.txt', 'r') as inpu:
            tests = inpu.readlines()
            inpu.close()
        for ltest in tests:
            im = './data/food101/' + ltest.split(' ')[0]
            label = ltest.split(' ')[1]
            cname = lab2cname[label]['name']
            test.extend(_collate(im, int(label), cname))

        with open('./data/food101/retrieved_data.txt', 'r') as inpu:
            retrievals = inpu.readlines()
            inpu.close()
        for lretrieval in retrievals:
            im = './data/food101/' + lretrieval.split(' ')[0]
            label = lretrieval.split(' ')[1]
            cname = lab2cname[label]['name']
            retrieval.extend(_collate(im, int(label), cname))

        return train, val, test, retrieval
