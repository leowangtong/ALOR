import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        self.lab2cname_file = os.path.join("./data/oxford_pets/oxford_pets_metrics-LAION400M.json")

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

        with open('./data/oxford_pets/ltrain.txt', 'r') as inpu:
            ltrains = inpu.readlines()
            inpu.close()
        for ltrain in ltrains:
            im = './data/oxford_pets/' + ltrain.split(' ')[0]
            label = ltrain.split(' ')[1]
            cname = lab2cname[label]['name']
            train.extend(_collate(im, int(label), cname))


        with open('./data/oxford_pets/val.txt', 'r') as inpu:
            vals = inpu.readlines()
            inpu.close()
        for lval in vals:
            im = './data/oxford_pets/' + lval.split(' ')[0]
            label = lval.split(' ')[1]
            cname = lab2cname[label]['name']
            val.extend(_collate(im, int(label), cname))

        with open('./data/oxford_pets/test.txt', 'r') as inpu:
            tests = inpu.readlines()
            inpu.close()
        for ltest in tests:
            im = './data/oxford_pets/' + ltest.split(' ')[0]
            label = ltest.split(' ')[1]
            cname = lab2cname[label]['name']
            test.extend(_collate(im, int(label), cname))

        with open('./data/oxford_pets/retrieved_data.txt', 'r') as inpu:
            retrievals = inpu.readlines()
            inpu.close()
        for lretrieval in retrievals:
            im = './data/oxford_pets/' + lretrieval.split(' ')[0]
            label = lretrieval.split(' ')[1]
            cname = lab2cname[label]['name']
            retrieval.extend(_collate(im, int(label), cname))

        return train, val, test, retrieval

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output
