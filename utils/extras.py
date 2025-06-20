import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip, RandomCrop
from PIL import Image
import open_clip, clip
import argparse
import json
import os
from torchvision import transforms

from .randaugment import RandAugmentMC


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TransformFixMatch(object):
    def __init__(self, n_px , mode='train'):
        self.weak = transforms.Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                              interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            _convert_image_to_rgb])
        
        self.strong = transforms.Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                              interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            _convert_image_to_rgb,
            RandAugmentMC(n=2, m=10)]) # add RandAugmentMC here for strong augmentation !!!
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


def transform(n_px , mode='train'):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if mode == 'train':
        return Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                              interpolation=Image.BICUBIC),
            # CenterCrop(n_px), # change to Center Crop
            RandomHorizontalFlip(),
            _convert_image_to_rgb,
            ToTensor(),
            normalize
        ])

OpenCLIP_model_dic_LAION400M = {
    'ViT-B/32': ('ViT-B-32-quickgelu', 'laion400m_e32'),
    'ViT-B/16': ('ViT-B-16', 'laion400m_e32'),
    'ViT-L/14': ('ViT-L-14', 'laion400m_e32')
}

OpenCLIP_model_dic_LAION2B = {
    'ViT-B/32': ('ViT-B-32', 'laion2b_e16'),
}

OPENCLIP_MODEL_DIC = {
    'laion400m': {
        'vitb32': ('laion400m_e32','ViT-B-32-quickgelu'),
        'vitb16': ('laion400m_e32','ViT-B-16'),
        'vitl14': ('laion400m_e32','ViT-L-14'),
    },
    'openai': {
        'vitb32': ('openai','ViT-B-32-quickgelu'),
        'vitb16': ('openai','ViT-B-16'),
        'vitl14': ('openai','ViT-L-14'),
        'rn50': ('openai','RN50')
    },
    'laion2b': {
        'vitb32': ('laion2b_s34b_b79k','ViT-B-32'),
        'vitb16': ('laion2b_s34b_b88k','ViT-B-16'),
        'vitl14': ('laion2b_s32b_b82k','ViT-L-14')
    }
}

CLIP_MODEL_DIC ={
    'vitb32': 'ViT-B/32',
    'vitb16': 'ViT-B/16',
    'rn50': 'RN50',
    'rn101': 'RN101',
}



def get_engine(model_cfg, device='cuda', mode='val'):

    cfgs = model_cfg.split('_')
    arch = cfgs[0]
    model_name = cfgs[1]
    pretraining_dataset = cfgs[2] if len(cfgs) == 3 else None
    
    if model_name == 'clip':
        arch = CLIP_MODEL_DIC[arch]
        model, preprocess = clip.load(arch, device)
        tokenizer = clip.tokenize
        # get the train preprocess for CLIP
        train_preprocess = preprocess

    elif model_name == 'openclip':
        corpus_config, model_arch = OPENCLIP_MODEL_DIC[pretraining_dataset][arch]
        model, train_preprocess, preprocess = open_clip.create_model_and_transforms(model_arch, pretrained=corpus_config)
        tokenizer = open_clip.get_tokenizer(model_arch)
    
    else:
        raise NotImplementedError

    # not using mixed precision
    model.float()
    model.to(device)

    if mode == 'train':
        return model, train_preprocess, preprocess, tokenizer
    elif mode == 'val':
        return model, preprocess, tokenizer
    else:
        raise NotImplementedError


def get_worstk_class(score, confusion_matrix, N=30, fname=None):

    confusing_pairs = {}

    # id to scname mapping
    with open('data/semi-aves/id_scname_dict.json', 'r') as f:
        id_scname_dict = json.load(f)

    # get top-k confusing classes of the confusion matrix for each class
    topk = 20
    topk_classes = {}
    for classid, row in enumerate(confusion_matrix):
        # each item in the dict is a tuple (str(classid), str(count))
        topk_classes[str(classid)] = sorted([(i, row[i]) for i in range(len(row))], key=lambda x: x[1], reverse=True)[:topk]

    for k, v in topk_classes.items():
        info = dict()
        for classid, count in v:
            info[classid] = (id_scname_dict[str(classid)], str(count))
        topk_classes[k] = info

    # get the target dict
    target_dict = {}
    for classid, row in enumerate(confusion_matrix):
        target_dict[str(classid)] = dict()
        target_dict[str(classid)]['sname'] = id_scname_dict[str(classid)][0]
        target_dict[str(classid)]['cname'] = id_scname_dict[str(classid)][1]
        target_dict[str(classid)]['zs_acc'] = score['per_class_recall'][classid]*100
        target_dict[str(classid)]['topk_confusion'] = topk_classes[str(classid)]
        target_dict[str(classid)]['freq'] = 10

    # sort the target dict by the increasing zs_acc
    target_dict = sorted(target_dict.items(), key=lambda x: x[1]['zs_acc'])

    C=15
    for k, v in target_dict[:N]: # here worst N classes
        classid = k
        cname = v['cname']
        if classid not in confusing_pairs:
            confusing_pairs[classid] = []
        # print('its top k confusions:')
        ct = 0
        max_conf_count = 0
        for kk, vv in v['topk_confusion'].items():
            conf_id = kk
            conf_cname = vv[0][1]
            conf_count = int(vv[1])
            # collecting the confusing pairs, excluding the self-confusion, including the tiers
            if int(conf_id) == int(classid):
                pass
            elif conf_count >= max_conf_count:
                max_conf_count = conf_count
                confusing_pairs[classid].append(conf_id) # this is a list
            else:
                pass            

            ct += 1
            if ct == C:
                break

    return confusing_pairs


# single-match precision based on classid
def cal_single_precision(true, obtained):

    obtained_keys = set(obtained.keys())
    true_keys = set(true.keys())

    # count how many true keys are in the obtained keys
    intersection = obtained_keys.intersection(true_keys)

    precision = len(intersection) / len(obtained_keys)
    return precision


def cal_pair_precision(true, obtained):

    # calculate the candidate pair precision
    obtained_keys = set(obtained.keys())
    true_keys = set(true.keys())

    # count how many true keys are in the obtained keys
    intersection = obtained_keys.intersection(true_keys)
    if len(intersection) == 0:
        return 0, 0

    pair_match_ct = 0
    for k in intersection:
        candidates = set(obtained[k])
        target = set(true[k])

        if len(candidates.intersection(target)) > 0:
            pair_match_ct += 1

    rel_precision = pair_match_ct / len(intersection)

    abs_precision = pair_match_ct / len(obtained_keys)

    return rel_precision, abs_precision

def get_class_num_list(test_file):
    
    with open(test_file, 'r') as f:
        lines = f.readlines()
    
    class_id_set =set()
    for line in lines:
        entries = line.strip('\n').split()
        classid = int(entries[1])
        class_id_set.add(classid)
    
    cls_num_list = [0] * len(class_id_set)
    for line in lines:
        entries = line.strip('\n').split()
        classid = int(entries[1])
        cls_num_list[classid] += 1

    return cls_num_list
