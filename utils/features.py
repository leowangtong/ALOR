import os
import clip
import json
import torch
import random
import argparse
import time
import copy
import pickle
from tqdm import tqdm

def prompt_sampler(prompt_tensors, sample_by='mean'):
    
    sampled_prompts = [] 
    for i in prompt_tensors.keys():
        if sample_by == 'mean':
            sampled_prompts.append(prompt_tensors[i]['mean'])
        elif sample_by == 'random':
            sampled_prompts.append(random.choice(prompt_tensors[i]['all']))
        else:
            raise NotImplementedError

    return torch.stack(sampled_prompts, dim=0)



def operate_on_prompt(model, text, operation, tokenize):

    if operation=='encode':
        features = model.encode_text(tokenize(text).cuda())
        features /= features.norm(dim=-1, keepdim=True) # Normalization. +++++
        return features # this is text embedding
    
    elif operation == 'tokenize':
        tokens = tokenize(text)
        return tokens # this is tokenized text

# Pre-extract all features and pass to data-loader.
def get_text_features(model, prompt_dict, tokenize, operation='encode'):

    tensor_dict = {}
    model.eval()
    with torch.no_grad():
        for key, info in prompt_dict.items():
            # key is the class_id
            source = {}
            prompts = []
            for prompt in info['corpus']:
                prompts.append(prompt)

            stacked_tensor = operate_on_prompt(model, prompts, operation, tokenize)
            stacked_tensor.cpu()
            
            source['all'] = stacked_tensor

            # also compute the mean tensor if operation is encode
            if operation == 'encode':
                mean_tensor = torch.mean(stacked_tensor, dim=0)
                mean_tensor /= mean_tensor.norm(dim=-1, keepdim=True) 
                source['mean'] = mean_tensor

            tensor_dict[key] = source

    return tensor_dict

def extract_test_feats(model, dataloader):

    img_feats_lst, labels_lst = [], []

    # for data in tqdm(dataloader):
    for data in dataloader:
        imgs, labels, text, source = data
        imgs = imgs.cuda()
        labels = labels.long()

        model.eval()
        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True) # Normalization.

        img_feats_lst.append(img_feats.cpu())
        labels_lst.append(labels.cpu())

    img_feats_store = torch.cat(img_feats_lst, dim=0)
    labels_store = torch.cat(labels_lst, dim=0)

    result = {'image_features': img_feats_store, 
                'labels': labels_store}
    
    return result
        
            
    