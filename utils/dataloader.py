from .datasets.dataset_utils import load_dataset, TensorDataset, MyUnlabeledDataset, TextTensorDataset
from torch.utils.data import DataLoader
from .features import extract_test_feats
import torch
import os
from .extras import transform
import numpy as np
from .extras import TransformFixMatch


def extract_dataloader(args, best_model, split, fea_path, preprocess, tokenized_text_prompts, bsz=128):

    # extract features using the best model
    dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=split,
                                preprocess=preprocess,
                                tokenized_text_prompts=tokenized_text_prompts,
                                )
    dataloader = DataLoader(dataset, batch_size=bsz, 
                            shuffle=False, num_workers=args.num_workers, drop_last=False)

    features = extract_test_feats(best_model, dataloader=dataloader)
    torch.save(features, fea_path) 

    dataset = TensorDataset(pre_extracted_path=fea_path, device=args.device)

    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, drop_last=False, num_workers=0) 

    return dataloader



def pre_extract_feature(args, logger, model, tokenized_text_prompts, preprocess):

    pre_extract_train_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_train_features.pth'
    pre_extract_val_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_val_features.pth'
    pre_extract_test_fea_path = f'{args.dataset_root}/pre_extracted/{args.dataset}_{args.model_cfg}_{args.seed}_test_features.pth'

    if args.recal_fea or not os.path.exists(pre_extract_train_fea_path):
        train_dataset = load_dataset(dataset_root=args.dataset_root, 
                                    split=args.train_split, 
                                    preprocess=transform(224, 'train'),
                                    tokenized_text_prompts=tokenized_text_prompts,
                                    pl_list=None)
        train_loader = DataLoader(train_dataset, batch_size=128, 
                                    shuffle=False, num_workers=args.num_workers, drop_last=False)

        train_features = extract_test_feats(model, dataloader=train_loader)
        torch.save(train_features, pre_extract_train_fea_path)
        logger.info(f'Extracted train features to {pre_extract_train_fea_path}')

    if args.recal_fea or not os.path.exists(pre_extract_val_fea_path):
        val_dataset = load_dataset(dataset_root=args.dataset_root, 
                                    split=args.val_split,                                    
                                    preprocess=preprocess,
                                    tokenized_text_prompts=tokenized_text_prompts,
                                    pl_list=None)
        val_loader = DataLoader(val_dataset, batch_size=128, 
                                    shuffle=False, num_workers=args.num_workers, drop_last=False)

        val_features = extract_test_feats(model, dataloader=val_loader)
        torch.save(val_features, pre_extract_val_fea_path)
        logger.info(f'Extracted val features to {pre_extract_val_fea_path}')
    
    if args.recal_fea or not os.path.exists(pre_extract_test_fea_path):
        test_dataset = load_dataset(dataset_root=args.dataset_root, 
                                    split=args.test_split,
                                    preprocess=preprocess,
                                    tokenized_text_prompts=tokenized_text_prompts,
                                    pl_list=None)
        test_loader = DataLoader(test_dataset, batch_size=128, 
                                    shuffle=False, num_workers=args.num_workers, drop_last=False)

        test_features = extract_test_feats(model, dataloader=test_loader)
        torch.save(test_features, pre_extract_test_fea_path)
        logger.info(f'Extracted test features to {pre_extract_test_fea_path}')
    
    return pre_extract_train_fea_path, pre_extract_val_fea_path, pre_extract_test_fea_path


def get_dataloader_preextracted(args, logger, pre_extract_train_fea_path, pre_extract_val_fea_path, 
                                pre_extract_test_fea_path, device):

    train_dataset = TensorDataset(pre_extracted_path=pre_extract_train_fea_path, device=device)
    logger.info(f'Loaded pre-extracted train features from: {pre_extract_train_fea_path}')
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, drop_last=True, num_workers=0)

    val_dataset = TensorDataset(pre_extracted_path=pre_extract_val_fea_path, device=device)
    logger.info(f'Loaded pre-extracted val features from: {pre_extract_val_fea_path}')
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)

    test_dataset = TensorDataset(pre_extracted_path=pre_extract_test_fea_path, device=device)
    logger.info(f'Loaded pre-extracted test features from: {pre_extract_test_fea_path}')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)    

    return train_loader, val_loader, test_loader


def get_dataloader(args, train_split, val_split, test_split, tokenized_text_prompts, preprocess, utrain_labels=None):

    train_dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=train_split,                                                                                                       
                                preprocess=transform(224, 'train'),
                                tokenized_text_prompts=tokenized_text_prompts,
                                )

    val_dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=val_split, 
                                preprocess=preprocess,
                                tokenized_text_prompts=tokenized_text_prompts,
                                )

    test_dataset = load_dataset(dataset_root=args.dataset_root, 
                                split=test_split, preprocess=preprocess,
                                tokenized_text_prompts=tokenized_text_prompts,
                                )        

    train_loader = DataLoader(train_dataset, batch_size=128, pin_memory=True,
                            shuffle=True, drop_last=False, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    test_loader = DataLoader(test_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader, train_dataset


def get_ul_dataloader(args, ul_split, tokenized_text_prompts, preprocess):

    ul_dataset = load_dataset(dataset_root=args.dataset_root,
                                split=ul_split, preprocess=preprocess,
                                tokenized_text_prompts=tokenized_text_prompts)

    ul_loader = DataLoader(ul_dataset, batch_size=128, drop_last=False, pin_memory=True,
                            shuffle=False, num_workers=args.num_workers)

    return ul_loader
    
    
def get_retrieve_fewshot_dataloader(args, retrieve_data, fewshot_data, tokenized_text_prompts, preprocess, utrain_labels=None):

    train_dataset_retr = load_dataset(dataset_root=args.dataset_root, 
                                split=retrieve_data,                                                                                                       
                                preprocess=transform(224, 'train'),
                                tokenized_text_prompts=tokenized_text_prompts,
                                pl_list=utrain_labels,
                                )

    train_dataset_fs = load_dataset(dataset_root=args.dataset_root, 
                                split=fewshot_data,                                                                                                       
                                preprocess=transform(224, 'train'),
                                tokenized_text_prompts=tokenized_text_prompts,
                                pl_list=utrain_labels,
                                )

    train_dataloader_retr = DataLoader(train_dataset_retr, batch_size=args.bsz, 
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    train_dataloader_fewshot = DataLoader(train_dataset_fs, batch_size=args.bsz, 
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    return train_dataloader_retr, train_dataloader_fewshot


def get_unlabeled_dataloader(args, unlabeled_split):

    u_train_dataset = MyUnlabeledDataset(dataset_root=args.dataset_root,
                                        split=unlabeled_split, 
                                        transform=TransformFixMatch(224, 'train'),
                                        )

    u_train_dataloader = DataLoader(u_train_dataset, batch_size=args.bsz*args.mu, 
                            shuffle=True, drop_last=True, num_workers=args.num_workers)

    return u_train_dataloader


def set_dataloaders(args, model, tokenized_text_prompts, preprocess, logger):    

    # pre-extracted features
    if args.pre_extracted:
        train_fea_path, val_fea_path, test_fea_path = pre_extract_feature(args, logger, model, tokenized_text_prompts, preprocess)
    
    # dataset
    utrain_labels = None

    if args.pre_extracted:
        train_loader, val_loader, test_loader = get_dataloader_preextracted(args, logger, train_fea_path, 
                                                                                val_fea_path, 
                                                                                test_fea_path, args.device)
    else:
        train_loader, val_loader, test_loader, train_dataset = get_dataloader(args, args.train_split, args.val_split, args.test_split,
                                                                    tokenized_text_prompts, preprocess, utrain_labels)

    logger.info(f'len(train_loader): {len(train_loader)}')
    logger.info(f'len(val_loader): {len(val_loader)}')
    logger.info(f'len(test_loader): {len(test_loader)}')

    return train_loader, val_loader, test_loader, train_dataset


def set_text_dataloader(args, logger, prompt_tensors, prompt_tensors_dict):

    logger.info(f'Cross-modal adaptation: train with {args.prompt_name} prompts.')
    if args.use_attribute:
        logger.info(f'Use attribute when making prompts.')
        text_dataloader = get_text_dataloader(args, prompt_tensors_dict['c-name_attribute'], args.device)
    else:
        text_dataloader = get_text_dataloader(args, prompt_tensors, args.device)

    return text_dataloader


def get_text_dataloader(args, prompt_tensors, device):

    text_dataset = TextTensorDataset(prompt_tensors, device) 
    text_dataloader = DataLoader(text_dataset, batch_size=args.bsz, shuffle=True, 
                                num_workers=0, drop_last=True)
    return text_dataloader
    


def extract_train_dataloader(args, best_model, split, fea_path, preprocess, tokenized_text_prompts, bsz=128):

    # extract features using the best model
    dataset = load_dataset(dataset_root=args.dataset_root, 
                            split=split,
                            preprocess=preprocess,
                            tokenized_text_prompts=tokenized_text_prompts,
                            pl_list=None)
    dataloader = DataLoader(dataset, batch_size=bsz, pin_memory=True,
                            shuffle=True, num_workers=args.num_workers, drop_last=True)

    features = extract_test_feats(best_model, dataloader=dataloader)
    torch.save(features, fea_path) 

    dataset = TensorDataset(pre_extracted_path=fea_path, device=args.device)
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=True, drop_last=True, num_workers=0) 

    return dataloader
