import os
import torch
from utils.models import set_model, set_classifier, MyLinear, save_test_scores, save_best_model  # , save_head_weights
import time
import numpy as np
from utils.parser import parse_args
from utils.logger import set_logger
from testing import validate, load_model
from testing import calculate_scores
from utils.datasets.dataset_utils import NUM_CLASSES_DICT
from utils.prompt import set_prompt
import copy
from utils.losses import set_loss
import torch.nn.functional as F
import random
import cv2
from utils.dataloader import get_ul_dataloader
from utils.training import set_training_seed, train_probing, run_zeroshot, train_ce, train_flyp
from utils.dataloader import extract_train_dataloader, extract_dataloader, set_dataloaders, set_text_dataloader
from utils.optimizers import set_optimizer, set_params
from active_learning.pcb import PCB
from active_learning.badge import BADGE
from active_learning.coreset import Coreset
from active_learning.entropy import Entropy
from active_learning.ALFA_mix import ALFAMix
from active_learning.LoGo import LoGo
from active_learning.TFS import TFS



def tuning(args, logger, model, preprocess, tokenized_text_prompts):
    # dataloaders
    train_loader, val_loader, test_loader, train_dataset = set_dataloaders(args, model, tokenized_text_prompts, preprocess, logger)
    text_dataloader = set_text_dataloader(args, logger, prompt_tensors,
                                          prompt_tensors_dict) if args.method == 'CMLP' else None
    test_loader_copy = copy.deepcopy(test_loader)

    loss = set_loss(args)
    params, logit_scale = set_params(args, model, classifier_head, logger)  # depend on method
    optimizer, scheduler, total_iter = set_optimizer(args, params, train_loader)

    args.loss = loss
    args.logit_scale = logit_scale
    args.optimizer = optimizer
    args.scheduler = scheduler

    # check zeroshot acc
    if args.check_zeroshot or args.method == 'zeroshot':
        logger.info(f"Check Zero-shot Acc ......")
        run_zeroshot(args, test_loader, model, logger, loss, logit_scale, classifier_head)
    if args.zeroshot_only or args.method == 'zeroshot':
        exit()

    # ---------- Training
    if args.method == 'probing':
        best_model, best_head, best_records, \
            best_logit_scale, val_loader, test_loader = train_probing(args, logger, loss_logger, model, classifier_head, \
                                                                      train_loader, val_loader, test_loader)

    elif args.method == 'finetune':
        best_model, best_head, \
            best_records, best_logit_scale = train_ce(args, logger, loss_logger, model, classifier_head, \
                                                      train_loader, val_loader, test_loader)

    elif args.method == 'FLYP':
        best_model, best_head, best_records, best_logit_scale = train_flyp(args, logger, loss_logger, model, tokenizer,
                                                                           train_loader, val_loader, test_loader,
                                                                           text_prompts)

    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")

    if args.method == 'dataset-cls':
        exit()

    # print the logit_scale
    logger.info(f"logit_scale: {round(logit_scale.item(), 8)}")
    logger.info(f"best_logit_scale: {round(best_logit_scale.item(), 8)}")

    return test_loader_copy, best_model, train_loader, best_logit_scale, best_head, train_dataset


if __name__ == '__main__':
    args = parse_args()
    logger, loss_logger = set_logger(args)
    set_training_seed(args)
    # load model
    model, preprocess, tokenizer = set_model(args, logger)
    zeroshot_model = copy.deepcopy(model)
    # make prompts
    prompt_tensors, text_prompts, \
        tokenized_text_prompts, prompt_tensors_dict = set_prompt(args, model, tokenizer, logger)
    # make classifier head
    classifier_head = set_classifier(args, prompt_tensors, logger)
    zeroshot_head = copy.deepcopy(classifier_head)
    classifier_head.to(args.device)
    
    test_loader, best_model, train_loader, best_logit_scale, best_head, train_dataset = tuning(args, logger, model, preprocess, tokenized_text_prompts)
    
    if args.ALMETHOD == "TFS":
        loss = set_loss(args)
        with open(f'./data/{args.dataset}/ltrain.txt', 'r') as f:
            all_ul = f.readlines()
        n_ul = len(all_ul)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_round{args.round}.txt', 'r') as f:
            labeled = f.readlines()
        for i in labeled:
            for j in all_ul:
                if j in i:
                    all_ul.remove(j)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt', 'w') as f:
            for i in all_ul:
                f.write(i)
        ul_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt'
        ul_split = [[ul_file], [os.path.join(args.dataset_path, args.dataset)]]
        output_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_round{int(args.round) + 1}.txt'
        ul_loader = get_ul_dataloader(args, ul_split, tokenized_text_prompts, preprocess)

        test_acc, _, test_acc_sk, test_macro_f1_sk, test_length, _, all_dict = validate(args, data_loader=ul_loader,
                                                                            model=best_model, logger=logger,
                                                                            loss=loss, logit_scale=best_logit_scale,
                                                                            classifier_head=best_head,
                                                                            dataset=args.dataset,
                                                                            output_dir=args.output_dir,
                                                                            device=args.device,
                                                                            )
        selector = TFS(best_model, ul_loader, best_logit_scale, best_head, len(prompt_tensors), labeled, all_dict, device='cuda')
        TFS_idx = selector.select(n_query=len(prompt_tensors))
        os.remove(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt')
        with open(f'./data/{args.dataset}/{output_file}', 'w') as f:
            for i in labeled:
                f.write(i)
            for i in TFS_idx:
                f.write(all_ul[i])

    elif args.ALMETHOD == "entropy":
        with open('./data/' + args.dataset + '/ltrain.txt', 'r') as f:
            all_ul = f.readlines()
        n_ul = len(all_ul)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_round{args.round}.txt', 'r') as f:
            labeled = f.readlines()
        for i in labeled:
            for j in all_ul:
                if j in i:
                    all_ul.remove(j)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt', 'w') as f:
            for i in all_ul:
                f.write(i)
        ul_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt'
        ul_split = [[ul_file], [os.path.join(args.dataset_path, args.dataset)]]
        output_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_round{int(args.round) + 1}.txt'
        ul_loader = get_ul_dataloader(args, ul_split, tokenized_text_prompts, preprocess)
        selector = Entropy(best_model, ul_loader, best_logit_scale, best_head, n_class=len(prompt_tensors),
                        device='cuda')
        Entropy_idx = selector.select(n_query=len(prompt_tensors))
        os.remove(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt')
        with open(f'./data/{args.dataset}/{output_file}', 'w') as f:
            for k in labeled:
                f.write(k)
            for k in Entropy_idx:
                f.write(all_ul[k])

    elif args.ALMETHOD == "badge_pcb":
        with open('./data/'+args.dataset+'/ltrain.txt', 'r') as f:
            all_ul = f.readlines()
        n_ul = len(all_ul)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_round{args.round}.txt', 'r') as f:
            labeled = f.readlines()
        for i in labeled:
            for j in all_ul:
                if j in i:
                    all_ul.remove(j)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt', 'w') as f:
            for i in all_ul:
                f.write(i)
        ul_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt'
        ul_split = [[ul_file], [os.path.join(args.dataset_path, args.dataset)]]
        output_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_round{int(args.round) + 1}.txt'
        ul_loader = get_ul_dataloader(args, ul_split, tokenized_text_prompts, preprocess)
        selector = BADGE(best_model, ul_loader, all_ul, best_logit_scale, best_head, args, n_class=len(prompt_tensors), device='cuda')
        BADGE_idx = selector.select(n_query=n_ul // 10)
        statistics = torch.zeros(len(prompt_tensors))
        for elem in train_dataset:
            statistics[elem[1]] += 1
        pcb_selector = PCB(best_model, ul_loader, BADGE_idx, n_class=len(prompt_tensors), statistics=statistics, device='cuda')
        BADGE_pcb_idx = pcb_selector.select(n_query=len(prompt_tensors))
        os.remove(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt')
        with open(f'./data/{args.dataset}/{output_file}', 'w') as f:
            for k in labeled:
                f.write(k)
            for k in BADGE_pcb_idx:
                f.write(all_ul[k])
    elif args.ALMETHOD == "badge":
        with open('./data/'+args.dataset+'/ltrain.txt', 'r') as f:
            all_ul = f.readlines()
        n_ul = len(all_ul)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_round{args.round}.txt', 'r') as f:
            labeled = f.readlines()
        for i in labeled:
            for j in all_ul:
                if j in i:
                    all_ul.remove(j)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt', 'w') as f:
            for i in all_ul:
                f.write(i)
        ul_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt'
        ul_split = [[ul_file], [os.path.join(args.dataset_path, args.dataset)]]
        output_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_round{int(args.round) + 1}.txt'
        ul_loader = get_ul_dataloader(args, ul_split, tokenized_text_prompts, preprocess)
        selector = BADGE(best_model, ul_loader, all_ul, best_logit_scale, best_head, args, n_class=len(prompt_tensors), device='cuda')
        BADGE_idx = selector.select(n_query=len(prompt_tensors))
        os.remove(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt')
        with open(f'./data/{args.dataset}/{output_file}', 'w') as f:
            for k in labeled:
                f.write(k)
            for k in BADGE_idx:
                f.write(all_ul[k])
    elif args.ALMETHOD == "coreset":
        with open('./data/' + args.dataset + '/ltrain.txt', 'r') as f:
            all_ul = f.readlines()
        n_ul = len(all_ul)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_round{args.round}.txt', 'r') as f:
            labeled = f.readlines()
        for i in labeled:
            for j in all_ul:
                if j in i:
                    all_ul.remove(j)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt', 'w') as f:
            for i in all_ul:
                f.write(i)
        ul_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt'
        ul_split = [[ul_file], [os.path.join(args.dataset_path, args.dataset)]]
        output_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_round{int(args.round) + 1}.txt'
        ul_loader = get_ul_dataloader(args, ul_split, tokenized_text_prompts, preprocess)
        val_x = train_loader
        selector = Coreset(best_model, ul_loader, val_x, n_class=len(prompt_tensors))
        Coreset_idx = selector.select(n_query=len(prompt_tensors))
        os.remove(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt')
        with open(f'./data/{args.dataset}/{output_file}', 'w') as f:
            for k in labeled:
                f.write(k)
            for k in Coreset_idx:
                f.write(all_ul[k])

    elif args.ALMETHOD == "logo":
        with open('./data/'+args.dataset+'/ltrain.txt', 'r') as f:
            all_ul = f.readlines()
        n_ul = len(all_ul)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_round{args.round}.txt', 'r') as f:
            labeled = f.readlines()
        for i in labeled:
            for j in all_ul:
                if j in i:
                    all_ul.remove(j)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt', 'w') as f:
            for i in all_ul:
                f.write(i)
        ul_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt'
        ul_split = [[ul_file], [os.path.join(args.dataset_path, args.dataset)]]
        output_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_round{int(args.round) + 1}.txt'
        ul_loader = get_ul_dataloader(args, ul_split, tokenized_text_prompts, preprocess)
        val_x = train_loader
        selector = LoGo(best_model, ul_loader, all_ul, best_logit_scale, best_head, args, n_class=len(prompt_tensors), device='cuda')
        logo_idx = selector.query(len(prompt_tensors))
        os.remove(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt')
        with open(f'./data/{args.dataset}/{output_file}', 'w') as f:
            for k in labeled:
                f.write(k)
            for k in logo_idx:
                f.write(all_ul[k])

    elif args.ALMETHOD == "alfa_mix":
        with open('./data/'+args.dataset+'/ltrain.txt', 'r') as f:
            all_ul = f.readlines()
        n_ul = len(all_ul)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_round{args.round}.txt', 'r') as f:
            labeled = f.readlines()
        for i in labeled:
            for j in all_ul:
                if j in i:
                    all_ul.remove(j)
        with open(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt', 'w') as f:
            for i in all_ul:
                f.write(i)
        ul_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt'
        ul_split = [[ul_file], [os.path.join(args.dataset_path, args.dataset)]]
        output_file = f'{args.ALMETHOD}_{args.method}_seed{args.seed}_round{int(args.round) + 1}.txt'
        ul_loader = get_ul_dataloader(args, ul_split, tokenized_text_prompts, preprocess)
        val_x = train_loader
        selector = ALFAMix(best_model, ul_loader, all_ul, train_loader, train_dataset, best_logit_scale, best_head,
                        args, n_class=len(prompt_tensors), device='cuda')
        ALFAMix_idx = selector.query(len(prompt_tensors))
        os.remove(f'./data/{args.dataset}/{args.ALMETHOD}_{args.method}_seed{args.seed}_after_round{args.round}.txt')
        with open(f'./data/{args.dataset}/{output_file}', 'w') as f:
            for k in labeled:
                f.write(k)
            for k in ALFAMix_idx:
                f.write(all_ul[k])

