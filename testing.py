import os
import torch
import time
import argparse
from utils.logger import get_logger
from utils.extras import get_engine
from torch.utils.data import DataLoader
from utils.models import MyLinear
from utils import features
from torchmetrics import ConfusionMatrix
# from torchmetrics import Accuracy
import numpy as np
import pickle
from time import time
import json
import random
from utils.datasets.dataset_utils import NUM_CLASSES_DICT, load_dataset, TensorDataset
from utils.prompt import prompt_maker
from utils.features import extract_test_feats
from sklearn.metrics import accuracy_score, f1_score


def load_model(args, logger, model, test_loader=None, classifier_head=None):
    logger.info(f'Loading model from: {args.model_path}')
    ckpt = torch.load(args.model_path)

    if 'clip' in ckpt:
        model.load_state_dict(ckpt['clip'])

        classifier_head.load_state_dict(ckpt['head'])
        logger.info(f'ckpt[test_acc]: {ckpt["test_acc"]}') 

    elif 'model' in ckpt:  # for SuperContrastive ckpt
        """
        # Missing key(s) in state_dict: "positional_embedding", "text_projection", 
        # "logit_scale", "token_embedding.weight", "ln_final.weight", "ln_final.bias". 
        """

        # load only the visual encoder weights, and keep others the same
        model.load_state_dict(ckpt['model'], strict=False)
        # here we initialize the classifier head with the zeroshot head weights
        classifier_head = classifier_head
        print('ckpt[epoch]:', ckpt['epoch'])
    else:
        print('ckpt.keys():', ckpt.keys())
        classifier_head.load_state_dict(ckpt['best_tau_head'])
        # raise ValueError('No model weights found in the checkpoint.')

    del ckpt

    if test_loader is not None:
        model_test_acc, _, _ = validate(args, data_loader=test_loader, model=model,
                                        logger=logger,
                                        loss=args.loss, logit_scale=args.logit_scale,
                                        classifier_head=classifier_head,
                                        dataset=args.dataset,
                                        device=args.device,
                                        pre_extracted=args.pre_extracted,
                                        )
        logger.info(f"Loaded Model Test Acc: {round(model_test_acc, 3)}")


def calculate_scores(confusion_matrix):
    # the diagonal of the confusion matrix is the number of correct predictions for each class
    # along the rows, we have the predicted labels, along the columns, we have the ground truth labels

    # the sum of each row of the confusion matrix is the total number of instances for each true class
    # divide the diagonal by the sum of each row is the same as TP / (TP + FN), which is the recall

    scores = {}
    num_class = confusion_matrix.shape[0]

    scores['acc'] = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # calculate the avg class accuracy 
    class_accuracy = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    avg_class_accuracy = class_accuracy.mean() * 100
    scores['avg_class_accuracy'] = avg_class_accuracy  # this is the micro accuracy, which would be different from the macro accuracy as in test_acc

    # calculate the per-class recall, precision and f1 score
    recall = dict()
    precision = dict()
    f1_score = dict()

    for i in range(num_class):
        tp = confusion_matrix[i, i]
        fn = np.sum(confusion_matrix[i, :]) - tp
        fp = np.sum(confusion_matrix[:, i]) - tp

        if tp + fn == 0:
            recall[i] = 0.0
        else:
            recall[i] = tp / (tp + fn)

        if tp + fp == 0:
            precision[i] = 0.0
        else:
            precision[i] = tp / (tp + fp)

        if tp == 0:
            f1_score[i] = 0
        else:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    scores['per_class_recall'] = recall
    scores['per_class_precision'] = precision
    scores['per_class_f1score'] = f1_score

    return scores


def validate_dataset(args, data_loader, model, logger, loss, logit_scale, classifier_head=None,
                     show_confusion_matrix=False, device='cuda',
                     dataset='semi-aves', output_dir='output',
                     predict_labels=False, predict_split='u_train', pre_extracted=False):
    model.eval()
    if classifier_head:
        classifier_head.eval()

    val_acc = 0
    val_count = 0

    if show_confusion_matrix:
        num_classes = classifier_head.num_classes
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        # along the rows, we have the predicted labels, along the columns, we have the ground truth labels

    with torch.no_grad():
        predicted_labels = []
        max_logits = []
        val_loss_batch = []
        for i, val_data in enumerate(data_loader):
            inputs, labels, texts, source = val_data
            inputs = inputs.to(device)
            labels = source.long().cuda()  # use the source as the labels for dataset classification

            if not pre_extracted:
                image_features = model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            else:
                image_features = inputs

            if classifier_head:
                logit = classifier_head(image_features)
            else:
                logit, _ = model(inputs, texts)

                # val loss
            logits = logit * logit_scale.exp()
            logits = logits.cuda()
            if args.loss_name == "WeightedCE":
                loss_batch = loss(logits, labels, source)
            else:
                loss_batch = loss(logits, labels)
            val_loss_batch.append(loss_batch.item())

            labels = labels.cpu()
            max_logits.append(torch.max(logit, dim=1).values.cpu().numpy())
            pred = torch.argmax(logit, dim=1).cpu()
            predicted_labels.append(pred.numpy())

            val_acc += torch.sum(pred == labels).item()
            val_count += labels.size(0)

            if show_confusion_matrix:
                confusion_matrix.update(pred, labels)

    # average class validation accuracy
    val_acc = (val_acc / val_count) * 100

    # average validation loss
    val_loss = np.mean(val_loss_batch)

    if predict_labels:
        predicted_labels = np.concatenate(predicted_labels)
        print('predict_labels.shape: ', predicted_labels.shape)
        predicted_labels = predicted_labels.tolist()

        max_logits = np.concatenate(max_logits)
        print('max_logits.shape: ', max_logits.shape)
        max_logits = max_logits.tolist()

        # save the predicted labels to a text file
        predicted_label_file = f'{output_dir}/{dataset}_{predict_split}_predicted_labels.txt'
        with open(predicted_label_file, 'w') as f:
            for item, logit in zip(predicted_labels, max_logits):
                f.write("%s %s\n" % (item, logit))
        logger.info(f'Predicted labels saved to: {predicted_label_file}')

    if show_confusion_matrix:
        confusion_matrix = confusion_matrix.compute().numpy()
        return val_acc, val_loss, confusion_matrix

    return val_acc, val_loss, None


def validate(args, data_loader, model, logger, loss, logit_scale, classifier_head=None,
             show_confusion_matrix=False, device='cuda',
             dataset='semi-aves', output_dir='output',
             predict_labels=False, predict_split='u_train', pre_extracted=False):
    model.eval()
    if classifier_head:
        classifier_head.eval()
    preds, corrects = [], []
    val_acc = 0
    val_count = 0
    all_dict = {}
    for i in range(200):
        all_dict[str(i)] = []

    if show_confusion_matrix:
        num_classes = classifier_head.num_classes
        confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        # along the rows, we have the predicted labels, along the columns, we have the ground truth labels

    with torch.no_grad():
        predicted_labels = []
        max_logits = []
        val_loss_batch = []
        for i, val_data in enumerate(data_loader):
            inputs, labels, texts, source = val_data
            inputs = inputs.to(device)
            labels = labels.long().cuda()

            if not pre_extracted:
                image_features = model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            else:
                image_features = inputs

            if classifier_head:
                logit = classifier_head(image_features)
            else:
                logit, _ = model(inputs, texts)

                # val loss
            logits = logit * logit_scale.exp()
            logits = logits.cuda()
            if args.loss_name == "WeightedCE":
                loss_batch = loss(logits, labels, source)
            else:
                loss_batch = loss(logits, labels)
            val_loss_batch.append(loss_batch.item())

            labels = labels.cpu()
            max_logits.append(torch.max(logit, dim=1).values.cpu().numpy())
            pred = torch.argmax(logit, dim=1).cpu()
            predicted_labels.append(pred.numpy())
            corrects.append(labels.numpy())
            preds.append(pred.numpy())


            val_acc += torch.sum(pred == labels).item()
            val_count += labels.size(0)

            if show_confusion_matrix:
                confusion_matrix.update(pred, labels)
            for j in range(len(pred)):
                all_dict[str(int(pred[j]))].append(i*128+j)
    # average class validation accuracy
    corrects = np.concatenate(corrects)
    corrects = corrects.tolist()
    preds = np.concatenate(preds)
    preds = preds.tolist()
    acc = accuracy_score(corrects, preds)
    macro_f1 = f1_score(corrects, preds, average='macro')
    length = len(corrects)
    val_acc = (val_acc / val_count) * 100

    # average validation loss
    val_loss = np.mean(val_loss_batch)

    if predict_labels:
        predicted_labels = np.concatenate(predicted_labels)
        print('predict_labels.shape: ', predicted_labels.shape)
        predicted_labels = predicted_labels.tolist()

        max_logits = np.concatenate(max_logits)
        print('max_logits.shape: ', max_logits.shape)
        max_logits = max_logits.tolist()

        # save the predicted labels to a text file
        predicted_label_file = f'{output_dir}/{dataset}_{predict_split}_predicted_labels.txt'
        with open(predicted_label_file, 'w') as f:
            for item, logit in zip(predicted_labels, max_logits):
                f.write("%s %s\n" % (item, logit))
        logger.info(f'Predicted labels saved to: {predicted_label_file}')

    if show_confusion_matrix:
        confusion_matrix = confusion_matrix.compute().numpy()
        return val_acc, val_loss, acc*100, macro_f1*100, length, confusion_matrix, all_dict

    return val_acc, val_loss, acc*100, macro_f1*100, length, None, all_dict


def validate_topK(data_loader, model, prompt_vectors, logger, device='cuda',
                  dataset='semi-inat-2021', show_confusion_matrix=True, k=3):
    with torch.no_grad():
        model.eval()
        correct, wrong, val_acc = 0, 0, 0
        val_count = 0

        if show_confusion_matrix:
            num_classes = len(prompt_vectors)  # For now.
            confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        for i, val_data in enumerate(data_loader):
            if dataset == 'semi-inat-2021':
                inputs, labels, l_target_k, l_target_p, l_target_c, l_target_o, l_target_f, l_target_g = val_data
            else:
                inputs, labels = val_data

            images = inputs.to(device)
            labels = labels.to(device).long()
            bsz = labels.shape[0]
            #print(bsz)
            logits = torch.zeros(num_classes, bsz)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            start = time.time()
            for j in range(num_classes):
                desc_prompts = prompt_tensors[j]['all']
                k = desc_prompts.shape[0]
                if (desc_prompts.shape[0] > 2):
                    k = 3
                desc_prompts = desc_prompts.to(device)
                desc_prompts = desc_prompts.squeeze()
                cosine_sim = image_features @ desc_prompts.t()
                top_k = cosine_sim.topk(k=k, dim=-1).values
                logits[j] = top_k.mean(dim=-1)

            #print(time.time()-start)
            logits = logits.to(device)
            pred = torch.argmax(logits, dim=0)
            val_acc += torch.sum(pred == labels).item()
            val_count += labels.size(0)
            if show_confusion_matrix:
                preds = pred.cpu()
                labels = labels.cpu()
                confusion_matrix.update(preds, labels)

            images.cpu()
        val_acc = (val_acc / val_count) * 100

        print(f'Top 1 validation accuracy: {val_acc}')
        logger.info(f'Top 1 validation accuracy: {val_acc}')
        quit()
        if show_confusion_matrix:
            return val_acc, confusion_matrix
    return val_acc
