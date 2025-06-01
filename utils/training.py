import random
import torch
import numpy as np
import copy
from .models import build_classifier_head, save_model_ckpt, save_test_scores
from .dataloader import extract_dataloader, extract_train_dataloader
from testing import validate, calculate_scores, validate_dataset, load_model


def set_training_seed(args):
    # set the seed for training
    random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)

    # this is critical for reproducibility for ResNet50 models
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_zeroshot(args, test_loader, model, logger, loss, logit_scale, classifier_head):
    if args.method == 'dataset-cls':
        zs_test_acc, zs_loss, zs_confusion_matrix = validate_dataset(args, data_loader=test_loader, model=model,
                                                                     logger=logger,
                                                                     loss=loss, logit_scale=logit_scale,
                                                                     classifier_head=classifier_head,
                                                                     show_confusion_matrix=True,
                                                                     dataset=args.dataset,
                                                                     output_dir=args.output_dir, device=args.device,
                                                                     pre_extracted=args.pre_extracted,
                                                                     )
    else:
        zs_test_acc, zs_loss, zs_confusion_matrix = validate(args, data_loader=test_loader, model=model, logger=logger,
                                                             loss=loss, logit_scale=logit_scale,
                                                             classifier_head=classifier_head,
                                                             show_confusion_matrix=True,
                                                             dataset=args.dataset,
                                                             output_dir=args.output_dir, device=args.device,
                                                             pre_extracted=args.pre_extracted,
                                                             )
    logger.info(f"+++++ Zero-shot Test Acc: {round(zs_test_acc, 3)}")


def train_probing(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ Train the model with Cross-Entropy Loss, linear probing"""

    logger.info(f"Start Training (linear probing) ......")

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    model.eval()
    classifier_head.train()

    best_records = {}
    best_val_acc = -1
    num_iter = 0
    for epoch in range(1, args.epochs + 1):

        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()

            if not args.pre_extracted:
                image_features = model.encode_image(images)
                image_feature = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_feature = images

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source)  # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()  # update learning rate for each iteration

        # validate after 1 epoch
        val_acc, val_loss, val_acc_sk, val_macro_f1_sk, val_length, confusion_matrix, all_dict = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                       loss=args.loss, logit_scale=logit_scale,
                                                       classifier_head=classifier_head, show_confusion_matrix=False,
                                                       dataset=args.dataset,
                                                       output_dir=args.output_dir, device=args.device,
                                                       pre_extracted=args.pre_extracted)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_confusion_matrix'] = best_confusion_matrix

        # test after 1 epoch
        test_acc, _, test_acc_sk, test_macro_f1_sk, test_length, _, all_dict = validate(args, data_loader=test_loader, model=model, logger=logger,
                                  loss=args.loss, logit_scale=logit_scale,
                                  classifier_head=best_head,
                                  dataset=args.dataset,
                                  output_dir=args.output_dir, device=args.device,
                                  pre_extracted=args.pre_extracted)

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(
            f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)}, {round(test_acc, 6)}, {round(test_macro_f1_sk, 6)}\n')
        loss_logger.flush()
        print(
            f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Val Macro F1: {round(val_macro_f1_sk, 3)}, Test Acc: {round(test_acc, 3)}, Test Macro F1: {round(test_macro_f1_sk, 3)}")

    logger.info(f'Probing done.')

    return best_model, best_head, best_records, best_logit_scale, val_loader, test_loader


def train_ce(args, logger, loss_logger, model, classifier_head, train_loader, val_loader, test_loader):
    """ Train the model with Cross-Entropy Loss, finetuning visual encoder and classifier"""

    print(f"Start standard finetuning ......")

    model.eval() if args.freeze_visual else model.train()
    classifier_head.train()

    logit_scale = args.logit_scale
    loss = args.loss.cuda()
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = classifier_head
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss_sum = 0
        for inputs, labels, text, source in train_loader:
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            source = source.to(args.device)

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = classifier_head(image_feature)
            logits = logits * logit_scale.exp()
            if args.loss_name == 'WeightedCE':
                total_loss = loss(logits, labels, source)  # for WeightedCE, needs to input the source
            else:
                total_loss = loss(logits, labels)
            train_loss = total_loss.item()
            train_loss_sum += train_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()  # update learning rate for each iteration

        if args.early_stop or epoch == args.epochs:
            val_acc, val_loss, val_acc_sk, val_macro_f1_sk, val_length, confusion_matrix, all_dict = validate(args, data_loader=val_loader, model=model, logger=logger,
                                                           loss=args.loss, logit_scale=logit_scale,
                                                           classifier_head=classifier_head, show_confusion_matrix=True,
                                                           dataset=args.dataset,
                                                           output_dir=args.output_dir, device=args.device,
                                                           pre_extracted=args.pre_extracted,
                                                           )
            scores = calculate_scores(confusion_matrix)

        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            # logger.info(f'Val Acc improved from {round(best_val_acc, 3)} to {round(val_acc, 3)}')
            best_val_acc = val_acc
            best_logit_scale = copy.deepcopy(logit_scale)
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(classifier_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        if args.early_stop or epoch == args.epochs:
            test_acc, _, test_acc_sk, test_macro_f1_sk, test_length, _, all_dict = validate(args, data_loader=test_loader, model=best_model, logger=logger,
                                      loss=args.loss, logit_scale=logit_scale,
                                      classifier_head=best_head,
                                      dataset=args.dataset,
                                      output_dir=args.output_dir, device=args.device,
                                      pre_extracted=args.pre_extracted,
                                      )

        train_loss_avg = train_loss_sum / len(train_loader)
        loss_logger.write(
            f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)}, {round(test_acc, 6)}, {round(test_macro_f1_sk, 6)}\n')
        loss_logger.flush()
        print(
            f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Val Macro F1: {round(val_macro_f1_sk, 3)}, Test Acc: {round(test_acc, 3)}, Test Macro F1: {round(test_macro_f1_sk, 3)}")
        
    logger.info(f'Training done.')
    logger.info(f'Best val Acc: {round(best_val_acc, 3)} at epoch {best_epoch}, iter {best_iter}')

    return best_model, best_head, best_records, best_logit_scale


def train_flyp(args, logger, loss_logger, model, tokenizer,
               train_dataloader, val_dataloader, test_dataloader, text_prompts):
    """ 
    Finetune like you pretrain
    Train the model with contrastive loss, using the text descriptions from labels.
    Can be modified to lock the text encoder.
    """

    assert (
                args.loss_name == 'CE' or args.loss_name == 'WeightedCE'), 'FLYP use CE loss for contrastive loss calculation.'

    logit_scale = args.logit_scale
    loss = args.loss
    optimizer = args.optimizer
    scheduler = args.scheduler

    val_loss = -1
    val_acc = -1
    test_acc = -1
    best_val_acc = -1
    best_epoch = -1
    best_iter = -1
    best_model = model
    best_head = None
    best_logit_scale = logit_scale
    best_records = {}
    num_iter = 0

    logger.info(f"Start Training FLYP ......")

    model.train()
    for epoch in range(1, args.epochs + 1):

        train_loss_sum = 0
        # train for 1 epoch
        for inputs, labels, tokenized_text, source in train_dataloader:
            optimizer.zero_grad()
            num_iter += 1
            images = inputs.to(args.device)
            labels = labels.to(args.device).long()
            tokenized_text = tokenized_text.to(
                args.device)  # currently we use 1 template for semi-aves as in prompt_maker(), can be updated to randomly sample 1 from the 80 prompts

            image_features = model.encode_image(images)
            image_feature = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts = tokenized_text.squeeze()
            text_features = model.encode_text(prompts)
            text_feature = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalization

            scale = logit_scale.exp()
            logits_per_image = scale * image_feature @ text_feature.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(logits_per_image.shape[0], dtype=torch.long).to(args.device)

            if args.loss_name == 'CE':
                total_loss = (loss(logits_per_image, labels) + loss(logits_per_text, labels)) / 2
            elif args.loss_name == 'WeightedCE':
                total_loss = (loss(logits_per_image, labels, source) + loss(logits_per_text, labels, source)) / 2
            else:
                raise ValueError(f'Loss {args.loss_name} not supported for FLYP training.')

            train_loss = total_loss.item()
            train_loss_sum += train_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()  # update learning rate for each iteration

        if args.early_stop or epoch == args.epochs:
            new_head = build_classifier_head(args, model, text_prompts, tokenizer)
            val_acc, val_loss, val_acc_sk, val_macro_f1_sk, val_length, confusion_matrix, all_dict = validate(args, data_loader=val_dataloader, model=model, logger=logger,
                                                           loss=loss, logit_scale=logit_scale,
                                                           classifier_head=new_head, show_confusion_matrix=True,
                                                           dataset=args.dataset,
                                                           output_dir=args.output_dir, device=args.device,
                                                           pre_extracted=False,
                                                           )
            scores = calculate_scores(confusion_matrix)

        # as we might modify the loss function for confusing pairs
        if (args.early_stop or epoch == args.epochs) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_logit_scale = logit_scale
            best_epoch = epoch
            best_iter = num_iter
            best_scores = copy.deepcopy(scores)
            best_confusion_matrix = copy.deepcopy(confusion_matrix)
            best_head = copy.deepcopy(new_head)
            best_model = copy.deepcopy(model)

            # save into the best_records
            best_records['best_val_acc'] = best_val_acc
            best_records['best_logit_scale'] = best_logit_scale
            best_records['best_epoch'] = best_epoch
            best_records['best_iter'] = best_iter
            best_records['best_scores'] = best_scores
            best_records['best_confusion_matrix'] = best_confusion_matrix

        if args.early_stop or epoch == args.epochs:
            test_acc, _, test_acc_sk, test_macro_f1_sk, test_length,  _, all_dict = validate(args, data_loader=test_dataloader, model=best_model, logger=logger,
                                      loss=loss, logit_scale=logit_scale,
                                      classifier_head=best_head,
                                      dataset=args.dataset,
                                      output_dir=args.output_dir, device=args.device,
                                      pre_extracted=False,
                                      )

        train_loss_avg = train_loss_sum / len(train_dataloader)
        loss_logger.write(
            f'{epoch},{num_iter},{round(train_loss_avg, 6)},{round(val_loss, 6)},{round(val_acc, 6)}, {round(test_acc, 6)}\n')
        loss_logger.flush()
        print(f"Epoch {int(epoch)}, Iter {num_iter}, Trn Loss: {round(train_loss_avg, 6)}, Val Loss: {round(val_loss, 6)}, Val Acc: {round(val_acc, 3)}, Val Macro F1: {round(val_macro_f1_sk, 3)}, Test Acc: {round(test_acc, 3)}, Test Macro F1: {round(test_macro_f1_sk, 3)}")

        ## save model checkpoints every X epochs
        if args.save_ckpt and (num_iter % args.save_freq == 0 or epoch == args.epochs):
            model_path = save_model_ckpt(args, best_records,
                                         model, new_head, optimizer, scheduler, logit_scale,
                                         val_acc, epoch, num_iter)
            logger.info(f'Model ckpt saved to: {model_path}')

    return best_model, best_head, best_records, best_logit_scale







