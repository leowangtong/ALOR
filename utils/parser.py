import argparse
import os
from utils.extras import get_class_num_list, str2bool
import yaml


def parse_args():

    parser = argparse.ArgumentParser(description='Arguments for script.')
    
    # logging
    parser.add_argument('--log_mode', type=str, default='both', choices=['console', 'file', 'both'], help='where to log.')
    parser.add_argument('--folder', type=str, default='output', help='Folder for saving output.')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix for Log file Name.')

    # model
    parser.add_argument('--model_cfg', type=str, default='vitb32_openclip_laion400m', 
                        choices=['vitb32_openclip_laion400m', 'vitb16_openclip_laion400m',
                                 'vitb32_openclip_laion2b', 'rn50_openclip_openai',
                                 'vitb32_clip', 'vitb16_clip', 'rn50_clip', 'rn101_clip'
                                 ],
                        help='ViT Transformer arch.')
    # parser.add_argument('--resume_path', type=str, help='Model path to resume training for.')
    parser.add_argument('--model_path', default=None, type=str, help='Model path to start training from.')    

    # prompt
    parser.add_argument('--prompt_name', type=str, default='most_common_name',
                        choices=['most_common_name', 'most_common_name_REAL', 'name', 'name-80prompts',
                                 'c-name', 's-name', 't-name', 'f-name', 'c-name-80prompts'], help='names for prompts.')
    parser.add_argument('--use_attribute', default=False, type=str2bool, help='Use attribute when making prompts.')

    # dataset
    parser.add_argument('--dataset', type=str, default='semi-aves', 
                        choices=['semi_aves', 'aircraft', 'stanford_cars', 'food101', 'oxford_pets'], 
                        help='Dataset name.')
    
    # retrieval
    parser.add_argument('--database', type=str, default='LAION400M', help='Database from which images are mined.')

    # training data
    parser.add_argument('--round', default=0, help='AL round')
    parser.add_argument('--val_split', type=str, default='fewshotX.txt', help='val file name.')
    parser.add_argument('--test_split', type=str, default='test.txt', help='test file name.')
    parser.add_argument('--seed', default=1, help='Random seeds for different splits.')
    parser.add_argument('--training_seed', type=int, default=1, help='Random seeds for training.')

    # training
    parser.add_argument('--method', type=str, default='finetune', choices=['zeroshot','probing', 'finetune', 'FLYP'], 
                        help='Method for training.')
    parser.add_argument('--cls_init', type=str, default='REAL-Prompt', choices=['random', 'text', 'REAL-Prompt', 'REAL-Linear'], 
                        help='Initialize the classifier head in different ways.')

    parser.add_argument('--pre_extracted', default=False, type=str2bool, help='use pre-extracted features.')
    parser.add_argument('--freeze_visual', default=False, type=str2bool, help='Freeze the visual encoder during training.')

    parser.add_argument('--check_zeroshot', action='store_true', help='check zeroshot acc.')
    parser.add_argument('--zeroshot_only', action='store_true', help='run zeroshot only.') 
    parser.add_argument('--early_stop', default=False, type=str2bool, help='use val set for early stopping.')    
    parser.add_argument('--epochs', type=int, default=0, help='number of epochs to train the model')
    parser.add_argument('--stop_epochs', type=int, default=50, help='number of epochs to stop the training of the model')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='Num of workers.') 
    parser.add_argument('--start_validation', type=int, default=0, help='Start validation after x iterations.')    
    parser.add_argument('--lr_classifier', type=float, default=1e-4, help='Learning rate for the classifier head.')
    parser.add_argument('--lr_backbone', type=float, default=1e-6, help='Learning rate for the visual encoder.')
    parser.add_argument('--lr_projector', type=float, default=None, help='Learning rate for the visual and text projector.')    
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay for model.')
    parser.add_argument('--bsz', type=int, default=32, help='Batch Size')
    parser.add_argument('--optim', type=str, default='AdamW', choices=['AdamW', 'SGD'], help='type of optimizer to use.')
    parser.add_argument('--temperature', type=float, default=0.07, help='Logit Scale for training')
    parser.add_argument('--alpha', type=float, default=0.5, help='mixing ratio for WiSE-FT, alpha=1.0 means no WiSE-FT ensembling.')
    parser.add_argument('--ALMETHOD', type=str, default='TFS', help='badge/coreset/entropy/logo/alfa_mix/random/TFS/badge_pcb')

    # loss
    parser.add_argument('--loss_name', type=str, default='CE', choices=['CE', 'WeightedCE', 'Focal', 'BalancedSoftmax'], help='type of loss function to use.')
    parser.add_argument('--dataset_wd', type=float, default=1.0, help='weight decay for dataset classification loss.')
    parser.add_argument('--fewshot_weight', type=float, default=1.0, help='fewshot weights for WeightedCE.')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='alpha for Focal loss.')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma for Focal loss.')  

    # save
    parser.add_argument('--save_ckpt', default=False, type=str2bool, help='Save model checkpoints or not.')
    parser.add_argument('--save_freq', type=int, default=10, help='Save Frequency in epoch.')
    
    args = parser.parse_args()

    # read the dataset and retrieved path from the config.yml file
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.dataset_path = config['dataset_path']
        args.retrieved_path = config['retrieved_path']

    if args.method == 'zeroshot':
        args.check_zeroshot = True
        args.zeroshot_only = True

    # adjust prompt_name based on cls_init
    if args.cls_init == 'REAL-Prompt' or args.cls_init == 'REAL-Linear':
        args.prompt_name = 'most_common_name'
    elif args.cls_init == 'text':
        args.prompt_name = 'name'
    elif args.cls_init == 'random':
        args.prompt_name = 'most_common_name'        

    if args.method == "probing":
        args.freeze_visual = True
        # args.pre_extracted = True # because stage 2 has to recalculate the feature using stage 2 model 
    else:
        args.freeze_visual = False
        args.pre_extracted = False

    if not args.freeze_visual:
        assert args.pre_extracted==False, \
            'visual encoder not frozen, pre-extracted features are not compatible.'

    if args.model_path is not None:
        assert args.pre_extracted==False, 'reloading a trained model, pre-extracted features are not compatible.'
 
 
    #---------- adjust the train and val split based on round, seed
    
    args.val_split = [['val.txt'], [os.path.join(args.dataset_path, args.dataset)]]
    args.test_split = [['test.txt'], [os.path.join(args.dataset_path, args.dataset)]]

    args.train_split = [[f'{args.ALMETHOD}_{args.method}_seed{args.seed}_round{args.round}.txt'], [os.path.join(args.dataset_path, args.dataset)]]
    args.early_stop = True


    # adjust folder
    args.folder = f'{args.folder}/output_{args.dataset}'

    args.dataset_root = f'data/{args.dataset}'

    return args
