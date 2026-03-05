#import data_setup, utils, mil, engine, ResNet, visualization
import utils, data_load, mil_tiago, engine
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from pathlib import Path
import time
import datetime
import numpy as np
from typing import List, Union
import pandas as pd
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler, CosineLRScheduler
from timm.utils import get_state_dict, ModelEma, NativeScaler
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os

def get_args_parser():
   
    parser = argparse.ArgumentParser('MIL - Version 2', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--output_dir', default='model_x', help='path where to save, empty for no saving')
    parser.add_argument('--data_path', default='', help='path to input file')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    
    # Wanb parameters
    parser.add_argument('--project_name', default='Thesis', help='name of the project')
    parser.add_argument('--hardware', default='Server', choices=['Server', 'Colab', 'MyPC'], help='hardware used')
    parser.add_argument('--run_name', default='MIL', help='name of the run')
    parser.add_argument('--wandb', action='store_false', default=True, help='whether to use wandb')
    
    # Data parameters
    parser.add_argument('--input_size', default=224, type=int, help='image size')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')
    parser.add_argument('--nb_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--pin_mem', default=True, type=bool, help='pin memory')
    
    # Training parameters
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--training', default=True, type=bool, help='training or testing')
    parser.add_argument('--finetune', default=False, type=bool, help='finetune or not')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--classifier_warmup_epochs', type=int, default=0, metavar='N')
    
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.0)')
        
    # MIL parameters
    parser.add_argument('--pooling_type', default='max', choices=['max', 'avg', 'topk'], type=str, help="")
    parser.add_argument('--mil_type', default='instance', choices=['instance', 'attention', 'embedding'], type=str, help="")
    parser.add_argument('--topk', default=25, type=int, help='topk for topk pooling')
    
    # Pretrained parameters
    parser.add_argument('--pretrained_feature_extractor_path', default='https://download.pytorch.org/models/resnet18-5c106cde.pth', 
                        type=str, help="")
    parser.add_argument('--feature_extractor_pretrained_dataset', default='ImageNet1k', type=str, metavar='DATASET')
    parser.add_argument('--feature_extractor_pretrained_model_name', default='resnet18', type=str, metavar='MODEL')
    parser.add_argument('--dataset_name', default='ISIC2019-Clean', type=str, metavar='DATASET')
        
    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate model on validation set')
    parser.add_argument('--evaluate_model_name', default='MIL_model_0.pth', type=str, help="")
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize model')
    parser.add_argument('--images_path', default="", type=str, help="")
    parser.add_argument('--visualize_relevant_patches', action='store_true', default=False, help='Visualize relevant patches')
    
    # Imbalanced dataset parameters
    parser.add_argument('--class_weights', action='store_true', default=True, help='Enabling class weighting')
    parser.add_argument('--class_weights_type', default='Manual', choices=['Median', 'Manual'], type=str, help="")
    
    # Optimizer parameters 
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', choices=['adamw', 'sgd'],
                        help='Optimizer (default: "adamw")')
    
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')

    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters 
    parser.add_argument('--lr_scheduler', action='store_true', default=True)
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER', choices=['step', 'multistep', 'cosine', 'plateau','poly', 'exp'],
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    
    # * Lr Cosine Scheduler Parameters
    parser.add_argument('--cosine_one_cycle', type=bool, default=False, help='Only use cosine one cycle lr scheduler')
    parser.add_argument('--lr_k_decay', type=float, default=1.0, help='LR k rate (default: 1.0)')
    parser.add_argument('--lr_cycle_mul', type=float, default=1.0, help='LR cycle mul (default: 1.0)')
    parser.add_argument('--lr_cycle_decay', type=float, default=1.0, help='LR cycle decay (default: 1.0)')
    parser.add_argument('--lr_cycle_limit', type=int, default=1, help= 'LR cycle limit(default: 1)')
    
    parser.add_argument('--lr-noise', type=Union[float, List[float]], default=None, help='Add noise to lr')
    parser.add_argument('--lr-noise-pct', type=float, default=0.1, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.1)')
    parser.add_argument('--lr-noise-std', type=float, default=0.05, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 0.05)')
    
    # * Warmup parameters
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-3)')
    
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--cooldown_epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=30, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')

    # * StepLR parameters
    parser.add_argument('--decay_epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    
    # * MultiStepLRScheduler parameters
    parser.add_argument('--decay_milestones', type=List[int], nargs='+', default=(10, 15), 
                        help='epochs at which to decay learning rate')
    
    # * The decay rate is transversal to many schedulers | However it has a different meaning for each scheduler
    # MultiStepLR: decay factor of learning rate | PolynomialLR: power factor | ExpLR: decay factor of learning rate
    parser.add_argument('--decay_rate', '--dr', type=float, default=1., metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Model EMA parameters -> Exponential Moving Average Model
    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=30, metavar='N')
    parser.add_argument('--delta', type=float, default=0.0, metavar='N')
    parser.add_argument('--counter_saver_threshold', type=int, default=2, metavar='N')
    
    # Data augmentation parameters 
    parser.add_argument('--batch_aug', action='store_true', default=False, help='whether to augment batch')
    parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT', help='Color jitter factor (default: 0.)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + \
                        "(default: rand-m9-mstd0.5-inc1)'),
    
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.1, metavar='PCT', help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='const', help='Random erase mode (default: "const")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False, help='Do not random erase first (clean) augmentation split')
    
    # Loss scaler
    parser.add_argument('--loss_scaler', action='store_true', default=False, help='Use loss scaler')
         
    return parser


def main(args):
    
    # Print arguments
    print("----------------- Args -------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("------------------------------------------\n")
    
    # Set device
    device = args.gpu if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Fix the seed for reproducibility
    utils.configure_seed(args.seed)
    cudnn.benchmark = True
    
    ################## Data Setup ##################
    
    noise = 1
    if noise == 1:
        train_csv = 'Files/train_noise_CT.csv'
    else:
        train_csv = 'Files/train_dataset.csv'
        
    # Warmup dataset (start with sampled balanced dataset) 
    warmup = False
    progressive_warmup = False
   
    if(warmup):
        train = True
        #Create balanced dataset
        utils.warmup_dataset(train_csv, train)
        balanced_dataset = 'Files/warmup_train.csv'
        #weight computing
        weight_tensor_warmup = utils.weight_computing(balanced_dataset)
        #create dataloader for simple warmup
        dataset_warmup = data_load.FeatureDataset(balanced_dataset,'Files/pathology_dataset.csv', bag_size=8, noise=noise)
        data_loader_warmup = DataLoader(dataset_warmup, batch_size=1, 
                                 shuffle=True, num_workers=args.num_workers, 
                                 pin_memory=args.pin_mem, drop_last=False)        
        if(progressive_warmup):
            #create datasets with progressively more majority classes and weight computing
            utils.progressive_warmup_create(train_csv, balanced_dataset)
            warmup_train1 ='Files/warmup_train1.csv'
            weight_tensor_warmup1 = utils.weight_computing(warmup_train1)
            warmup_train2 ='Files/warmup_train2.csv'
            weight_tensor_warmup2 = utils.weight_computing(warmup_train2)
            #create dataloader for dataset1
            dataset_warmup1 = data_load.FeatureDataset(warmup_train1,'Files/pathology_dataset.csv', bag_size=8, noise=noise)
            data_loader_warmup1 = DataLoader(dataset_warmup1, batch_size=1, 
                                     shuffle=True, num_workers=args.num_workers, 
                                     pin_memory=args.pin_mem, drop_last=False)
            #create dataloader for dataset2
            dataset_warmup2 = data_load.FeatureDataset(warmup_train2,'Files/pathology_dataset.csv', bag_size=8, noise=noise)
            data_loader_warmup2 = DataLoader(dataset_warmup2, batch_size=1, 
                                     shuffle=True, num_workers=args.num_workers, 
                                     pin_memory=args.pin_mem, drop_last=False)

    
    
    #Create dataloaders for final train and validation
    dataset_train = data_load.FeatureDataset(train_csv,'Files/pathology_dataset.csv', bag_size=8, noise=noise)
    data_loader_train = DataLoader(dataset_train, batch_size=1, 
                             shuffle=True, num_workers=args.num_workers, 
                             pin_memory=args.pin_mem, drop_last=False)
    
    dataset_val = data_load.FeatureDataset('Files/val_dataset.csv','Files/pathology_dataset.csv', bag_size=8, noise=noise)
    data_loader_val = DataLoader(dataset_val, batch_size=1, 
                             shuffle=False, num_workers=args.num_workers, 
                             pin_memory=args.pin_mem, drop_last=False)
        
        
    ############################ Define the Model ############################
    model = mil_tiago.InstanceMIL(num_classes=2, 
                                N=8,
                                dropout=args.dropout,
                                pooling_type=args.pooling_type,
                                device=device,
                                args=args,
                                patch_extractor=None)
    
    model.to(device)

    model_ema = None 
    if args.model_ema:
        print('-> Creating EMA model\n')
        model_ema = ModelEma(model,decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
    

    #Calculate weights for the train loss
    weight_tensor_train = utils.weight_computing(train_csv)
    print("WEIGHTS TRAIN:", weight_tensor_train)
    
    #Loss intialization
    criterion_train = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_train[0])
    
    if warmup:
        criterion_warmup = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_warmup[0])
        if progressive_warmup:
            criterion_warmup1 = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_warmup1[0])
            criterion_warmup2 = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_warmup2[0])
         
    #Define optimizer and LR scheduler
    optimizer = create_optimizer(args=args, model=model)
    
    lr_scheduler = None
    if args.lr_scheduler:
        if args.sched == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
        elif args.sched == 'step':
            step_lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.01)
            lr_scheduler = step_lr_schedulerlr_scheduler = step_lr_scheduler
        else:   
            lr_scheduler, _ = create_scheduler(args, optimizer)    
    

   
    
    
    # Define the loss scaler
    if args.loss_scaler:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None
   

    train_results = {'loss': [], 'acc': [] , 'lr': []}
    val_results = {'loss': [], 'acc': [], 'f1': [], 'cf_matrix': [], 'bacc': [], 'precision': [], 'recall': []}
    best_val_bacc = 0.0
 
    output_dir = Path(args.output_dir)
    early_stopping = engine.EarlyStopping(patience=args.patience, delta=args.delta)
    
    print("******** NUMBER OF EPOCHS:" , (args.epochs + args.cooldown_epochs))
    
    for epoch in range(args.start_epoch, (args.epochs + args.cooldown_epochs)):
        # Train the model
        if warmup and epoch < 50:
            if progressive_warmup and epoch < 20:
                dataset = data_loader_warmup
                criterion = criterion_warmup
            elif progressive_warmup and epoch < 35 and epoch >= 20:
                dataset = data_loader_warmup1
                criterion = criterion_warmup1
            elif progressive_warmup and epoch < 50 and epoch >= 35:
                dataset = data_loader_warmup2
                criterion = criterion_warmup2
        else:
            warmup = False
            dataset = data_loader_train
            criterion = criterion_train

        train_stats = engine.train_step(
            model=model,
            dataloader=dataset,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            loss_scaler=loss_scaler,
            lr_scheduler=lr_scheduler,
            model_ema=model_ema,
            warmup=False,  # This is already handled by dataset selection above
            args=args
        )
        
        if args.sched == 'step':
            lr_scheduler.step() #step LR
        else:
            lr_scheduler.step(epoch+1) #others
        
        # Validation
        results = engine.evaluation(
            model=model,
            dataloader=data_loader_val,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            args=args
        )

        train_results['loss'].append(train_stats['train_loss'])
        train_results['acc'].append(train_stats['train_acc'])
        train_results['lr'].append(train_stats['train_lr'])
        val_results['acc'].append(results['acc1'])
        val_results['loss'].append(results['loss'])
        val_results['f1'].append(results['f1_score'])
        val_results['cf_matrix'].append(results['confusion_matrix'])
        val_results['precision'].append(results['precision'])
        val_results['recall'].append(results['recall'])
        val_results['bacc'].append(results['bacc'])

        print(f"Epoch: {epoch+1} | lr: {train_stats['train_lr']:.5f} | Train Loss: {train_stats['train_loss']:.4f} | "
              f"Train Acc: {train_stats['train_acc']:.4f} | Val. Loss: {results['loss']:.4f} | "
              f"Val. Acc: {results['acc1']:.4f} | Val. Bacc: {results['bacc']:.4f} | F1-score: {np.mean(results['f1_score']):.4f}")
        
        #Early Stopping only after warmup!
        if warmup == False:
            if results['bacc'] > best_val_bacc:
                # Only want to save the best checkpoints if the best val bacc and the early stopping counter is less than the threshold
                best_val_bacc = results['bacc']
                best_results = results
                best_epoch = epoch  # Update the best epoch here
                early_stopping(val_loss=results['loss'], model=model)
                print(best_epoch)
            if early_stopping.early_stop:
                print("\t[INFO] Early stopping - Stop training")
                break
    """
    **
    ****
    ******
    *********
    ***********  END OF THE TRAINING LOOP *************************************
    *********
    ******
    ****
    **
    """
    utils.plot_training_stats(train_results, val_results)
    if best_results:
        checkpoint_path = output_dir / 'best_checkpoint.pth'
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': best_epoch,  # Use the best_epoch variable here
            'args': args,
        }
        if args.lr_scheduler:
            checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
        if model_ema is not None:
            checkpoint_dict['model_ema'] = get_state_dict(model_ema)

        utils.save_on_master(checkpoint_dict, checkpoint_path)

        # Print metrics for the best checkpoint
        best_train_loss = train_results['loss'][best_epoch]
        best_train_acc = train_results['acc'][best_epoch]
        print("BEST CHECKPOIN: \n")
        print(f"Train Loss: {best_train_loss:.4f} | Train Acc: {best_train_acc:.4f} | "
              f"Val Loss: {best_results['loss']:.4f} | Val Acc: {best_results['acc1']:.4f} | "
              f"Val Bacc: {best_results['bacc']:.4f} | Val F1-Score: {np.mean(best_results['f1_score']):.4f}")
        print(f"[INFO] Best Val. Bacc: {(best_val_bacc * 100):.2f}% | [INFO] Saving model as 'best_checkpoint.pth'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MIL - Version 2', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)
    
    
