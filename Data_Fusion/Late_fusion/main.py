import utils, data_load, networks, engine
import random
import torch
import torch.nn as nn
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os

def update_results(train_results, val_results, train_stats, results, train_modalities, val_modalities):
    train_results['loss_CT'].append(train_stats['train_loss_CT'])
    train_results['loss_MR'].append(train_stats['train_loss_MR'])
    train_results['loss_Path'].append(train_stats['train_loss_Path'])
    train_results['loss_Clingen'].append(train_stats['train_loss_Clingen'])
    train_results['acc'].append(train_stats['train_acc'])
    train_results['lr'].append(train_stats['train_lr'])
    
    val_results['acc'].append(results['acc1'])
    val_results['loss_CT'].append(results['loss_CT'])
    val_results['loss_MR'].append(results['loss_MR'])
    val_results['loss_Path'].append(results['loss_Path'])
    val_results['loss_Clingen'].append(results['loss_Clingen'])
    val_results['f1'].append(results['f1_score'])
    val_results['cf_matrix'].append(results['confusion_matrix'])
    val_results['precision'].append(results['precision'])
    val_results['recall'].append(results['recall'])
    val_results['bacc'].append(results['bacc'])
    
    train_ind_metrics['BACC_CT'].append(train_modalities['BACC_CT'])
    train_ind_metrics['BACC_MR'].append(train_modalities['BACC_MR'])
    train_ind_metrics['BACC_Path'].append(train_modalities['BACC_Path'])
    train_ind_metrics['BACC_Clingen'].append(train_modalities['BACC_Clingen'])
    train_ind_metrics['Recall_CT'].append(train_modalities['Recall_CT'])
    train_ind_metrics['Recall_MR'].append(train_modalities['Recall_MR'])
    train_ind_metrics['Recall_Path'].append(train_modalities['Recall_Path'])
    train_ind_metrics['Recall_Clingen'].append(train_modalities['Recall_Clingen'])
    train_ind_metrics['specificity_CT'].append(train_modalities['specificity_CT'])
    train_ind_metrics['specificity_MR'].append(train_modalities['specificity_MR'])
    train_ind_metrics['specificity_Path'].append(train_modalities['specificity_Path'])
    train_ind_metrics['specificity_Clingen'].append(train_modalities['specificity_Clingen'])
    
    val_ind_metrics['BACC_CT'].append(val_modalities['BACC_CT'])
    val_ind_metrics['BACC_MR'].append(val_modalities['BACC_MR'])
    val_ind_metrics['BACC_Path'].append(val_modalities['BACC_Path'])
    val_ind_metrics['BACC_Clingen'].append(val_modalities['BACC_Clingen'])
    val_ind_metrics['Recall_CT'].append(val_modalities['Recall_CT'])
    val_ind_metrics['Recall_MR'].append(val_modalities['Recall_MR'])
    val_ind_metrics['Recall_Path'].append(val_modalities['Recall_Path'])
    val_ind_metrics['Recall_Clingen'].append(val_modalities['Recall_Clingen'])
    val_ind_metrics['specificity_CT'].append(val_modalities['specificity_CT'])
    val_ind_metrics['specificity_MR'].append(val_modalities['specificity_MR'])
    val_ind_metrics['specificity_Path'].append(val_modalities['specificity_Path'])
    val_ind_metrics['specificity_Clingen'].append(val_modalities['specificity_Clingen'])


def get_args_parser():
   
    parser = argparse.ArgumentParser('Early Fusion', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--output_dir', default='model_x', help='path where to save, empty for no saving')
    parser.add_argument('--checkpoint_name', default='best_checkpoint.pth', help='name of the checkpoint')
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--pin_mem', default=True, type=bool, help='pin memory')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    ##Training parameters
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (default: 0.0)')
    
    # Learning rate schedule parameters 
    parser.add_argument('--lr_scheduler', action='store_true', default=True)
    parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER', choices=['step', 'multistep', 'cosine', 'plateau','poly', 'exp'],
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-1)')
    parser.add_argument('--step_size', type=float, default=15, metavar='StepSize',
                        help='StepSize (default: 10)')
    parser.add_argument('--gamma', type=float, default=1e-1, metavar='Gamma',
                        help='learning rate (default: 1e-2)')
    
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
    
    #Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER', choices=['adam', 'adamw', 'sgd'],
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
    
    # Loss scaler
    parser.add_argument('--loss_scaler', action='store_true', default=False, help='Use loss scaler')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=30, metavar='N')
    parser.add_argument('--delta', type=float, default=0.0, metavar='N')
    parser.add_argument('--counter_saver_threshold', type=int, default=2, metavar='N')
    
    return parser





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Early Fusion', parents=[get_args_parser()])
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    
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

    ################### DATA SETUP ########################
    noise = 0
    if noise == 1:
        train_csv = 'Files/train_noise_CT.csv'
    else:
        train_csv = 'Files/train.csv'
    
    gen_dataset = 'Files/General_dataset_fixed.csv'
    test_csv = 'Files/test.csv'
    CT_files = 'Files/patients_with_labels_CT.csv'
    MR_files = 'Files/patients_with_labels_MR.csv'
    PATH_files = 'Files/slide_location.csv'
    clinical = 'Files/encoded_three_updated.csv'
    CT_choice = 'Files/CT_position_FINAL.csv'
    MR_choice = 'Files/MR_position_FINAL.csv'
    PATH_choice = 'Files/Path_position_FINAL.csv'
   
    # Warmup dataset (start with sampled balanced dataset) 
    warmup = False
    progressive_warmup = True
   
    if(warmup):
        train = True
        #Create balanced dataset
        utils.warmup_dataset(train_csv, train)
        balanced_dataset = 'Files/warmup_train.csv'
        #weight computing
        weight_tensor_warmup = utils.weight_computing(balanced_dataset)
        #create dataloader for simple warmup
        dataset_warmup = data_load.FeatureDataset(balanced_dataset, CT_files, MR_files, PATH_files, clinical, CT_choice, MR_choice, PATH_choice, noise = noise, train = train)
        
        data_loader_warmup = DataLoader(dataset_warmup, batch_size=14, 
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
            dataset_warmup1 = data_load.FeatureDataset(warmup_train1,CT_files, MR_files, PATH_files, clinical, CT_choice, MR_choice, PATH_choice, noise = noise, train = train)
            data_loader_warmup1 = DataLoader(dataset_warmup1, batch_size=14, 
                                     shuffle=True, num_workers=args.num_workers, 
                                     pin_memory=args.pin_mem, drop_last=False)
            
            #create dataloader for dataset2
            dataset_warmup2 = data_load.FeatureDataset(warmup_train2, CT_files, MR_files, PATH_files, clinical, CT_choice, MR_choice, PATH_choice, noise = noise, train = train)
            data_loader_warmup2 = DataLoader(dataset_warmup2, batch_size=14, 
                                     shuffle=True, num_workers=args.num_workers, 
                                     pin_memory=args.pin_mem, drop_last=False)
    
    
    
    
    
    #Defining the dataloaders
    train = True
    dataset_train = data_load.FeatureDataset(train_csv, CT_files, MR_files, PATH_files, clinical, CT_choice, MR_choice, PATH_choice, noise = noise, train = train)
    data_loader_train = DataLoader(dataset_train, batch_size=14, 
                         shuffle=False, num_workers=args.num_workers, 
                         pin_memory=args.pin_mem, drop_last=False)
    train = False
    dataset_test = data_load.FeatureDataset(test_csv, CT_files, MR_files, PATH_files, clinical, CT_choice, MR_choice, PATH_choice, noise = noise, train = train)
    data_loader_test = DataLoader(dataset_test, batch_size=14, 
                         shuffle=False, num_workers=args.num_workers, 
                         pin_memory=args.pin_mem, drop_last=False)
    
    #Defining the model
    model_CT = networks.CT_classifier(args=args)
    model_CT.to(device)
    model_MR = networks.MR_classifier(args=args)
    model_MR.to(device)
    model_Path = networks.Path_classifier(args=args)
    model_Path.to(device)
    model_Clingen = networks.Clingen_classifier(args=args)
    model_Clingen.to(device)
    
    models = [model_CT,model_MR,model_Path,model_Clingen]

    
    weight_tensor_train = utils.weight_computing(train_csv)
    print("WEIGHTS TRAIN:", weight_tensor_train)
    criterion_CT = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_train[1])
    criterion_MR = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_train[1])
    criterion_Path = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_train[1])
    criterion_Clingen = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_train[1])
    
    losses = [criterion_CT,criterion_MR,criterion_Path,criterion_Clingen]
    
    if warmup:
        criterion_warmup = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_warmup[1])
        if progressive_warmup:
            criterion_warmup1 = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_warmup1[1])
            criterion_warmup2 = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor_warmup2[1])
    
    #Define optimizer and LR scheduler
    optimizer_CT = create_optimizer(args=args, model=model_CT)
    optimizer_MR = create_optimizer(args=args, model=model_MR)
    optimizer_Path = create_optimizer(args=args, model=model_Path)
    optimizer_Clingen = create_optimizer(args=args, model=model_Clingen)
    
    optimizers = [optimizer_CT,optimizer_MR,optimizer_Path,optimizer_Clingen]
    
    lr_scheduler = None
    if args.lr_scheduler:
        if args.sched == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        elif args.sched == 'step':
            # Create a single StepLR scheduler and pass it to all optimizers
            
            step_lr_scheduler = StepLR(optimizer_CT, step_size=args.step_size, gamma=args.gamma)
            
            for optimizer in optimizers:
                optimizer.param_groups[0]['lr'] = step_lr_scheduler.get_last_lr()[0]  # Set the initial learning rate for all optimizers
            
            lr_scheduler = step_lr_scheduler
        else:   
            lr_scheduler, _ = create_scheduler(optimizer_CT, optimizers)
            for optimizer in optimizers:
                optimizer.param_groups[0]['lr'] = lr_scheduler.get_last_lr()[0]  # Set the initial learning rate for all optimizers
    
  
    # Define the loss scaler
    if args.loss_scaler:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None

    
    train_results = {'loss_CT': [], 'loss_MR': [], 'loss_Path': [], 'loss_Clingen': [],  'acc': [] , 'lr': []}
    val_results = {'loss_CT': [], 'loss_MR': [],'loss_Path': [],'loss_Clingen': [], 'acc': [], 'f1': [], 'cf_matrix': [], 'bacc': [], 'precision': [], 'recall': []}
    train_ind_metrics = {
            'BACC_CT': [],
            'BACC_MR': [],
            'BACC_Path': [],
            'BACC_Clingen': [],
            'Recall_CT': [],
            'Recall_MR': [],
            'Recall_Path': [],
            'Recall_Clingen': [],
            'specificity_CT': [],
            'specificity_MR': [],
            'specificity_Path': [],
            'specificity_Clingen': []
    }
    
    val_ind_metrics = {
        'BACC_CT': [],
        'BACC_MR': [],
        'BACC_Path': [],
        'BACC_Clingen': [],
        'Recall_CT': [],
        'Recall_MR': [],
        'Recall_Path': [],
        'Recall_Clingen': [],
        'specificity_CT': [],
        'specificity_MR': [],
        'specificity_Path': [],
        'specificity_Clingen': []
    }
    best_val_bacc = 0.0
    early_stopping = engine.EarlyStopping(patience=args.patience, delta=args.delta)
    
    for epoch in range(args.start_epoch, args.epochs):
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
            #criterion = criterion_train        
        
        train_modalities, train_stats = engine.train_step(models = models,
                                            dataloader=data_loader_train,
                                            losses= losses,
                                            optimizers =optimizers,
                                            device=device,
                                            epoch=epoch+1,
                                            loss_scaler=loss_scaler,
                                            lr_scheduler=lr_scheduler,
                                            combinations_csv = train_csv,
                                            args=args)
        if args.sched == 'step' or args.sched == 'exp':
            lr_scheduler.step() #step LR
        else:
            lr_scheduler.step(epoch+1) #others
       
        #removed wand
        val_modalities, results = engine.evaluation(models = models,
                                        dataloader = data_loader_test,
                                        losses = losses,
                                        device=device,
                                        epoch=epoch+1,
                                        combinations_csv = test_csv,
                                        args=args)
     
        update_results(train_results, val_results, train_stats, results, train_modalities, val_modalities)
        

        # Print accuracy for each combination
        """
        print("TRAIN:")
        for combination_str, data in train_stats.get('combination_accuracy', {}).items():
            total = data["total"]
            correct = data["correct"]
            accuracy = data["accuracy"]
            # Convert the combination string back to a tuple for display
            combination_tuple = eval(combination_str)
            print(f"Combination {combination_tuple}:",f"Total: {total}", f"Correct: {correct}",f"Accuracy: {accuracy:.3%}")
        """
        """
        print("VAL:")
        for combination_str, data in results.get('combination_accuracy', {}).items():
            total = data["total"]
            correct = data["correct"]
            accuracy = data["accuracy"]
            # Convert the combination string back to a tuple for display
            combination_tuple = eval(combination_str)
            print(f"Combination {combination_tuple}:",f"Total: {total}", f"Correct: {correct}",f"Accuracy: {accuracy:.3%}")
        """
        print("Validation Metrics:")
        for combination, bacc_value in results.get('combination_bacc', {}).items():
            print(f"Epoch: {epoch+1} | Combination: {combination} | Validation BACC: {bacc_value:.4f}")

        print("Epoch Stats:")
        print(f"Epoch: {epoch+1} | lr: {train_stats['train_lr']:.8f} | "
              f"Train Acc: {train_stats['train_acc']:.4f} | "
              f"Val. Acc: {results['acc1']:.4f} | "
              f"Val. Bacc: {results['bacc']:.4f} | "
              f"F1-score: {np.mean(results['f1_score']):.4f}")

        print("Training Losses:")
        print(f"Train Loss_CT: {train_results['loss_CT'][-1]:.5f} | "
              f"Train Loss_MR: {train_results['loss_MR'][-1]:.5f} | "
              f"Train Loss_Path: {train_results['loss_Path'][-1]:.5f} | "
              f"Train Loss_Clingen: {train_results['loss_Clingen'][-1]:.5f}")

        print("Validation Losses:")
        print(f"Val Loss_CT: {val_results['loss_CT'][-1]:.5f} | "
              f"Val Loss_MR: {val_results['loss_MR'][-1]:.5f} | "
              f"Val Loss_Path: {val_results['loss_Path'][-1]:.5f} | "
              f"Val Loss_Clingen: {val_results['loss_Clingen'][-1]:.5f}")

        print("Validation Metrics:")
        print(f"Val BACC_CT: {val_modalities['BACC_CT']:.5f} | "
              f"Val BACC_MR: {val_modalities['BACC_MR']:.5f} | "
              f"Val BACC_Path: {val_modalities['BACC_Path']:.5f} | "
              f"Val BACC_Clingen: {val_modalities['BACC_Clingen']:.5f} |\n"
              f"Val Recall_CT: {val_modalities['Recall_CT']:.5f} | "
              f"Val Recall_MR: {val_modalities['Recall_MR']:.5f} | "
              f"Val Recall_Path: {val_modalities['Recall_Path']:.5f} | "
              f"Val Recall_Clingen: {val_modalities['Recall_Clingen']:.5f} |\n"
              f"Val specificity_CT: {val_modalities['specificity_CT']:.5f} | "
              f"Val specificity_MR: {val_modalities['specificity_MR']:.5f} | "
              f"Val specificity_Path: {val_modalities['specificity_Path']:.5f} | "
              f"Val specificity_Clingen: {val_modalities['specificity_Clingen']:.5f}")



        
        print("*****************EPOCH**************", epoch+1)
        
        
        
        if warmup == False:
            if results['bacc'] > best_val_bacc:
                # Only want to save the best checkpoints if the best val bacc and the early stopping counter is less than the threshold
                best_train = train_stats
                best_val_bacc = results['bacc']
                best_results = results
                best_metrics = val_modalities
                best_epoch = epoch  # Update the best epoch here
                #early_stopping(val_loss=results['loss'], model=model)
                print(best_epoch)
                   

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
    
    # Access the best model's state_dict and validation results
    """
    utils.plot_training_stats(train_results, val_results)
    if best_results:
        checkpoint_path = output_dir / args.checkpoint_name
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': best_epoch,  # Use the best_epoch variable here
            'args': args,
        }
        if args.lr_scheduler:
            checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()

        utils.save_on_master(checkpoint_dict, checkpoint_path)
        """
    
    if best_results:
        print("BEST CHECKPOINT Epoch:", best_epoch + 1)
        print("Validation Metrics:")
        for combination, bacc_value in best_results.get('combination_bacc', {}).items():
            print(f"Combination: {combination} | Best Val BACC: {bacc_value:.4f}")
        
        print("Training Losses:")
        print(f"Train Loss_CT: {best_train.get('loss_CT', 0.0):.5f} | "
              f"Train Loss_MR: {best_train.get('loss_MR', 0.0):.5f} | "
              f"Train Loss_Path: {best_train.get('loss_Path', 0.0):.5f} | "
              f"Train Loss_Clingen: {best_train.get('loss_Clingen', 0.0):.5f}")


        print("Validation Losses:")
        print(f"Val Loss_CT: {best_results.get('loss_CT', 0.0):.5f} | "
              f"Val Loss_MR: {best_results.get('loss_MR', 0.0):.5f} | "
              f"Val Loss_Path: {best_results.get('loss_Path', 0.0):.5f} | "
              f"Val Loss_Clingen: {best_results.get('loss_Clingen', 0.0):.5f}")

        print("Validation Metrics:")
        print(f"Val BACC_CT: {best_metrics.get('BACC_CT', 0.0):.5f} | "
              f"Val BACC_MR: {best_metrics.get('BACC_MR', 0.0):.5f} | "
              f"Val BACC_Path: {best_metrics.get('BACC_Path', 0.0):.5f} | "
              f"Val BACC_Clingen: {best_metrics.get('BACC_Clingen', 0.0):.5f} |\n"
              f"Val Recall_CT: {best_metrics.get('Recall_CT', 0.0):.5f} | "
              f"Val Recall_MR: {best_metrics.get('Recall_MR', 0.0):.5f} | "
              f"Val Recall_Path: {best_metrics.get('Recall_Path', 0.0):.5f} | "
              f"Val Recall_Clingen: {best_metrics.get('Recall_Clingen', 0.0):.5f} |\n"
              f"Val specificity_CT: {best_metrics.get('specificity_CT', 0.0):.5f} | "
              f"Val specificity_MR: {best_metrics.get('specificity_MR', 0.0):.5f} | "
              f"Val specificity_Path: {best_metrics.get('specificity_Path', 0.0):.5f} | "
              f"Val specificity_Clingen: {best_metrics.get('specificity_Clingen', 0.0):.5f}")
        
        print(f"BEST VAL BACC: {(best_val_bacc):.4f}")

