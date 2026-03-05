import utils, data_load, enc_dec, engine_rec
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


def get_args_parser():
   
    parser = argparse.ArgumentParser('Early Fusion', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--output_dir', default='enc_dc/experiment_18', help='path where to save, empty for no saving')
    parser.add_argument('--checkpoint_name', default='NOT USED', help='name of the checkpoint')
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--pin_mem', default=True, type=bool, help='pin memory')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    ##Training parameters
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    
    # Learning rate schedule parameters 
    parser.add_argument('--lr_scheduler', action='store_true', default=True)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', choices=['step', 'multistep', 'cosine', 'plateau','poly', 'exp'],
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-1)')
    parser.add_argument('--step_size', type=float, default=60, metavar='StepSize',
                        help='StepSize (default: 10)')
    parser.add_argument('--gamma', type=float, default=1e-1, metavar='Gamma',
                        help='learning rate (default: 1e-2)')
    
    #Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', choices=['adam', 'adamw', 'sgd'],
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
    
    
    # LR additional 
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
    

    # Loss scaler
    parser.add_argument('--loss_scaler', action='store_true', default=False, help='Use loss scaler')
    
    # Early stopping parameters
    parser.add_argument('--patience', type=int, default=30, metavar='N')
    parser.add_argument('--delta', type=float, default=0.0, metavar='N')
    parser.add_argument('--counter_saver_threshold', type=int, default=2, metavar='N')
    
    # Fusion methods
    parser.add_argument('--fusion', type=str, default='mean', metavar='FusionMethod')
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
    noise = 1
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
    encoder = enc_dec.Encoder(args=args)
    encoder.to(device)
    decoder = enc_dec.Decoder(args=args)
    decoder.to(device)
    
    model_parameters = list(encoder.parameters()) + list(decoder.parameters())
    
    #Define optimizer and LR scheduler
    optimizer = create_optimizer(args=args, model = model_parameters)
    
    criterion_train = nn.MSELoss()
    
    lr_scheduler = None
    if args.lr_scheduler:
        if args.sched == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        elif args.sched == 'step':
            step_lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            lr_scheduler = step_lr_schedulerlr_scheduler = step_lr_scheduler
        else:   
            lr_scheduler, _ = create_scheduler(args, optimizer)    
    
  
    # Define the loss scaler
    if args.loss_scaler:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None

    
    train_results = {'loss': [], 'lr': []}
    val_results = {'loss': []}
    best_loss_eval = 100000000
    
    for epoch in range(args.start_epoch, args.epochs):

        dataset = data_loader_train
        criterion = criterion_train        
        train_stats = engine_rec.train_step(encoder=encoder,
                                            decoder = decoder,
                                            dataloader=data_loader_train,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            device=device,
                                            epoch=epoch+1,
                                            lr_scheduler=lr_scheduler,
                                            combinations_csv = train_csv,
                                            args=args)
        
        if args.sched == 'step' or args.sched == 'exp':
            lr_scheduler.step() #step LR
        else:
            lr_scheduler.step(epoch+1) #others
       
        #removed wand
        results = engine_rec.evaluation(encoder=encoder,
                                        decoder = decoder,
                                        dataloader=data_loader_test,
                                        criterion=criterion,
                                        device=device,
                                        epoch=epoch+1,
                                        combinations_csv = test_csv,
                                        args=args)
        
        train_results['loss'].append(train_stats['loss'])
        train_results['lr'].append(train_stats['lr'])
        
        val_results['loss'].append(results['loss'])
        
        print("*****************EPOCH**************", epoch+1)
        
        #print("main:", train_stats, results)
        

        if results['loss'] < best_loss_eval:
            # Only want to save the best checkpoints if the best val bacc and the early stopping counter is less than the threshold
            best_loss_train = train_stats['loss']
            best_loss_eval = results['loss']
            best_epoch = epoch  # Update the best epoch here
            
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
    
    if best_loss_train:
        if not os.path.exists(output_dir):
            # If it doesn't exist, create the directory
            os.makedirs(output_dir)
        
        
        print(f"BEST CHECKPOINT Epoch {best_epoch + 1}:")
        print("train:", best_loss_train, "\n validation:", best_loss_eval)
        checkpoint_path = output_dir / 'encoder.pth'
        checkpoint_dict = {
            'model': encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': best_epoch,  # Use the best_epoch variable here
            'args': args,
        }
        if args.lr_scheduler:
            checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
        
        utils.save_on_master(checkpoint_dict, checkpoint_path)
        
        checkpoint_path = output_dir / 'decoder.pth'
        checkpoint_dict = {
            'model': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': best_epoch,  # Use the best_epoch variable here
            'args': args,
        }
        if args.lr_scheduler:
            checkpoint_dict['lr_scheduler'] = lr_scheduler.state_dict()
        utils.save_on_master(checkpoint_dict, checkpoint_path)
        


   
    """
    # Access the best model's state_dict and validation results
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

        # Print metrics for the best checkpoint
        best_train_loss = train_results['loss'][best_epoch]
        best_train_acc = train_results['acc'][best_epoch]
        print(f"BEST CHECKPOINT Epoch {best_epoch + 1}:")
        for combination, bacc_value in best_results.get('combination_bacc', {}).items():
            print(f"Combination: {combination} | Best Val BACC: {bacc_value:.4f}")
        
        
        print(f"Train Loss: {best_train_loss:.4f} | Train Acc: {best_train_acc:.4f} | "
              f"Val Loss: {best_results['loss']:.4f} | Val Acc: {best_results['acc1']:.4f} | "
              f"Val Bacc: {best_results['bacc']:.4f} | Val F1-Score: {np.mean(best_results['f1_score']):.4f}")
        print(f"[INFO] Best Val. Bacc: {(best_val_bacc * 100):.2f}% | [INFO] Saving model as 'best_checkpoint.pth'")
        """