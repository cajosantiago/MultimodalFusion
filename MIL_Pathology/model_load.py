import utils, data_load, mil_tiago
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
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os


def get_args_parser():
   
    parser = argparse.ArgumentParser('Early Fusion', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--pin_mem', default=True, type=bool, help='pin memory')
    
    ##Training parameters
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (default: 0.0)')

    
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Load', parents=[get_args_parser()])
    args = parser.parse_args()
    
    
    # Print arguments
    print("----------------- Args -------------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("------------------------------------------\n")
 
    device = args.gpu if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    ################### DATA SETUP ########################
    patient_list = 'Files/patient_list.csv'
    CT_scans = 'Files/pathology_dataset.csv'

    noise = 0
    #Defining the dataloaders
    dataset_CT = data_load.FeatureDataset(patient_list, CT_scans, bag_size=8, noise = noise)
    data_loader_CT = DataLoader(dataset_CT, batch_size=1, 
                             shuffle=True, num_workers=args.num_workers, 
                             pin_memory=args.pin_mem, drop_last=False)
        

    model = mil_tiago.InstanceMIL(num_classes=2, 
                                N=8,
                                dropout=args.dropout,
                                pooling_type= None,
                                device=device,
                                args=args,
                                patch_extractor=None)
    
        
    # Replace 'path/to/MIL-instance-max-aLaDeit-best_checkpoint.pth' with the actual file path
    checkpoint_path = 'model_x/best_checkpoint_27.pth'
    checkpoint = torch.load(checkpoint_path)

    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    patient_position_data = []
    with torch.no_grad():
          for batch_idx, (inputs, labels, patient, numb_fmap) in enumerate(data_loader_CT):
                print(batch_idx)
                # Send data to target device
                labels = labels.float()
                #IF PATIENT SURVIVES - 0
                #IF PATIENT DIES - 1
                inputs, labels = inputs.to(device), labels.to(device)
                numb_fmap = numb_fmap.to(device)
                #with torch.cuda.amp.autocast():
                scores, position = model(inputs, numb_fmap) # 2.Forward pass
                for pat, pos in zip(patient, position):
                    patient_position_data.append((pat, pos.item()))
                
    
    df = pd.DataFrame(patient_position_data, columns=['case_id', 'chosen_exam'])
    df = df.sort_values(by='case_id')
    csv_file_path = "chosen_exams/patient_position_data.csv"
    df.to_csv(csv_file_path, index=False)