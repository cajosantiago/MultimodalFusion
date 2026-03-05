import utils, data_load, enc_dec
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
import umap
import seaborn as sns
from typing import List, Union
import pandas as pd
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler, CosineLRScheduler
from timm.utils import get_state_dict, ModelEma, NativeScaler
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os

def sort_dimensions(CT, MR, clingen):
    CT_feat = CT.squeeze(dim=3)
    MR_feat = MR.squeeze(dim=3)
    #print("2", CT_feat.shape)
    # (2) Convert x to shape (Batch_size, embedding_size)
    CT_feat = CT_feat.reshape(-1, CT_feat.size(2)) #(1,512)
    MR_feat = MR_feat.reshape(-1, MR_feat.size(2)) #(1,512)
    CG_feat = clingen.reshape(-1, clingen.size(2)) #(1,17)
    #print("3", CT_feat.shape)
    
    return CT_feat, MR_feat, CG_feat

def get_modality_combination(patient_ids, df):
    combinations = []

    for patient_id in patient_ids:
        # Check if the patient_id exists in the DataFrame
        if patient_id in df['Subject ID'].values:
            patient_data = df[df['Subject ID'] == patient_id].iloc[0]  # Get the row for the patient
            modalities = ['CT', 'MR', 'pathology', 'genetic', 'clinical']
            combination = [1 if patient_data[modality] == 1 else 0 for modality in modalities]
            combinations.append(tuple(combination))
        else:
            # Handle the case where patient_id does not exist
            print(f"Patient ID {patient_id} not found in the DataFrame.")
            combinations.append(None)  # or handle it in a way that makes sense for your task

    return combinations



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
    
    gen_dataset = 'Files/concatenated.csv'
    train_csv = 'Files/train.csv'
    test_csv = 'Files/test.csv'
    CT_files = 'Files/patients_with_labels_CT.csv'
    MR_files = 'Files/patients_with_labels_MR.csv'
    PATH_files = 'Files/slide_location.csv'
    clinical = 'Files/encoded_three_updated.csv'
    CT_choice = 'Files/CT_position_FINAL.csv'
    MR_choice = 'Files/MR_position_FINAL.csv'
    PATH_choice = 'Files/Path_position_FINAL.csv'
    modalities_csv = pd.read_csv(gen_dataset)

    train = False
    noise = False
    dataset_test = data_load.FeatureDataset(gen_dataset, CT_files, MR_files, PATH_files, clinical, CT_choice, MR_choice, PATH_choice, noise = noise, train = train)
    dataloader = DataLoader(dataset_test, batch_size=1, 
                         shuffle=False, num_workers=args.num_workers, 
                         pin_memory=args.pin_mem, drop_last=False)
        

    #Defining the model
    encoder = enc_dec.Encoder(args=args)
    decoder = enc_dec.Decoder(args=args)
    
    checkpoint_path_enc = 'enc_dc/experiment_18/encoder.pth'
    checkpoint_enc = torch.load(checkpoint_path_enc)
    
    checkpoint_path_dc = 'enc_dc/experiment_18/decoder.pth'
    checkpoint_dc = torch.load(checkpoint_path_dc)

    # Load the state dictionary into the model
    encoder.load_state_dict(checkpoint_enc['model'])
    encoder.to(device)
    encoder.eval()
    decoder.load_state_dict(checkpoint_dc['model'])
    decoder.to(device)
    decoder.eval()
    
    output_dir = "plots/UMAP/test"
    if not os.path.exists(output_dir):
        # If it doesn't exist, create the directory
        os.makedirs(output_dir)
      
    output_dir_MR = 'recons_MR'
    output_dir_CT = 'recons_CT'    
    os.makedirs(output_dir_MR, exist_ok=True)
    os.makedirs(output_dir_CT, exist_ok=True)
    
    # Create data storage for CSV
    csv_data_MR = []
    csv_data_CT = []

    with torch.no_grad():
        all_CT_feat = []
        all_reconstructed_CT = []
        all_MR_feat = []  # Accumulate MR_feat data from all batches
        all_reconstructed_MR = []  # Accumulate reconstructed_MR data from all batches
        all_feature_path_feat = []  # Accumulate feature_path_feat data from all batches
        all_reconstructed_feature_path = []  # Accumulate reconstructed_feature_path data from all batches
        all_clingen_feat = []  # Accumulate clingen_feat data from all batches
        all_reconstructed_clingen = []  # Accumulate reconstructed_clingen data from all batches
        all_labels = []  # Accumulate labels from all batches
        new_CT = []
        new_MR = []
        
        for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
            
            CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
            CT, MR, feature_path, clingen = CT.float(), MR.float(), feature_path.float(), clingen.float()  # Cast input data to float
            feature_path_feat = feature_path
            
            combinations = get_modality_combination(patient_ids, modalities_csv)
            
            CT_feat, MR_feat, clingen_feat = sort_dimensions(CT, MR, clingen)
            
            # Forward pass through the encoder
            encoded_features = encoder(CT, MR, feature_path, clingen, combinations)

            # Forward pass through the decoder
            reconstructed_CT, reconstructed_MR, reconstructed_feature_path, reconstructed_clingen = decoder(encoded_features)

            # Convert the tensors to NumPy arrays for UMAP
            CT_feat_np = CT_feat.cpu().numpy().reshape(len(CT_feat), -1)
            MR_feat_np = MR_feat.cpu().numpy().reshape(len(MR_feat), -1)
            feature_path_feat_np = feature_path_feat.cpu().numpy().reshape(len(feature_path_feat), -1)
            clingen_feat_np = clingen_feat.cpu().numpy().reshape(len(clingen_feat), -1)

            reconstructed_CT_np = reconstructed_CT.cpu().numpy().reshape(len(reconstructed_CT), -1)
            reconstructed_MR_np = reconstructed_MR.cpu().numpy().reshape(len(reconstructed_MR), -1)
            reconstructed_feature_path_np = reconstructed_feature_path.cpu().numpy().reshape(len(reconstructed_feature_path), -1)
            reconstructed_clingen_np = reconstructed_clingen.cpu().numpy().reshape(len(reconstructed_clingen), -1)
            
            # Accumulate data from the current batch
            for i, patient_combinations in enumerate(combinations):
                if patient_combinations[0] == 1:
                    all_CT_feat.append(CT_feat_np)
                    all_reconstructed_CT.append(reconstructed_CT_np)
                else:
                    new_CT.append(reconstructed_CT_np)
                    patient_id = patient_ids[i]
                    filename = os.path.join(output_dir_CT, f'{patient_id}.npz')
                    np.savez(filename, reconstructed_CT_np)
                    csv_data_CT.append({'case_id': patient_id, 'file_location': filename})
                
                
                if patient_combinations[1] == 1:
                    all_MR_feat.append(MR_feat_np)
                    all_reconstructed_MR.append(reconstructed_MR_np)
                else:
                    new_MR.append(reconstructed_MR_np)
                    patient_id = patient_ids[i]
                    filename = os.path.join(output_dir_MR, f'{patient_id}.npz')
                    np.savez(filename, reconstructed_MR_np)
                    csv_data_MR.append({'case_id': patient_id, 'file_location': filename})
                
                if patient_combinations[2] == 1:
                    all_feature_path_feat.append(feature_path_feat_np)
                    all_reconstructed_feature_path.append(reconstructed_feature_path_np)
                else:
                    print("path")
                if patient_combinations[4] == 1:
                    all_clingen_feat.append(clingen_feat_np)
                    all_reconstructed_clingen.append(reconstructed_clingen_np)
                else:
                    print("clingen")
            
            all_labels.append(label.cpu().numpy())

        # Concatenate data from all batches
        all_CT_feat = np.concatenate(all_CT_feat, axis=0)
        all_MR_feat = np.concatenate(all_MR_feat, axis=0)
        all_feature_path_feat = np.concatenate(all_feature_path_feat, axis=0)
        all_clingen_feat = np.concatenate(all_clingen_feat, axis=0)

        all_reconstructed_CT = np.concatenate(all_reconstructed_CT, axis=0)
        all_reconstructed_MR = np.concatenate(all_reconstructed_MR, axis=0)
        all_reconstructed_feature_path = np.concatenate(all_reconstructed_feature_path, axis=0)
        all_reconstructed_clingen = np.concatenate(all_reconstructed_clingen, axis=0)

        all_labels = np.concatenate(all_labels, axis=0)
        
        new_CT = np.concatenate(new_CT, axis = 0)
        new_MR = np.concatenate(new_MR, axis = 0)

        # Perform UMAP on the accumulated data for each feature
        #umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        # Create separate UMAP models for each modality
        umap_model_CT = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_model_MR = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_model_path = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_model_clingen = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)


        umap_CT_feat = umap_model_CT.fit_transform(all_CT_feat)
        umap_MR_feat = umap_model_MR.fit_transform(all_MR_feat)
        umap_feature_path_feat = umap_model_path.fit_transform(all_feature_path_feat)
        umap_clingen_feat = umap_model_clingen.fit_transform(all_clingen_feat)

        umap_reconstructed_CT = umap_model_CT.fit_transform(all_reconstructed_CT)
        umap_reconstructed_MR = umap_model_MR.fit_transform(all_reconstructed_MR)
        umap_reconstructed_feature_path = umap_model_path.fit_transform(all_reconstructed_feature_path)
        umap_reconstructed_clingen = umap_model_clingen.fit_transform(all_reconstructed_clingen)
        
        umap_new_ct = umap_model_CT.fit_transform(new_CT)
        umap_new_mr = umap_model_MR.fit_transform(new_MR)
    
    # Create pandas DataFrames for MR and CT CSV data
    df_MR = pd.DataFrame(csv_data_MR)
    df_CT = pd.DataFrame(csv_data_CT)

    # Save the DataFrames to separate CSV files for MR and CT
    csv_file_path_MR = 'reconstucted_MR.csv'
    csv_file_path_CT = 'reconstructed_CT.csv'
    df_MR.to_csv(csv_file_path_MR, index=False)
    df_CT.to_csv(csv_file_path_CT, index=False)

           
    # Create a single plot for CT data without labels
    plt.figure(figsize=(12, 6))
    # Plot original CT_feat data
    sns.scatterplot(x=umap_CT_feat[:, 0], y=umap_CT_feat[:, 1], palette='viridis', label='Original CT_feat')
    # Plot reconstructed CT data
    sns.scatterplot(x=umap_reconstructed_CT[:, 0], y=umap_reconstructed_CT[:, 1], palette='plasma', label='Reconstructed CT')
    #Plot new generated CT data
    sns.scatterplot(x=umap_new_ct[:, 0], y=umap_new_ct[:, 1], label='New CT')
    plt.title('UMAP Projection of Original, Reconstructed and Generated CT')
    plt.legend()
    # Save the UMAP plot for CT as an image
    
    output_file_CT = os.path.join(output_dir, 'umap_combined_CT.png')
    plt.savefig(output_file_CT)
    plt.close()  # Close the plot to release resources

    # Create a single plot for MR data without labels
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=umap_MR_feat[:, 0], y=umap_MR_feat[:, 1], palette='viridis', label='Original MR_feat')
    sns.scatterplot(x=umap_reconstructed_MR[:, 0], y=umap_reconstructed_MR[:, 1], palette='plasma', label='Reconstructed MR')
    sns.scatterplot(x=umap_new_mr[:, 0], y=umap_new_mr[:, 1], label='New MR')
    plt.title('UMAP Projection of Original, Reconstructed MR and Generated MR')
    # Save the UMAP plot for MR as an image
    output_file_MR = os.path.join(output_dir, 'umap_combined_MR.png')
    plt.savefig(output_file_MR)
    plt.close()  # Close the plot to release resources
    
    # Create a single plot for Path data without labels
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=umap_feature_path_feat[:, 0], y=umap_feature_path_feat[:, 1], palette='viridis', label='Original Path_feat')
    sns.scatterplot(x=umap_reconstructed_feature_path[:, 0], y=umap_reconstructed_feature_path[:, 1], palette='plasma', label='Reconstructed Path')
    plt.title('UMAP Projection of Original and Reconstructed Path')
    # Save the UMAP plot for MR as an image
    output_file_path = os.path.join(output_dir, 'umap_combined_Path.png')
    plt.savefig(output_file_path)
    plt.close()  # Close the plot to release resources
    
        # Create a single plot for MR data without labels
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=umap_clingen_feat[:, 0], y=umap_clingen_feat[:, 1], palette='viridis', label='Original Clingen_feat')
    sns.scatterplot(x=umap_reconstructed_clingen[:, 0], y=umap_reconstructed_clingen[:, 1], palette='plasma', label='Reconstructed Clingen')
    plt.title('UMAP Projection of Original and Reconstructed Clingen')
    # Save the UMAP plot for MR as an image
    output_file_clingen = os.path.join(output_dir, 'umap_combined_clingen.png')
    plt.savefig(output_file_clingen)
    plt.close()  # Close the plot to release resources

    
    




