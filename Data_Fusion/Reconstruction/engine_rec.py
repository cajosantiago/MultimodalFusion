import torch
import utils
from timm.utils import ModelEma
from datetime import datetime
import torch.nn as nn
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
import pandas as pd


def train_step(encoder, decoder, dataloader, criterion, optimizer, device, epoch, lr_scheduler, combinations_csv, args):
    encoder.train()
    decoder.train()
    train_loss = 0.0
    modalities_csv = load_combinations_csv(combinations_csv)
    results = {}

    for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
        CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
        CT, MR, feature_path, clingen = CT.float(), MR.float(), feature_path.float(), clingen.float()  # Cast input data to float
        
        #check if the modalities exist for the patients:
        combinations = get_modality_combination(patient_ids, modalities_csv)
        
        CT_feat, MR_feat, clingen_feat = sort_dimensions(CT, MR, clingen)
        #print("ORIGINAL:", CT_feat[0])
        # Forward pass through the encoder
        encoded_features = encoder(CT, MR, feature_path, clingen, combinations)
        
        # Forward pass through the decoder
        reconstructed_CT, reconstructed_MR, reconstructed_feature_path, reconstructed_clingen = decoder(encoded_features)
        #print("RECON:", reconstructed_CT[0])
        

        #get the weight for each loss depending if the modality exist for the patient or not
        weights_comb = get_combination_weights(combinations, modalities_csv)
        
        # Calculate reconstruction loss for each modality
        loss_CT_sum = 0
        loss_MR_sum = 0
        loss_feature_path_sum = 0
        loss_clingen_sum = 0
        CT_count = 0
        MR_count = 0
        Path_count = 0
        Clingen_count = 0
        
        for i in range(len(patient_ids)):
            if weights_comb[i][0] == 1:
                loss_CT_sum = loss_CT_sum + criterion(reconstructed_CT[i], CT_feat[i])
                CT_count += 1
            if weights_comb[i][1] == 1:
                loss_MR_sum = loss_MR_sum + criterion(reconstructed_MR[i], MR_feat[i])
                MR_count += 1
            if weights_comb[i][2] == 1:
                loss_feature_path_sum = loss_feature_path_sum + criterion(reconstructed_feature_path[i], feature_path[i])
                Path_count += 1
            if weights_comb[i][3] == 1:
                loss_clingen_sum = loss_clingen_sum + criterion(reconstructed_clingen[i], clingen_feat[i])
                Clingen_count += 1
            """
            print(i)
            print(weights_comb[i])
            print("\n LOSS CT:",criterion(reconstructed_CT[i], CT_feat[i]), 
                  "\n MR:",criterion(reconstructed_MR[i], MR_feat[i]), 
                  "\n Path:",criterion(reconstructed_feature_path[i], feature_path[i]) , 
                  "\n Clingen:",criterion(reconstructed_clingen[i], clingen_feat[i]))
            print("\n CT:",loss_CT_sum, "\n MR:",loss_MR_sum, "\n Path:",loss_feature_path_sum , "\n Clingen:",loss_clingen_sum)
            """
        
        # Calculate the average losses, handling division by zero
        loss_CT = safe_average_loss(loss_CT_sum, CT_count)
        loss_MR = safe_average_loss(loss_MR_sum, MR_count)
        loss_feature_path = safe_average_loss(loss_feature_path_sum, Path_count)
        loss_clingen = safe_average_loss(loss_clingen_sum, Clingen_count)
        #print("\n CT:",loss_CT, "\n MR:",loss_MR, "\n Path:",loss_feature_path , "\n Clingen:",loss_clingen)
        
        """
        loss_CT = criterion(reconstructed_CT, CT_feat)
        loss_MR = criterion(reconstructed_MR, MR_feat)
        loss_feature_path = criterion(reconstructed_feature_path, feature_path)
        loss_clingen = criterion(reconstructed_clingen, clingen_feat)
        """ 
        # Combine losses for all modalities
        total_loss = loss_CT + loss_MR + loss_feature_path + loss_clingen
        #print("Total:", total_loss)
        

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    # Average the loss over all batches
    train_loss /= len(dataloader)
    print("Train loss:", train_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    results['lr'] = current_lr
    results['loss'] = train_loss

    return results



def evaluation(encoder,
               decoder,
               dataloader: torch.utils.data.DataLoader, 
               criterion: nn.Module, 
               device: torch.device,
               epoch: int,
               combinations_csv,
               args=None):
    
    encoder.eval()
    decoder.eval()
    eval_loss = 0.0
    modalities_csv = load_combinations_csv(combinations_csv)
    results = {}

    with torch.no_grad():
        for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
            CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
            CT, MR, feature_path, clingen = CT.float(), MR.float(), feature_path.float(), clingen.float()  # Cast input data to float

            CT_feat, MR_feat, clingen_feat = sort_dimensions(CT, MR, clingen)
            
            #check if the modalities exist for the patients:
            combinations = get_modality_combination(patient_ids, modalities_csv)
            # Forward pass through the encoder
            encoded_features = encoder(CT, MR, feature_path, clingen, combinations)

            # Forward pass through the decoder
            reconstructed_CT, reconstructed_MR, reconstructed_feature_path, reconstructed_clingen = decoder(encoded_features)
            

            #get the weight for each loss depending if the modality exist for the patient or not
            weights_comb = get_combination_weights(combinations, modalities_csv)
            
            loss_CT_sum = 0
            loss_MR_sum = 0
            loss_feature_path_sum = 0
            loss_clingen_sum = 0
            CT_count = 0
            MR_count = 0
            Path_count = 0
            Clingen_count = 0

            for i in range(len(patient_ids)):
                if weights_comb[i][0] == 1:
                    loss_CT_sum = loss_CT_sum + criterion(reconstructed_CT[i], CT_feat[i])
                    CT_count += 1
                if weights_comb[i][1] == 1:
                    loss_MR_sum = loss_MR_sum + criterion(reconstructed_MR[i], MR_feat[i])
                    MR_count += 1
                if weights_comb[i][2] == 1:
                    loss_feature_path_sum = loss_feature_path_sum + criterion(reconstructed_feature_path[i], feature_path[i])
                    Path_count += 1
                if weights_comb[i][3] == 1:
                    loss_clingen_sum = loss_clingen_sum + criterion(reconstructed_clingen[i], clingen_feat[i])
                    Clingen_count += 1

            loss_CT = safe_average_loss(loss_CT_sum, CT_count)
            loss_MR = safe_average_loss(loss_MR_sum, MR_count)
            loss_feature_path = safe_average_loss(loss_feature_path_sum, Path_count)
            loss_clingen = safe_average_loss(loss_clingen_sum, Clingen_count)
            print("\n CT:",loss_CT, "\n MR:",loss_MR, "\n Path:",loss_feature_path , "\n Clingen:",loss_clingen)
            
            """
            loss_CT = criterion(reconstructed_CT, CT_feat)
            loss_MR = criterion(reconstructed_MR, MR_feat)
            loss_feature_path = criterion(reconstructed_feature_path, feature_path)
            loss_clingen = criterion(reconstructed_clingen, clingen_feat)
            """
            # Combine losses for all modalities
            total_loss = loss_CT + loss_MR + loss_feature_path + loss_clingen

            eval_loss += total_loss.item()

    # Average the loss over all batches
    eval_loss /= len(dataloader)
    print("Eval loss:", eval_loss)
    results['loss'] = eval_loss

    return results


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

def get_combination_weights(combinations, modalities_csv):
    weights_list = []
    for combination in combinations:
        weight_CT, weight_MR, weight_PATH, weight_CLINGEN = 1.0, 1.0, 1.0, 1.0

        # Update weights based on the combination, ignoring GEN
        if combination[0] == 0:
            weight_CT = 0.0
        if combination[1] == 0:
            weight_MR = 0.0
        if combination[2] == 0:
            weight_PATH = 0.0

        weights_list.append((weight_CT, weight_MR, weight_PATH, weight_CLINGEN))

    return weights_list

def load_combinations_csv(combinations_csv):
    if combinations_csv is not None:
        return pd.read_csv(combinations_csv)
    return None

def safe_average_loss(sum_loss, count):
    return sum_loss / count if count != 0 else 0