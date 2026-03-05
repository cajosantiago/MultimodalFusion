import torch
import utils
from timm.utils import ModelEma
from datetime import datetime

from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
import pandas as pd

def train_step(model, dataloader, criterion, optimizer, device, epoch, loss_scaler, lr_scheduler=None, combinations_csv=None, args=None):
    modalities_csv = load_combinations_csv(combinations_csv)
    combination_accuracy = {}
    modalities_stats = {}
    model.train()
    train_loss, train_acc = 0, 0
    train_stats = {}
    lr_num_updates = epoch * len(dataloader)
    
    pred_CT = []
    pred_MR = []
    pred_Path = []
    pred_Clingen = []
    label_CT = []
    label_MR = []
    label_Path = []
    label_Clingen = []

    for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
        label = label.float()
        label = label.view(-1, 1)
        CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
        
        combinations = get_modality_combination(patient_ids, modalities_csv)
        
        ct_output, mr_output, path_output, clingen_output, scores = model(CT, MR, feature_path, clingen, combinations)
        
      
        loss = criterion(scores, label)
        
        predictions = (torch.sigmoid(scores) > 0.5).float()
        
        for i, patient_id in enumerate(patient_ids):
                combination = combinations[i]

                if combination[0] == 1:
                    pred_CT.append(predictions[i].cpu().numpy())
                    label_CT.append(label[i].cpu().numpy())
                if combination[1] == 1:
                    pred_MR.append(predictions[i].cpu().numpy())
                    label_MR.append(label[i].cpu().numpy())
                if combination[2] == 1:
                    pred_Path.append(predictions[i].cpu().numpy())
                    label_Path.append(label[i].cpu().numpy())
                if combination[4] == 1:
                    pred_Clingen.append(predictions[i].cpu().numpy())
                    label_Clingen.append(label[i].cpu().numpy())
                #print(f"Combination for Patient {patient_id}: {combination}")

                # Calculate accuracy for this combination
                if combination not in combination_accuracy:
                    combination_accuracy[combination] = {"correct": 0, "total": 0}
                combination_accuracy[combination]["correct"] += (predictions[i] == label[i]).sum().item()
                combination_accuracy[combination]["total"] += len(predictions[i])
                
        
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.sched != 'exp' and args.sched != 'step' and not args.cosine_one_cycle:
            lr_scheduler.step_update(num_updates=lr_num_updates)



        #print("pred:",predictions)
        train_acc += (predictions == label).sum().item()/len(scores)
        #print("\n\n\n")
    
    # Adjust metrics to get average loss and accuracy per batch 
    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    

    
    train_stats['train_loss'] = train_loss
    train_stats['train_acc'] = train_acc
    train_stats['train_lr'] = current_lr
    train_stats['combination_accuracy'] = combination_accuracy
    
    modalities_stats['BACC_CT'] = balanced_accuracy_score(label_CT, pred_CT)
    modalities_stats['BACC_MR'] = balanced_accuracy_score(label_MR, pred_MR)
    modalities_stats['BACC_Path'] = balanced_accuracy_score(label_Path, pred_Path)
    modalities_stats['BACC_Clingen'] = balanced_accuracy_score(label_Clingen, pred_Clingen)
    
    modalities_stats['Recall_CT'] = recall_score(label_CT, pred_CT)
    modalities_stats['Recall_MR'] = recall_score(label_MR, pred_MR)
    modalities_stats['Recall_Path'] = recall_score(label_Path, pred_Path)
    modalities_stats['Recall_Clingen'] = recall_score(label_Clingen, pred_Clingen)
    
    modalities_stats['specificity_CT'] = calculate_specificity(pred_CT,label_CT)
    modalities_stats['specificity_MR'] = calculate_specificity(pred_MR,label_MR)
    modalities_stats['specificity_Path'] = calculate_specificity(pred_Path,label_Path)
    modalities_stats['specificity_Clingen'] = calculate_specificity(pred_Clingen,label_Clingen)
    
    """
    for combination, acc_data in combination_accuracy.items():
        accuracy = acc_data["correct"] / acc_data["total"]
        print(f"Accuracy for Combination {combination}: {accuracy}")
    """ 
        
    return modalities_stats, train_stats





def evaluation(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               criterion: torch.nn.Module, 
               device: torch.device,
               epoch: int,
               combinations_csv=None,
               args=None):
        
    # Switch to evaluation mode
    model.eval()    
    preds = []
    targets = []
    test_loss, test_acc = 0, 0
    results = {}
    combination_accuracy = {}
    modalities_stats = {}
    
    if combinations_csv is not None:
        modalities_csv = pd.read_csv(combinations_csv)
    else:
        modalities_csv = None
    
    # Create a dictionary to group patients by their combination
    combination_patients = {}
    pred_CT = []
    pred_MR = []
    pred_Path = []
    pred_Clingen = []
    label_CT = []
    label_MR = []
    label_Path = []
    label_Clingen = []
    
    for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
        # Initialize a dictionary to group predictions and labels by combination
        combination_predictions = {}
        combination_labels = {}
        
        label = label.float()
        label = label.view(-1, 1)
        CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
        
        #Get the modality combination for each patient
        combinations = get_modality_combination(patient_ids, modalities_csv)
        
        with torch.no_grad(), torch.inference_mode():
            
            CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
            ct_output, mr_output, path_output, clingen_output, scores = model(CT, MR, feature_path, clingen, combinations)
            predictions = (torch.sigmoid(scores) > 0.5).float()
           
            loss = criterion(scores, label)
            test_loss += loss.item()
            
            for i, patient_id in enumerate(patient_ids):
                combination = combinations[i]
                if combination[0] == 1:
                    pred_CT.append(predictions[i].cpu().numpy())
                    label_CT.append(label[i].cpu().numpy())
                if combination[1] == 1:
                    pred_MR.append(predictions[i].cpu().numpy())
                    label_MR.append(label[i].cpu().numpy())
                if combination[2] == 1:
                    pred_Path.append(predictions[i].cpu().numpy())
                    label_Path.append(label[i].cpu().numpy())
                if combination[4] == 1:
                    pred_Clingen.append(predictions[i].cpu().numpy())
                    label_Clingen.append(label[i].cpu().numpy())
                # Store patient predictions and labels in the corresponding combination group
                if combination not in combination_patients:
                    combination_patients[combination] = []
                combination_patients[combination].append((patient_id, predictions[i].cpu().numpy(), label[i].cpu().numpy()))

                # Calculate accuracy for this combination
                if combination not in combination_accuracy:
                    combination_accuracy[combination] = {"correct": 0, "total": 0}

                combination_accuracy[combination]["correct"] += (predictions[i] == label[i]).sum().item()
                combination_accuracy[combination]["total"] += len(predictions[i])


        test_acc += (predictions == label).sum().item() / len(scores)

        # Convert predictions and labels to NumPy arrays and append to lists
        preds.append(predictions.cpu().numpy())
        targets.append(label.cpu().numpy())
        
    # Concatenate the lists of predictions and labels to obtain NumPy arrays
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    #create text file with predictions
    utils.write_predictions_to_file(epoch, combination_patients)

    """
    # Print predictions and labels grouped by combination
    for combination, patients in combination_patients.items():
        print(f"Combination {combination}:")
        for patient_id, pred, true in patients:
            print(f"Patient ID: {patient_id}, Prediction: {pred}, True Label: {true}")
    """
    
    # Calculate BACC per combination
    combination_bacc = {}
    for combination, patients in combination_patients.items():
        comb_pred = []  # Clear comb_pred for each combination
        comb_label = []  # Clear comb_label for each combination

        for patient_id, pred, true in patients:
            comb_pred.append(pred)
            comb_label.append(true)

        # Check if there is only one class in predictions for this combination
        if len(set(tuple(map(tuple, comb_pred)))) == 1:
            # Handle the case where all predictions are the same class
            # You can choose to skip this combination or assign a default value
            # In this example, we'll assign a BACC of 0 for such cases
            bacc = 1
        else:
            bacc = balanced_accuracy_score(comb_label, comb_pred)

        #print(combination)
        #print("pred:", ", ".join(map(str, comb_pred)))
        #print("true:", ", ".join(map(str, comb_label)))
        #print("BACC:", bacc)

        combination_bacc[combination] = bacc

    # Calculate Balanced Accuracy (BACC) per combination
    results['confusion_matrix'] = confusion_matrix(targets, preds)
    results['f1_score'] = f1_score(targets, preds, average=None, zero_division=1) 
    results['precision'] = precision_score(targets, preds, average=None, zero_division=1)
    results['recall'] = recall_score(targets, preds, average=None)
    results['bacc'] = balanced_accuracy_score(targets, preds)
    results['acc1'], results['loss'] = test_acc, test_loss
    results['combination_bacc'] = combination_bacc  # Include BACC per combination
    results['combination_accuracy'] = combination_accuracy
    
    modalities_stats['BACC_CT'] = balanced_accuracy_score(label_CT, pred_CT)
    modalities_stats['BACC_MR'] = balanced_accuracy_score(label_MR, pred_MR)
    modalities_stats['BACC_Path'] = balanced_accuracy_score(label_Path, pred_Path)
    modalities_stats['BACC_Clingen'] = balanced_accuracy_score(label_Clingen, pred_Clingen)
    
    modalities_stats['Recall_CT'] = recall_score(label_CT, pred_CT)
    modalities_stats['Recall_MR'] = recall_score(label_MR, pred_MR)
    modalities_stats['Recall_Path'] = recall_score(label_Path, pred_Path)
    modalities_stats['Recall_Clingen'] = recall_score(label_Clingen, pred_Clingen)
    
    modalities_stats['specificity_CT'] = calculate_specificity(pred_CT,label_CT)
    modalities_stats['specificity_MR'] = calculate_specificity(pred_MR,label_MR)
    modalities_stats['specificity_Path'] = calculate_specificity(pred_Path,label_Path)
    modalities_stats['specificity_Clingen'] = calculate_specificity(pred_Clingen,label_Clingen)

    return modalities_stats, results


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


                
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

import pandas as pd

def load_combinations_csv(combinations_csv):
    if combinations_csv is not None:
        return pd.read_csv(combinations_csv)
    return None



def calculate_specificity(pred_CT, label_CT):
    # Ensure that pred_CT and label_CT have the same length
    if len(pred_CT) != len(label_CT):
        raise ValueError("Input lists must have the same length")

    # Initialize counters for TN and FP
    tn = 0  # True Negatives
    fp = 0  # False Positives

    # Calculate TN and FP
    for i in range(len(pred_CT)):
        if label_CT[i] == 0 and pred_CT[i] == 0:
            tn += 1
        elif label_CT[i] == 0 and pred_CT[i] == 1:
            fp += 1

    # Calculate specificity
    if tn + fp == 0:
        print(fp,tn)
        specificity = 0.0  # Avoid division by zero
    else:
        specificity = tn / (tn + fp)

    return specificity