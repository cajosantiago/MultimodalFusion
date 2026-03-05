import torch
import utils
from timm.utils import ModelEma
from datetime import datetime

from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, \
    balanced_accuracy_score
import pandas as pd

def train_step(models, dataloader, losses, optimizers, 
               device, epoch, loss_scaler, 
               lr_scheduler=None, combinations_csv=None, args=None):
    
    lr_num_updates = epoch * len(dataloader)
    train_loss, train_acc = 0, 0
    train_stats = {}
    modalities_stats = {}
    lr_num_updates = epoch * len(dataloader)
    modalities_csv = load_combinations_csv(combinations_csv)
    combination_accuracy = {}
    predictions_CT = []
    predictions_MR = []
    predictions_Path = []
    predictions_Clingen = []
    label_CT = []
    label_MR = []
    label_Path = []
    label_Clingen = []
    
 
    
    for model in models:
        model.train()
    
    train_loss_CT,train_loss_MR,train_loss_Path,train_loss_Clingen = 0,0,0,0
    
    #model.train()
    for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
        label = label.float()
        label = label.view(-1, 1)
        CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
        combinations = get_modality_combination(patient_ids, modalities_csv)
        model_outputs = []
        for batch_idx, (model, loss, optimizer) in enumerate(zip(models, losses, optimizers)):
            optimizer.zero_grad()

            # Choose the correct modality and forward pass
            if model == models[0]:  # CT model
                output = model(CT)
                loss_modality = loss(output, label)
                train_loss_CT += loss_modality.item()
                for i, combination in enumerate(combinations):
                    if combination[0] == 1:
                        prediction = (torch.sigmoid(output[i]) > 0.5).float()
                        predictions_CT.append(prediction)
                        label_CT.append(label[i])
                model_outputs.append(output)

            elif model == models[1]:  # MR model
                output = model(MR)
                loss_modality = loss(output, label)
                train_loss_MR += loss_modality.item()
                for i, combination in enumerate(combinations):
                    if combination[1] == 1:
                        prediction = (torch.sigmoid(output[i]) > 0.5).float()
                        predictions_MR.append(prediction)
                        label_MR.append(label[i])
                model_outputs.append(output)

            elif model == models[2]:  # Path model
                output = model(feature_path)
                loss_modality = loss(output, label)
                train_loss_Path += loss_modality.item()
                for i, combination in enumerate(combinations):
                    if combination[2] == 1:
                        prediction = (torch.sigmoid(output[i]) > 0.5).float()
                        predictions_Path.append(prediction)
                        label_Path.append(label[i])
                model_outputs.append(output)

            elif model == models[3]:  # Clingen model
                output = model(clingen)
                loss_modality = loss(output, label)
                train_loss_Clingen += loss_modality.item()
                for i, combination in enumerate(combinations):
                    if combination[4] == 1:
                        prediction = (torch.sigmoid(output[i]) > 0.5).float()
                        predictions_Clingen.append(prediction)
                        label_Clingen.append(label[i])
                model_outputs.append(output)

            loss_modality.backward()
            optimizer.step()
        
        
        #Calculate detection rates
        detection_rates = calculate_detection_rates(model_outputs, label)
        weights_detection = calculate_weights(detection_rates)
        weighted_prob_vectors = [weights_detection[i] * model_outputs[i] for i in range(len(model_outputs))]
        scores = torch.sum(torch.stack(weighted_prob_vectors), dim=0)
        predictions = (torch.sigmoid(scores) > 0.5).float()
        train_acc += (predictions == label).sum().item() / len(scores)
        if modalities_csv is not None:
            for i, combination in enumerate(combinations):
                if combination not in combination_accuracy:
                    combination_accuracy[combination] = {"correct": 0, "total": 0}
                combination_accuracy[combination]["correct"] += (predictions[i] == label[i]).sum().item()
                combination_accuracy[combination]["total"] += len(predictions[i])
                
        if args.sched != 'exp' and args.sched != 'step' and not args.cosine_one_cycle:
            lr_scheduler.step_update(num_updates=lr_num_updates)
    
    current_lr = optimizer.param_groups[0]['lr']
    # Calculate BACC and Recall for each modality
    bacc_CT = balanced_accuracy_score(torch.cat(label_CT).cpu().numpy(), torch.cat(predictions_CT).cpu().numpy())
    recall_CT = recall_score(torch.cat(label_CT).cpu().numpy(), torch.cat(predictions_CT).cpu().numpy())
    print(f"CT Modality - BACC: {bacc_CT}, Recall: {recall_CT}")

    bacc_MR = balanced_accuracy_score(torch.cat(label_MR).cpu().numpy(), torch.cat(predictions_MR).cpu().numpy())
    recall_MR = recall_score(torch.cat(label_MR).cpu().numpy(), torch.cat(predictions_MR).cpu().numpy())
    print(f"MR Modality - BACC: {bacc_MR}, Recall: {recall_MR}")

    bacc_Path = balanced_accuracy_score(torch.cat(label_Path).cpu().numpy(), torch.cat(predictions_Path).cpu().numpy())
    recall_Path = recall_score(torch.cat(label_Path).cpu().numpy(), torch.cat(predictions_Path).cpu().numpy())
    print(f"Path Modality - BACC: {bacc_Path}, Recall: {recall_Path}")

    bacc_Clingen = balanced_accuracy_score(torch.cat(label_Clingen).cpu().numpy(), torch.cat(predictions_Clingen).cpu().numpy())
    recall_Clingen = recall_score(torch.cat(label_Clingen).cpu().numpy(), torch.cat(predictions_Clingen).cpu().numpy())
    print(f"Clingen Modality - BACC: {bacc_Clingen}, Recall: {recall_Clingen}")    
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Store the total loss for each modality in the train_stats dictionary
    train_stats['train_loss_CT'] = train_loss_CT
    train_stats['train_loss_MR'] = train_loss_MR
    train_stats['train_loss_Path'] = train_loss_Path
    train_stats['train_loss_Clingen'] = train_loss_Clingen
    train_stats['train_loss'] = train_loss
    train_stats['train_acc'] = train_acc
    train_stats['train_lr'] = current_lr
    train_stats['combination_accuracy'] = combination_accuracy
    
    modalities_stats['BACC_CT'] = bacc_CT
    modalities_stats['BACC_MR'] = bacc_MR
    modalities_stats['BACC_Path'] = bacc_Path
    modalities_stats['BACC_Clingen'] = bacc_Clingen
    modalities_stats['Recall_CT'] = recall_CT
    modalities_stats['Recall_MR'] = recall_MR
    modalities_stats['Recall_Path'] = recall_Path
    modalities_stats['Recall_Clingen'] = recall_Clingen
    

    return modalities_stats, train_stats





def evaluation(models,
               dataloader: torch.utils.data.DataLoader, 
               losses,
               device: torch.device,
               epoch: int,
               combinations_csv=None,
               args=None):
    eval_loss = 0
    eval_acc = 0
    results = {}
    modalities_stats = {}
    modalities_csv = load_combinations_csv(combinations_csv)
    combination_accuracy = {}
    predictions_CT = []
    predictions_MR = []
    predictions_Path = []
    predictions_Clingen = []
    label_CT = []
    label_MR = []
    label_Path = []
    label_Clingen = []
    preds = []
    targets = []
    combination_patients = {}

    for model in models:
        model.eval()  # Switch to evaluation mode

    eval_loss_CT, eval_loss_MR, eval_loss_Path, eval_loss_Clingen = 0, 0, 0, 0

    with torch.no_grad(), torch.inference_mode():  # Disable gradient calculation for evaluation
        for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
            label = label.float()
            label = label.view(-1, 1)
            CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
            combinations = get_modality_combination(patient_ids, modalities_csv)

            model_outputs = []
            for model, loss in zip(models, losses):
                # Forward pass
                if model == models[0]:  # CT model
                    output = model(CT)
                    loss_modality = loss(output, label)
                    eval_loss_CT += loss_modality.item()
                    for i, combination in enumerate(combinations):
                        if combination[0] == 1:
                            prediction = (torch.sigmoid(output[i]) > 0.5).float()
                            predictions_CT.append(prediction)
                            label_CT.append(label[i])
                    model_outputs.append(output)
                if model == models[1]:  # MR model
                    output = model(MR)
                    loss_modality = loss(output, label)
                    eval_loss_MR += loss_modality.item()
                    for i, combination in enumerate(combinations):
                        if combination[1] == 1:
                            prediction = (torch.sigmoid(output[i]) > 0.5).float()
                            predictions_MR.append(prediction)
                            label_MR.append(label[i])
                    model_outputs.append(output)
                if model == models[2]:  # Path model
                    output = model(feature_path)
                    loss_modality = loss(output, label)
                    eval_loss_Path += loss_modality.item()
                    for i, combination in enumerate(combinations):
                        if combination[2] == 1:
                            prediction = (torch.sigmoid(output[i]) > 0.5).float()
                            predictions_Path.append(prediction)
                            label_Path.append(label[i])
                    model_outputs.append(output)
                if model == models[3]:  # Clingen model
                    output = model(clingen)
                    loss_modality = loss(output, label)
                    eval_loss_Clingen += loss_modality.item()
                    for i, combination in enumerate(combinations):
                        if combination[4] == 1:
                            prediction = (torch.sigmoid(output[i]) > 0.5).float()
                            predictions_Clingen.append(prediction)
                            label_Clingen.append(label[i])
                    model_outputs.append(output)


                # Repeat similar code for MR, Path, and Clingen models

            # Calculate detection rates
            detection_rates = calculate_detection_rates(model_outputs, label)
            weights_detection = calculate_weights(detection_rates)
            weighted_prob_vectors = [weights_detection[i] * model_outputs[i] for i in range(len(model_outputs))]
            scores = torch.sum(torch.stack(weighted_prob_vectors), dim=0)
            predictions = (torch.sigmoid(scores) > 0.5).float()
            eval_acc += (predictions == label).sum().item() / len(scores)
            preds.append(predictions.cpu().numpy())
            targets.append(label.cpu().numpy())
            
            if modalities_csv is not None:
                combinations = get_modality_combination(patient_ids, modalities_csv)
                for i, patient_id in enumerate(patient_ids):
                    combination = combinations[i]
                    
                    # Store patient predictions and labels in the corresponding combination group
                    if combination not in combination_patients:
                        combination_patients[combination] = []
                    combination_patients[combination].append((patient_id, predictions[i].cpu().numpy(), label[i].cpu().numpy()))
                    
                    # Calculate accuracy for this combination
                    if combination not in combination_accuracy:
                        combination_accuracy[combination] = {"correct": 0, "total": 0}
                    
                    combination_accuracy[combination]["correct"] += (predictions[i] == label[i]).sum().item()
                    combination_accuracy[combination]["total"] += len(predictions[i])

    eval_loss = eval_loss / len(dataloader)
    eval_acc = eval_acc / len(dataloader)
    
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
            # In this example, we'll assign a BACC of 0.5 for such cases
            bacc = 0.5
        else:
            bacc = balanced_accuracy_score(comb_pred, comb_label)

        #print(combination)
        #print("pred:", ", ".join(map(str, comb_pred)))
        #print("true:", ", ".join(map(str, comb_label)))
        #print("BACC:", bacc)

        combination_bacc[combination] = bacc

    #create text file with predictions
    utils.write_predictions_to_file(epoch, combination_patients)
    
    # Concatenate the lists of predictions and labels to obtain NumPy arrays
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    print("\n **VALIDATION**")
    bacc_CT = balanced_accuracy_score(torch.cat(label_CT).cpu().numpy(), torch.cat(predictions_CT).cpu().numpy())
    recall_CT = recall_score(torch.cat(label_CT).cpu().numpy(), torch.cat(predictions_CT).cpu().numpy())
    print(f"CT Modality - BACC: {bacc_CT}, Recall: {recall_CT}")

    bacc_MR = balanced_accuracy_score(torch.cat(label_MR).cpu().numpy(), torch.cat(predictions_MR).cpu().numpy())
    recall_MR = recall_score(torch.cat(label_MR).cpu().numpy(), torch.cat(predictions_MR).cpu().numpy())
    print(f"MR Modality - BACC: {bacc_MR}, Recall: {recall_MR}")

    bacc_Path = balanced_accuracy_score(torch.cat(label_Path).cpu().numpy(), torch.cat(predictions_Path).cpu().numpy())
    recall_Path = recall_score(torch.cat(label_Path).cpu().numpy(), torch.cat(predictions_Path).cpu().numpy())
    print(f"Path Modality - BACC: {bacc_Path}, Recall: {recall_Path}")

    bacc_Clingen = balanced_accuracy_score(torch.cat(label_Clingen).cpu().numpy(), torch.cat(predictions_Clingen).cpu().numpy())
    recall_Clingen = recall_score(torch.cat(label_Clingen).cpu().numpy(), torch.cat(predictions_Clingen).cpu().numpy())
    print(f"Clingen Modality - BACC: {bacc_Clingen}, Recall: {recall_Clingen}")    




    
    # Calculate Balanced Accuracy (BACC) per combination
    results['confusion_matrix'] = confusion_matrix(targets, preds)
    results['f1_score'] = f1_score(targets, preds, average=None, zero_division=1) 
    results['precision'] = precision_score(targets, preds, average=None, zero_division=1)
    results['recall'] = recall_score(targets, preds, average=None)
    results['bacc'] = balanced_accuracy_score(targets, preds)
    results['acc1'], results['loss'] = eval_acc, eval_loss
    results['combination_bacc'] = combination_bacc  # Include BACC per combination
    results['combination_accuracy'] = combination_accuracy
    results['eval_loss_CT'] = eval_loss_CT
    results['eval_loss_MR'] = eval_loss_MR
    results['eval_loss_Path'] = eval_loss_Path
    results['eval_loss_Clingen'] = eval_loss_Clingen
    
    
    
    modalities_stats['BACC_CT'] = bacc_CT
    modalities_stats['BACC_MR'] = bacc_MR
    modalities_stats['BACC_Path'] = bacc_Path
    modalities_stats['BACC_Clingen'] = bacc_Clingen
    modalities_stats['Recall_CT'] = recall_CT
    modalities_stats['Recall_MR'] = recall_MR
    modalities_stats['Recall_Path'] = recall_Path
    modalities_stats['Recall_Clingen'] = recall_Clingen

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

def calculate_weights(detection_rates):
    # Calculate weights based on detection rates
    weights = [1 - dr for dr in detection_rates]
    return weights

def weighted_late_fusion(model_outputs, weights):
    # Weighted late fusion
    fused_output = torch.zeros_like(model_outputs[0])
    for i, output in enumerate(model_outputs):
        fused_output += weights[i] * output
    return fused_output

def calculate_detection_rates(model_outputs, labels):
    # Initialize lists to store detection rates for each modality
    detection_rates = []

    # Iterate through model outputs and labels
    for output in model_outputs:
        prediction = (torch.sigmoid(output) > 0.5).float()
        
        # Calculate true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)
        TP = ((prediction == 1) & (labels == 1)).sum().item()
        TN = ((prediction == 0) & (labels == 0)).sum().item()
        FP = ((prediction == 1) & (labels == 0)).sum().item()
        FN = ((prediction == 0) & (labels == 1)).sum().item()
        
        # Calculate detection rate (DR) using the formula: DR = TN / (TP + TN + FP + FN)
        DR = TP / (TP + TN + FP + FN)

        # Append the detection rate to the list
        detection_rates.append(DR)

    return detection_rates