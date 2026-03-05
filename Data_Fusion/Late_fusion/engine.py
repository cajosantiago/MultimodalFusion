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
    
    modality_data = {
        "CT": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },
        "MR": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },
        "Path": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },
        "Clingen": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },        
    }

    fusion_data = {
        "Final": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },
    }

    for model in models:
        model.train()
    
    train_loss_CT,train_loss_MR,train_loss_Path,train_loss_Clingen = 0,0,0,0
    
    #model.train()
    for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
        label = label.float()
        label = label.view(-1, 1)
        CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
        combinations = get_modality_combination(patient_ids, modalities_csv)
        
        input_CT = []
        input_MR = []
        label_insta_CT = []
        label_insta_MR = []
        label_insta_Path = []
        label_insta_Clingen = []
        

        for batch_idx, (model, loss, optimizer) in enumerate(zip(models, losses, optimizers)):
            optimizer.zero_grad()
            loss_modality = None
            indices_to_append = []
            # Choose the correct modality and forward pass
            if model == models[0]:  # CT model
                for i, combination in enumerate(combinations):
                    if combination[0] == 1:
                        input_CT.append(CT[i])
                        label_insta_CT.append(label[i])
                        indices_to_append.append(i)

                if input_CT and label_insta_CT:
                    output = model(torch.cat(input_CT))
                    output = output.squeeze(dim=1)
                    loss_modality = loss(output, torch.cat(label_insta_CT))
                    train_loss_CT += loss_modality.item()
                    prediction = (torch.sigmoid(output) > 0.5).float()

                    # Loop through the patients in the batch and store data in the dictionary
                    for i, index in enumerate(indices_to_append):
                        modality_data["CT"]["patient_id"].append(patient_ids[index])
                        modality_data["CT"]["combination"].append(combinations[index])
                        modality_data["CT"]["label"].append(label_insta_CT[i].item())
                        modality_data["CT"]["output"].append(output[i].item())
                        modality_data["CT"]["prediction"].append(prediction[i].item())

            elif model == models[1]:  # MR model
                for i, combination in enumerate(combinations):
                    if combination[1] == 1:
                        input_MR.append(MR[i])
                        label_insta_MR.append(label[i])
                        indices_to_append.append(i)
                        
                if input_MR and label_insta_MR:
                    output = model(torch.cat(input_MR))
                    output = output.squeeze(dim=1)
                    loss_modality = loss(output, torch.cat(label_insta_MR))
                    train_loss_MR += loss_modality.item()
                    prediction = (torch.sigmoid(output) > 0.5).float()
                    
                    for i, index in enumerate(indices_to_append):
                        modality_data["MR"]["patient_id"].append(patient_ids[index])
                        modality_data["MR"]["combination"].append(combinations[index])
                        modality_data["MR"]["label"].append(label_insta_MR[i].item())
                        modality_data["MR"]["output"].append(output[i].item())
                        modality_data["MR"]["prediction"].append(prediction[i].item())
                    

            elif model == models[2]:  # Path model
                output = model(feature_path)
                loss_modality = loss(output, label)
                train_loss_Path += loss_modality.item()
                prediction = (torch.sigmoid(output) > 0.5).float()
                for i, patient_id in enumerate(patient_ids):
                    modality_data["Path"]["patient_id"].append(patient_id)
                    modality_data["Path"]["combination"].append(combinations[i])
                    modality_data["Path"]["label"].append(label[i].item())
                    modality_data["Path"]["output"].append(output[i].item())
                    modality_data["Path"]["prediction"].append(prediction[i].item())

            elif model == models[3]:  # Clingen model
                output = model(clingen)
                loss_modality = loss(output, label)
                train_loss_Clingen += loss_modality.item()
                prediction = (torch.sigmoid(output) > 0.5).float()
                for i, patient_id in enumerate(patient_ids):
                    modality_data["Clingen"]["patient_id"].append(patient_id)
                    modality_data["Clingen"]["combination"].append(combinations[i])
                    modality_data["Clingen"]["label"].append(label[i].item())
                    modality_data["Clingen"]["output"].append(output[i].item())
                    modality_data["Clingen"]["prediction"].append(prediction[i].item())
            
            if loss_modality is not None:
                loss_modality.backward()
                optimizer.step()
                
        if args.sched != 'exp' and args.sched != 'step' and not args.cosine_one_cycle:
            lr_scheduler.step_update(num_updates=lr_num_updates)


    bacc_CT, recall_CT, specificity_CT = calculate_metrics_for_modality(modality_data, "CT")
    #print(f"CT Modality - BACC: {bacc_CT:.5f}, Recall: {recall_CT:.5f}, Specificity: {specificity_CT:.5f}")
    bacc_MR, recall_MR, specificity_MR = calculate_metrics_for_modality(modality_data, "MR")
    #print(f"MR Modality - BACC: {bacc_MR:.5f}, Recall: {recall_MR:.5f}, Specificity: {specificity_MR:.5f}")
    bacc_Path, recall_Path, specificity_Path = calculate_metrics_for_modality(modality_data, "Path")
    #print(f"Path Modality - BACC: {bacc_Path:.5f}, Recall: {recall_Path:.5f}, Specificity: {specificity_Path:.5f}")
    bacc_Clingen, recall_Clingen, specificity_Clingen = calculate_metrics_for_modality(modality_data, "Clingen")
    #print(f"Clingen Modality - BACC: {bacc_Clingen:.5f}, Recall: {recall_Clingen:.5f}, Specificity: {specificity_Clingen:.5f}")


    
    modalities_stats['BACC_CT'] = bacc_CT
    modalities_stats['BACC_MR'] = bacc_MR
    modalities_stats['BACC_Path'] = bacc_Path
    modalities_stats['BACC_Clingen'] = bacc_Clingen
    modalities_stats['Recall_CT'] = recall_CT
    modalities_stats['Recall_MR'] = recall_MR
    modalities_stats['Recall_Path'] = recall_Path
    modalities_stats['Recall_Clingen'] = recall_Clingen
    modalities_stats['specificity_CT'] = recall_CT
    modalities_stats['specificity_MR'] = recall_MR
    modalities_stats['specificity_Path'] = recall_Path
    modalities_stats['specificity_Clingen'] = recall_Clingen
    
    #Calculate weights
    weights = [bacc_CT, bacc_MR, bacc_Path, bacc_Clingen]
    
    fusion_data = fill_fusion_data(modality_data, fusion_data, weights)
    train_accuracy = calculate_train_accuracy(fusion_data)
    
    #print(f"Training Accuracy: {train_accuracy}")
    train_stats['train_acc'] = train_acc
    
    current_lr = optimizer.param_groups[0]['lr']
    train_stats['train_lr'] = current_lr
    
    combination_accuracy, bacc_per_modality = calculate_combination_accuracy(fusion_data)
    #print_comb_acc(combination_accuracy)
    train_stats['combination_accuracy'] = combination_accuracy

    ############################### AVERAGE DAS LOSSES ##################################
    # Calculate average losses for each modality
    average_loss_CT = train_loss_CT / len(modality_data["CT"]["patient_id"]) if len(modality_data["CT"]["patient_id"]) > 0 else 0
    average_loss_MR = train_loss_MR / len(modality_data["MR"]["patient_id"]) if len(modality_data["MR"]["patient_id"]) > 0 else 0
    average_loss_Path = train_loss_Path / len(modality_data["Path"]["patient_id"]) if len(modality_data["Path"]["patient_id"]) > 0 else 0
    average_loss_Clingen = train_loss_Clingen / len(modality_data["Clingen"]["patient_id"]) if len(modality_data["Clingen"]["patient_id"]) > 0 else 0

    # Assign the average losses to the train_stats dictionary
    train_stats['train_loss_CT'] = average_loss_CT
    train_stats['train_loss_MR'] = average_loss_MR
    train_stats['train_loss_Path'] = average_loss_Path
    train_stats['train_loss_Clingen'] = average_loss_Clingen
    
    #print(f"CT_LOSS: {average_loss_CT:.5f}, MR_LOSS: {average_loss_MR:.5f}, PATH_LOSS: {average_loss_Path:.5f}, CLIN_LOSS: {average_loss_Clingen:.5f}")
    y_pred = fusion_data["Final"]["prediction"]
    y_true = fusion_data["Final"]["label"]
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    train_stats['bacc'] = bacc
    


    
    """
    for modality, data in fusion_data.items():
        print(f"Modality: {modality}")
        for key, value in data.items():
            print(f"{key}: {value}")
            print()  # Add a blank line to separate modalities
   
    patients_per_modality = {}
    for modality, data in fusion_data.items():
        patients = data["patient_id"]
        unique_patients = set(patients)  # Using a set to get unique patient IDs
        patients_per_modality[modality] = len(unique_patients)

    # Print the count of patients per modality
    for modality, count in patients_per_modality.items():
        print(f"{modality}: {count} patients")
     """

    
    return modalities_stats, train_stats

def evaluation(models,
               dataloader: torch.utils.data.DataLoader, 
               losses,
               device: torch.device,
               epoch: int,
               combinations_csv=None,
               args=None):
    
    eval_acc = 0
    results = {}
    modalities_stats = {}
    modalities_csv = load_combinations_csv(combinations_csv)
    combination_accuracy = {}
    
    modality_data = {
        "CT": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },
        "MR": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },
        "Path": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },
        "Clingen": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },        
    }

    fusion_data = {
        "Final": {
            "patient_id": [],
            "combination": [],
            "label": [],
            "output": [],
            "prediction": []
        },
    }


    for model in models:
        model.eval()  # Switch to evaluation mode

    eval_loss_CT, eval_loss_MR, eval_loss_Path, eval_loss_Clingen = 0, 0, 0, 0

    with torch.no_grad(), torch.inference_mode():  # Disable gradient calculation for evaluation
        for batch_idx, (patient_ids, CT, MR, feature_path, clingen, label) in enumerate(dataloader):
           
            label = label.float()
            label = label.view(-1, 1)
            
            CT, MR, feature_path, clingen, label = CT.to(device), MR.to(device), feature_path.to(device), clingen.to(device), label.to(device)
            combinations = get_modality_combination(patient_ids, modalities_csv)

            input_CT = []
            input_MR = []
            label_insta_CT = []
            label_insta_MR = []
            label_insta_Path = []
            label_insta_Clingen = []

            for batch_idx, (model, loss) in enumerate(zip(models, losses)):
                loss_modality = None
                indices_to_append = []
                # Choose the correct modality and forward pass
                if model == models[0]:  # CT model
                    for i, combination in enumerate(combinations):
                        if combination[0] == 1:
                            input_CT.append(CT[i])
                            label_insta_CT.append(label[i])
                            indices_to_append.append(i)

                    if input_CT and label_insta_CT:
                        output = model(torch.cat(input_CT))
                        output = output.squeeze(dim=1)
                        loss_modality = loss(output, torch.cat(label_insta_CT))
                        eval_loss_CT += loss_modality.item()
                        prediction = (torch.sigmoid(output) > 0.5).float()

                        # Loop through the patients in the batch and store data in the dictionary
                        for i, index in enumerate(indices_to_append):
                            modality_data["CT"]["patient_id"].append(patient_ids[index])
                            modality_data["CT"]["combination"].append(combinations[index])
                            modality_data["CT"]["label"].append(label_insta_CT[i].item())
                            modality_data["CT"]["output"].append(output[i].item())
                            modality_data["CT"]["prediction"].append(prediction[i].item())

                elif model == models[1]:  # MR model
                    for i, combination in enumerate(combinations):
                        if combination[1] == 1:
                            input_MR.append(MR[i])
                            label_insta_MR.append(label[i])
                            indices_to_append.append(i)

                    if input_MR and label_insta_MR:
                        output = model(torch.cat(input_MR))
                        output = output.squeeze(dim=1)
                        loss_modality = loss(output, torch.cat(label_insta_MR))
                        eval_loss_MR += loss_modality.item()
                        prediction = (torch.sigmoid(output) > 0.5).float()

                        for i, index in enumerate(indices_to_append):
                            modality_data["MR"]["patient_id"].append(patient_ids[index])
                            modality_data["MR"]["combination"].append(combinations[index])
                            modality_data["MR"]["label"].append(label_insta_MR[i].item())
                            modality_data["MR"]["output"].append(output[i].item())
                            modality_data["MR"]["prediction"].append(prediction[i].item())


                elif model == models[2]:  # Path model
                    output = model(feature_path)
                    loss_modality = loss(output, label)
                    eval_loss_Path += loss_modality.item()
                    prediction = (torch.sigmoid(output) > 0.5).float()
                    for i, patient_id in enumerate(patient_ids):
                        modality_data["Path"]["patient_id"].append(patient_id)
                        modality_data["Path"]["combination"].append(combinations[i])
                        modality_data["Path"]["label"].append(label[i].item())
                        modality_data["Path"]["output"].append(output[i].item())
                        modality_data["Path"]["prediction"].append(prediction[i].item())

                elif model == models[3]:  # Clingen model
                    output = model(clingen)
                    loss_modality = loss(output, label)
                    eval_loss_Clingen += loss_modality.item()
                    prediction = (torch.sigmoid(output) > 0.5).float()
                    for i, patient_id in enumerate(patient_ids):
                        modality_data["Clingen"]["patient_id"].append(patient_id)
                        modality_data["Clingen"]["combination"].append(combinations[i])
                        modality_data["Clingen"]["label"].append(label[i].item())
                        modality_data["Clingen"]["output"].append(output[i].item())
                        modality_data["Clingen"]["prediction"].append(prediction[i].item())

    
    #Calculate BACC and Recall for each modality
    bacc_CT, recall_CT, specificity_CT = calculate_metrics_for_modality(modality_data, "CT")
    #print(f"CT Modality - BACC: {bacc_CT:.5f}, Recall: {recall_CT:.5f}, Specificity: {specificity_CT:.5f}")

    bacc_MR, recall_MR, specificity_MR = calculate_metrics_for_modality(modality_data, "MR")
    #print(f"MR Modality - BACC: {bacc_MR:.5f}, Recall: {recall_MR:.5f}, Specificity: {specificity_MR:.5f}")

    bacc_Path, recall_Path, specificity_Path = calculate_metrics_for_modality(modality_data, "Path")
    #print(f"Path Modality - BACC: {bacc_Path:.5f}, Recall: {recall_Path:.5f}, Specificity: {specificity_Path:.5f}")

    bacc_Clingen, recall_Clingen, specificity_Clingen = calculate_metrics_for_modality(modality_data, "Clingen")
    #print(f"Clingen Modality - BACC: {bacc_Clingen:.5f}, Recall: {recall_Clingen:.5f}, Specificity: {specificity_Clingen:.5f}")

    
    modalities_stats['BACC_CT'] = bacc_CT
    modalities_stats['BACC_MR'] = bacc_MR
    modalities_stats['BACC_Path'] = bacc_Path
    modalities_stats['BACC_Clingen'] = bacc_Clingen
    
    modalities_stats['Recall_CT'] = recall_CT
    modalities_stats['Recall_MR'] = recall_MR
    modalities_stats['Recall_Path'] = recall_Path
    modalities_stats['Recall_Clingen'] = recall_Clingen
    
    modalities_stats['specificity_CT'] = recall_CT
    modalities_stats['specificity_MR'] = recall_MR
    modalities_stats['specificity_Path'] = recall_Path
    modalities_stats['specificity_Clingen'] = recall_Clingen
    
    
    #Calculate weights
    weights = [bacc_CT, bacc_MR, bacc_Path, bacc_Clingen]
    
    fusion_data = fill_fusion_data(modality_data, fusion_data, weights)
    
    val_acc = calculate_train_accuracy(fusion_data)

    #print(f"Validation Accuracy: {val_acc}")
    results['acc1'] = val_acc
    
    #Calculate accuracy per combination
    combination_accuracy, combination_pre_bacc = calculate_combination_accuracy(fusion_data)
    #print_comb_acc(combination_accuracy)
    
    #calculate bacc per combination
    combination_bacc = calculate_bacc(combination_pre_bacc)
    """
    for combination_str, bacc in combination_bacc.items():
        print(f"Combination: {combination_str}, BACC: {bacc}")
    """
    results['combination_accuracy'] = combination_accuracy
    results['combination_bacc'] = combination_bacc

    # Calculate average losses for each modality
    average_loss_CT = eval_loss_CT / len(modality_data["CT"]["patient_id"]) if len(modality_data["CT"]["patient_id"]) > 0 else 0
    average_loss_MR = eval_loss_MR / len(modality_data["MR"]["patient_id"]) if len(modality_data["MR"]["patient_id"]) > 0 else 0
    average_loss_Path = eval_loss_Path / len(modality_data["Path"]["patient_id"]) if len(modality_data["Path"]["patient_id"]) > 0 else 0
    average_loss_Clingen = eval_loss_Clingen / len(modality_data["Clingen"]["patient_id"]) if len(modality_data["Clingen"]["patient_id"]) > 0 else 0
    #print(f"CT_LOSS: {average_loss_CT:.5f}, MR_LOSS: {average_loss_MR:.5f}, PATH_LOSS: {average_loss_Path:.5f}, CLIN_LOSS: {average_loss_Clingen:.5f}")
    
    # Assign the average losses to the train_stats dictionary
    results['loss_CT'] = average_loss_CT
    results['loss_MR'] = average_loss_MR
    results['loss_Path'] = average_loss_Path
    results['loss_Clingen'] = average_loss_Clingen
    

    
    #create text file with predictions
    utils.late_predictions(epoch, fusion_data)
    
    #Late fusion prediction and targets
    preds = fusion_data["Final"]["prediction"]
    targets = fusion_data["Final"]["label"]
    
    # Calculate Balanced Accuracy (BACC) per combination
    results['confusion_matrix'] = confusion_matrix(targets, preds)
    results['f1_score'] = f1_score(targets, preds, average=None, zero_division=1) 
    results['precision'] = precision_score(targets, preds, average=None, zero_division=1)
    results['recall'] = recall_score(targets, preds, average=None)
    results['bacc'] = balanced_accuracy_score(targets, preds)

    
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


def weighted_late_fusion(model_outputs, weights):
    # Weighted late fusion
    fused_output = torch.zeros_like(model_outputs[0])
    for i, output in enumerate(model_outputs):
        fused_output += weights[i] * output
    return fused_output


def calculate_detection_rate(modality_data):
    weights = {}
    
    for modality, data in modality_data.items():
        labels = data["label"]
        predictions = data["prediction"]
        
        
        # Calculate True Positives (TP), True Negatives (TN),
        # False Positives (FP), and False Negatives (FN)
        TP = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1)
        TN = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0)
        FP = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
        FN = sum(1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 0)
        
        # Calculate the detection rate (DR)
        DR = TP / (TP + TN + FP + FN)
        
        weights[modality] = 1-DR
    
    return weights

def fill_fusion_data(modality_data, fusion_data, weights_detection):
    final_data = fusion_data["Final"]
    path_data = modality_data["Path"]

    for patient_id, combination, label in zip(path_data["patient_id"], path_data["combination"], path_data["label"]):
        # Initialize output and prediction for this patient
        print("\n************NEW PATIENT*********")
        final_output = 0
        final_prediction = 0
        
        

        # Check if the patient exists in CT, MR, and Clingen modalities
        patient_exists_in_ct = patient_id in modality_data["CT"]["patient_id"]
        patient_exists_in_mr = patient_id in modality_data["MR"]["patient_id"]
        patient_exists_in_path = patient_id in modality_data["Path"]["patient_id"]
        patient_exists_in_clingen = patient_id in modality_data["Clingen"]["patient_id"]
        
        

        # Retrieve outputs for CT, MR, Path, and Clingen modalities if the patient exists
        output_ct = modality_data["CT"]["output"][modality_data["CT"]["patient_id"].index(patient_id)] if patient_exists_in_ct else 0
        output_mr = modality_data["MR"]["output"][modality_data["MR"]["patient_id"].index(patient_id)] if patient_exists_in_mr else 0
        output_path = modality_data["Path"]["output"][path_data["patient_id"].index(patient_id)]
        output_clingen = modality_data["Clingen"]["output"][modality_data["Clingen"]["patient_id"].index(patient_id)] if patient_exists_in_clingen else 0

        # Calculate the sum of weights for available modalities

        available_weights_sum = (
            weights_detection[0] * patient_exists_in_ct +
            weights_detection[1] * patient_exists_in_mr +
            weights_detection[2] * patient_exists_in_path +
            weights_detection[3] * patient_exists_in_clingen
        )
        
        # Normalize the weights for available modalities
        normalized_weights = [
            weights_detection[0] * patient_exists_in_ct / available_weights_sum,
            weights_detection[1] * patient_exists_in_mr / available_weights_sum,
            weights_detection[2] / available_weights_sum,
            weights_detection[3] * patient_exists_in_clingen / available_weights_sum
        ]
        
       

        # Multiply outputs by corresponding weights and stack them
        weighted_outputs = [
            normalized_weights[0] * output_ct,
            normalized_weights[1] * output_mr,
            normalized_weights[2] * output_path,
            normalized_weights[3] * output_clingen
        ]
        
        
        
        stacked_outputs = sum(weighted_outputs)

        stacked_outputs_tensor = torch.tensor(stacked_outputs)  # Convert to a PyTorch tensor
        
        
        predictions = (torch.sigmoid(stacked_outputs_tensor) > 0.5).float()

        # Append data to the "Final" section of Fusion_data
        final_data["patient_id"].append(patient_id)
        final_data["combination"].append(combination)
        final_data["label"].append(label)
        final_data["output"].append(stacked_outputs)
        final_data["prediction"].append(predictions)

    return fusion_data

def calculate_train_accuracy(fusion_data):
    predictions = fusion_data["Final"]["prediction"]
    labels = fusion_data["Final"]["label"]
    
    correct_predictions = sum(pred == label for pred, label in zip(predictions, labels))
    
    train_accuracy = correct_predictions / len(predictions)
    
    return train_accuracy

def calculate_metrics_for_modality(modality_data, modality_name):
    label = modality_data[modality_name]["label"]
    prediction = modality_data[modality_name]["prediction"]

    # Calculate balanced accuracy and recall
    bacc = balanced_accuracy_score(label, prediction)
    recall = recall_score(label, prediction)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(label, prediction).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return bacc, recall, specificity

def print_comb_acc(combination_accuracy):
    for combination_str, data in combination_accuracy.items():
        total = data["total"]
        correct = data["correct"]
        accuracy = data["accuracy"]

        # Convert the combination string back to a tuple for display
        combination_tuple = eval(combination_str)

        print(f"Combination {combination_tuple}:",f"Total: {total}", f"Correct: {correct}",f"Accuracy: {accuracy:.3%}")

        
    
def calculate_combination_accuracy(fusion_data):
    combination_accuracy = {}
    total_correct_accuracy = {}

    for i in range(len(fusion_data["Final"]["combination"])):
        combination = fusion_data["Final"]["combination"][i]
        label = fusion_data["Final"]["label"][i]
        prediction = fusion_data["Final"]["prediction"][i]

        combination_str = str(combination)

        if combination_str not in combination_accuracy:
            combination_accuracy[combination_str] = {"total": 0, "correct": 0, "y_true": [], "y_pred": []}

        combination_accuracy[combination_str]["total"] += 1
        if label == prediction:
            combination_accuracy[combination_str]["correct"] += 1

        combination_accuracy[combination_str]["y_true"].append(label)
        combination_accuracy[combination_str]["y_pred"].append(prediction)

    for combination_str, data in combination_accuracy.items():
        total = data["total"]
        correct = data["correct"]
        accuracy = correct / total if total > 0 else 0.0
        total_correct_accuracy[combination_str] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        }

    return total_correct_accuracy, combination_accuracy


def calculate_bacc(combination_accuracy):
    bacc_per_combination = {}
    for combination_str, data in combination_accuracy.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        
        # Calculate BACC for the combination
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

        # Check if denominators are zero before division
        if (tp + fn) == 0 or (tn + fp) == 0:
            bacc = 1  # Handle the case of division by zero
            
        else:
            bacc = 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
        
        bacc_per_combination[combination_str] = bacc

    return bacc_per_combination


