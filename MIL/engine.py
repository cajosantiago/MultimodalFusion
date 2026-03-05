"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from timm.utils import ModelEma

from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, \
    balanced_accuracy_score

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



#delete loss_scaler, max_norm, lr_scheduler, model_ema, wandb
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               criterion: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int,
               loss_scaler,
               lr_scheduler=None,
               model_ema: Optional[ModelEma] = None,
               warmup = False,
               args = None):
        
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    train_stats = {}
    train_loss = 0.0
    lr_num_updates = epoch * len(dataloader)
    
    # Loop through data loader data batches
    for batch_idx, (inputs, labels, patient, numb_fmap) in enumerate(dataloader):
        #print(batch_idx)
        # Send data to target device
        labels = labels.float()
        #IF PATIENT SURVIVES - 0
        #IF PATIENT DIES - 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        #with torch.cuda.amp.autocast():
        scores, position = model(inputs, numb_fmap) # 2.Forward pass
        #input = 1 - scores  
        loss = criterion(scores, labels) # 3. Compute and accumulate loss

        train_loss += loss.item() 
        
        loss.backward() # 3. Backward pass
        
        if batch_idx % 10 == 0:
            #print("update weights")
            # 5. Update weights
            optimizer.step() 
             # 1. Clear gradients
            optimizer.zero_grad()
            
        """
        if not args.cosine_one_cycle:
            lr_scheduler.step_update(num_updates=lr_num_updates)
        """
            
        # Calculate and accumulate accuracy metric across all batches
             
        predictions = (torch.sigmoid(scores) > 0.5).float() #for when the max pool outputs the highest probability of being classified as 1
        
        train_acc += (predictions == labels).sum().item()
        #print("\n TRAIN ACC:", train_acc)
        #print("\nLABEL:", labels, "\n SCORES:", scores, "\nPRED:", predictions)
        
    
    # Adjust metrics to get average loss and accuracy per batch 
    current_lr = optimizer.param_groups[0]['lr']
    print("Current learning rate:", current_lr)
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    train_stats['train_loss'] = train_loss
    train_stats['train_acc'] = train_acc
    train_stats['train_lr'] = optimizer.param_groups[0]['lr']
    
    return train_stats

def evaluation(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               criterion: torch.nn.Module, 
               device: torch.device,
               epoch: int,
               args=None):
    
    #criterion = torch.nn.CrossEntropyLoss()

    # Switch to evaluation mode
    model.eval()
    
    preds = []
    targets = []
    test_loss, test_acc = 0, 0
    results = {}
    
    for batch_idx, (inputs, labels, patient, numb_fmap) in enumerate(dataloader):
        
        #inputs, targets_ = inputs.to(device, non_blocking=True), targets_.to(device, non_blocking=True)
        #print("val", batch_idx)
        # Compute output
        labels = labels.float()
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad(), torch.inference_mode():
            scores, position = model(inputs, numb_fmap)
            
            loss = criterion(scores, labels)
            test_loss += loss.item()
            
            predictions = (torch.sigmoid(scores) > 0.5).float() #when the max pool outputs the highest probability of being classified as 0.
            
            #print("pred:", predictions, "label", labels)
    
        # Calculate and accumulate accuracy
        test_acc += (predictions == labels).sum().item()
        
        preds.append(predictions.cpu().numpy())
        targets.append(labels.cpu().numpy())
        

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)

    # Compute Metrics
    results['confusion_matrix'], results['f1_score'] = confusion_matrix(targets, preds), f1_score(targets, preds, average=None) 
    results['precision'], results['recall'] = precision_score(targets, preds, average=None), recall_score(targets, preds, average=None)
    results['bacc'] = balanced_accuracy_score(targets, preds)
    results['acc1'], results['loss'] = test_acc, test_loss

    return results