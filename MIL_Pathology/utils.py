import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np
import random

import os
from collections import Counter
from pathlib import Path

import torch
import torchvision.utils as vutils
import torch.serialization
import torch.distributed as dist
from torchinfo import summary
from sklearn.utils.class_weight import compute_class_weight
    
def plot_confusion_matrix(confusion_matrix, class_names, output_dir, args):
    
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(str(output_dir) + '/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.clf()
    
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.
    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def configure_seed(
    seed: int = 42
):
  """Configure the random seed.
    Args:
        seed (int): The random seed. Default value is 42.
  """
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
      
def model_summary(model, args):
    # Print a summary using torchinfo (uncomment for actual output)
    summ = summary(model=model, 
                   input_size=(args.batch_size, 3, 224, 224), # (batch_size, color_channels, height, width)
                   col_names=["input_size", "output_size", "num_params", "trainable"],
                   col_width=20,
                   row_settings=["var_names"])   
    return summ    

def Load_Pretrained_MIL_Model(path, model, args):
    # Load the pretrained Mil model
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
        
    checkpoint_keys = set(checkpoint['model'].keys())
    model_keys = set(model.state_dict().keys())
    unmatched_keys = checkpoint_keys.symmetric_difference(model_keys)
    for k in unmatched_keys:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint['model'][k]
            
    model.load_state_dict(checkpoint['model'])
    

def warmup_dataset(csv, train):
    df = pd.read_csv(csv)
    # Count the occurrences of each class
    class_counts = df['vital_status_12'].value_counts()

    # Find the minority and majority classes
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    # Undersample the majority class randomly
    undersampled_df = df[df['vital_status_12'] == minority_class]
    majority_class_df = df[df['vital_status_12'] == majority_class].sample(n=len(undersampled_df), random_state=42)
    undersampled_df = pd.concat([undersampled_df, majority_class_df])

    # Shuffle the undersampled data
    undersampled_df = undersampled_df.sample(frac=1, random_state=42)
    
    if train:
        # Save the undersampled data to a new CSV file
        undersampled_df.to_csv('Files/warmup_train.csv', index=False)

    else: 
        undersampled_df.to_csv('Files/warmup_val.csv', index=False)
        
def progressive_warmup_create(train_csv, warmup_csv):
    
    # Read the original dataset
    original_dataset = pd.read_csv(train_csv)

    # Read the undersampled dataset
    undersampled_dataset = pd.read_csv(warmup_csv)

    # Calculate the number of majority class samples to add
    num_majority_samples = int((original_dataset['vital_status_12'].value_counts().max() / 3) * 2)

    # Exclude the case IDs already present in the undersampled dataset
    undersampled_case_ids = set(undersampled_dataset['case_id'])
    original_majority_samples = original_dataset[original_dataset['vital_status_12'] == 1]
    filtered_majority_samples = original_majority_samples[~original_majority_samples['case_id'].isin(undersampled_case_ids)]

    # Select the desired proportions of the filtered majority class samples
    num_majority_samples_1 = int(num_majority_samples / 3)
    num_majority_samples_2 = int((2 * num_majority_samples) / 3)

    majority_samples_1 = filtered_majority_samples.head(num_majority_samples_1)
    majority_samples_2 = filtered_majority_samples.head(num_majority_samples_2)

    # Create warmup_train1.csv by concatenating the undersampled dataset and majority_samples_1
    warmup_train1 = pd.concat([undersampled_dataset, majority_samples_1])

    # Save warmup_train1.csv
    warmup_train1.to_csv('Files/warmup_train1.csv', index=False)

    # Create warmup_train2.csv by concatenating the undersampled dataset and majority_samples_2
    warmup_train2 = pd.concat([undersampled_dataset, majority_samples_2])

    # Save warmup_train2.csv
    warmup_train2.to_csv('Files/warmup_train2.csv', index=False)

        
def weight_computing(csv):
    train_df = pd.read_csv(csv)
    #train_df['vital_status_12'] = train_df['vital_status_12'].map({0: 1, 1: 0})
    labels_train = train_df['vital_status_12']
    labels_train= np.array(labels_train)
  
    
    weights = compute_class_weight(class_weight='balanced', classes= np.unique(labels_train), y=labels_train)
    weights_tensor = torch.FloatTensor(weights)
    
    return weights_tensor


def plot_training_stats(train_results, val_results):
    # Plotting training and validation loss
    fig_loss = plt.figure(figsize=(10, 5))
    plt.plot(train_results['loss'], label='Training Loss')
    plt.plot(val_results['loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    save_plot(fig_loss, 'loss_plot.png')
    plt.show()

    # Plotting training and validation accuracy
    fig_acc = plt.figure(figsize=(10, 5))
    plt.plot(train_results['acc'], label='Training Accuracy')
    plt.plot(val_results['acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    save_plot(fig_acc, 'accuracy_plot.png')
    plt.show()

    # Plotting learning rate
    fig_lr = plt.figure(figsize=(10, 5))
    plt.plot(train_results['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.legend()
    save_plot(fig_lr, 'learning_rate_plot.png')
    plt.show()

    # Plotting other metrics (e.g., F1-score, balanced accuracy)
    fig_metrics = plt.figure(figsize=(10, 5))
    plt.plot(val_results['f1'], label='F1-score')
    plt.plot(val_results['bacc'], label='Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend()
    save_plot(fig_metrics, 'metrics_plot.png')
    plt.show()


def save_plot(fig, filename):
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath)
