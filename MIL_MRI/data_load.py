import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys


class FeatureDataset(Dataset):
    
    def __init__(self, csv_patients, csv_file_location, bag_size):
        patients_df = pd.read_csv(csv_patients)
        
        #switch the labels for MIL
        patients_df['vital_status_12'] = patients_df['vital_status_12'].map({0: 1, 1: 0})
        #IF PATIENT SURVIVES - 0
        #IF PATIENT DIES - 1
                
        self.original = patients_df['original']
        self.patients = patients_df['case_id'].tolist()
        self.labels = patients_df['vital_status_12'].tolist()
        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.data = pd.read_csv(csv_file_location)        
        #self.image_paths = self.data['File Location'].tolist()
        self.bag_size = bag_size

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index):
        bag_feature_map, bag_label, numb_fmap = self.create_bag(index)
        patient = self.patients[index]
        
        return bag_feature_map, bag_label, patient, numb_fmap

    def create_bag(self, index):
        bag_feature_map = torch.zeros(self.bag_size, 512, 1)
        bag_label = self.labels[index]
        file_locations = self.data.loc[self.data['case_id'] == self.patients[index], 'File Location'].tolist()
        original = self.original[index]
        
        torch.set_printoptions(threshold=sys.maxsize)
        numb_fmap = 0
        for i in range(self.bag_size):
            image_path = file_locations[i]
  
            if image_path == "Not applicable":
                feature_map = torch.zeros(1, 512, 1)
            else:
                numb_fmap = numb_fmap + 1
                feature_map_path = os.path.join(image_path, "feature_map/feature_map.npz")
                # Load the feature map
                features = np.load(feature_map_path)['arr_0']
                # To add noise or not - data augmentation
                #noise_select = random.randint(1, 2)
                noise_select = 1
                if noise_select == 1:
                    mean = 0.0  # Mean of the Gaussian distribution
                    std_dev = 0.001  # Standard deviation of the Gaussian distribution
                    noisy_feature_map = add_gaussian_noise(features, mean, std_dev)

                elif noise_select == 2:
                    salt_prob = 0.01  # Probability of adding salt noise 
                    pepper_prob = 0.01  # Probability of adding pepper noise 
                    noisy_feature_map = add_salt_and_pepper_noise(features, salt_prob, pepper_prob)

                original_feature_map = torch.from_numpy(noisy_feature_map).float()
         
                
                global_avg_pool = F.avg_pool3d(original_feature_map, kernel_size=(7, 56, 56))  # [1, 512, 1, 1, 1]
                feature_map = global_avg_pool.view(global_avg_pool.size(0), global_avg_pool.size(1), global_avg_pool.size(2))  # [1, 512, 1]
                #print(feature_map.shape)
                
                # Check if there are any inf values
                has_inf = torch.any(torch.isinf(feature_map))

                # Print the result
                if has_inf:
                    print(has_inf)
        
            bag_feature_map[i] = feature_map
            
        return bag_feature_map, bag_label, numb_fmap

def add_gaussian_noise(feature_maps, mean, std_dev):
    noise = np.random.normal(mean, std_dev, size=feature_maps.shape)
    noisy_feature_maps = feature_maps + noise
    return noisy_feature_maps

def add_salt_and_pepper_noise(feature_maps, salt_prob, pepper_prob):
    noisy_feature_maps = np.copy(feature_maps)
    batch_size, channels, depth, height, width = feature_maps.shape

    # Generate random indices for salt and pepper noise
    salt_indices = np.random.choice([True, False], size=(batch_size, channels, depth, height, width), p=[salt_prob, 1-salt_prob])
    pepper_indices = np.random.choice([True, False], size=(batch_size, channels, depth, height, width), p=[pepper_prob, 1-pepper_prob])

    # Add salt noise
    noisy_feature_maps[salt_indices] = 1.0  # Set salt noise pixels to maximum value

    # Add pepper noise
    noisy_feature_maps[pepper_indices] = 0.0  # Set pepper noise pixels to minimum value

    return noisy_feature_maps