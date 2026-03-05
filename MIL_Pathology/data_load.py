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
    
    def __init__(self, csv_patients, csv_file_location, bag_size, noise):
        patients_df = pd.read_csv(csv_patients)
        
        #switch the labels for MIL
        patients_df['vital_status_12'] = patients_df['vital_status_12'].map({0: 1, 1: 0})
        #IF PATIENT SURVIVES - 0
        #IF PATIENT DIES - 1
        
        self.patients = patients_df['case_id'].tolist()
        self.labels = patients_df['vital_status_12'].tolist()
        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.data = pd.read_csv(csv_file_location)        
        #self.image_paths = self.data['File Location'].tolist()
        self.bag_size = bag_size
        self.noise = noise
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index):
        bag_feature_map, bag_label, numb_fmap = self.create_bag(index)
     
        patient = self.patients[index]
        
        return bag_feature_map, bag_label, patient, numb_fmap

    def create_bag(self, index):
        bag_feature_map = torch.zeros(self.bag_size, 1, 1024)
        bag_label = self.labels[index]
        file_locations = self.data.loc[self.data['case_id'] == self.patients[index], 'slide_location'].tolist()
    
        
        torch.set_printoptions(threshold=sys.maxsize)
        numb_fmap = 0
        for i in range(self.bag_size):
            image_path = file_locations[i]
  
            if image_path == "Not applicable":
                feature_map = torch.zeros(1, 1024)
            else:
                numb_fmap = numb_fmap + 1
                feature_map_path = image_path
                # Load the feature map
                features = np.load(feature_map_path)['arr_0']
                
                # To add noise or not - data augmentation
                
                
                if self.noise == 1:
                    mean = 0.0  # Mean of the Gaussian distribution
                    std_dev = 0.005  # Standard deviation of the Gaussian distribution
                    noisy_feature_map = add_gaussian_noise(features, mean, std_dev)
                    feature_map = torch.from_numpy(noisy_feature_map)
                    
                else:
                    feature_map = torch.from_numpy(features)
           
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
