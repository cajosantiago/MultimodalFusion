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
    
    def __init__(self, csv):
  
        
        self.data = pd.read_csv(csv)
        self.patients = self.data['case_id'].tolist()
        self.image_paths = self.data['File Location'].tolist()
    

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        patient = self.patients[index]
        file_location = self.image_paths[index]
        feature_map_path = os.path.join(file_location, "feature_map/feature_map.npz")
        # Load the feature map
        features = np.load(feature_map_path)['arr_0']
        # To add noise or not - data augmentation
        original_feature_map = torch.from_numpy(features).float()

        global_avg_pool = F.avg_pool3d(original_feature_map, kernel_size=(7, 56, 56))  # [1, 512, 1, 1, 1]
        feature_map = global_avg_pool.view(global_avg_pool.size(0), global_avg_pool.size(1), global_avg_pool.size(2))  # [1, 512, 1]
    
        # Check if there are any inf values
        has_inf = torch.any(torch.isinf(feature_map))

        # Print the result
        if has_inf:
            print(has_inf)
                    
        return feature_map
