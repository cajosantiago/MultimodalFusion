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
    
    def __init__(self, dataset_csv, CT_data, MR_data, PATH_data, clinical_data, CT_choice, MR_choice, PATH_choice, noise, train):
        self.general_df = pd.read_csv(dataset_csv)
        self.patients = self.general_df['Subject ID'].tolist()
        self.labels = self.general_df['vital_status_12'].tolist()
        
        self.choice_CT = pd.read_csv(CT_choice)
        self.choice_MR = pd.read_csv(MR_choice)
        self.choice_PATH = pd.read_csv(PATH_choice)
        
        self.CT = pd.read_csv(CT_data)
        self.MR = pd.read_csv(MR_data)
        self.PATH = pd.read_csv(PATH_data)
        self.clinic = pd.read_csv(clinical_data)
        self.clinic = self.clinic.fillna(-1)
        
        self.noise = noise
        self.train = train
            
    def __len__(self):
        return len(self.patients)
        
    
    def __getitem__(self, index):
        patient = self.patients[index]
        label = self.labels[index]
                
        #******************************TEMPORARLY******************************
        # Check if CT exam exists for that patient
        if self.general_df.loc[index, 'CT'] == 1:
            CT_MR_flag = 0
            feature_CT = self.get_radiology(CT_MR_flag, patient)
            
        else:
            feature_CT = torch.zeros(1, 512, 1)
            #print("ESTE", feature_CT.shape)
  
         # Check if MR exam exists for that patient
        if self.general_df.loc[index, 'MR'] == 1:     
            CT_MR_flag = 1
            feature_MR = self.get_radiology(CT_MR_flag, patient)
        else: 
            feature_MR = torch.zeros(1, 512, 1)
            
        if self.general_df.loc[index, 'clinical'] == 1:
            feature_clingen = self.clinic[self.clinic['case_id'] == patient].drop(columns=['vital_status_12', 'case_id']).values
            #print("New:", feature_clingen)
            #if feature_clingen.shape[0] == 0:
                #print(patient)
                #print(feature_clingen)
            #shape (1,14)
        else:
            print("yoo, no clinical here")
            
        # Check if WSI exam exists for that patient
        if self.general_df.loc[index, 'pathology'] == 1:     
            
            feature_path = self.get_pathology(patient)
        else: 
            feature_path = torch.zeros(1024)
        
                    
        
        #print("CT", feature_CT.shape, "MR", feature_MR.shape, "path", feature_path.shape, "Clingen", feature_clingen.shape,)
        
        return patient, feature_CT, feature_MR, feature_path, feature_clingen, label
    
    def get_radiology(self, flag, patient):
        # flag 0 - CT
        # flag 1 - MR
        if flag == 1:
            # Get the chosen exam for the given patient from self.choice_MR
            chosen_exam = self.choice_MR.loc[self.choice_MR['case_id'] == patient, 'chosen_exam'].values
            # Use the chosen_exam to select the corresponding "File Location" from self.MR
            file_location = self.MR.loc[self.MR['case_id'] == patient, 'File Location'].iloc[chosen_exam].values
        else:
            # Get the chosen exam for the given patient from self.choice_CT
            chosen_exam = self.choice_CT.loc[self.choice_CT['case_id'] == patient, 'chosen_exam'].values
            #print("Patient", patient, "Exam", chosen_exam)
            
            # Use the chosen_exam to select the corresponding "File Location" from self.CT
            file_location = self.CT.loc[self.CT['case_id'] == patient, 'File Location'].iloc[chosen_exam].values
            #print(file_location)
        
        
        if len(file_location) > 0:
            feature_map_path = os.path.join(file_location[0], "GAP_feature_map.npz")
            features_map = np.load(feature_map_path)['arr_0']
            
            if self.train == False:
                self.noise = 0
            
            if self.noise == 1:
                mean = 0.0  # Mean of the Gaussian distribution
                std_dev = 0.01  # Standard deviation of the Gaussian distribution
                noisy_feature_map = add_gaussian_noise(features_map, mean, std_dev)
                feature = torch.from_numpy(noisy_feature_map).float()
                feature = feature.squeeze(dim=1)
                #print("Noise:", feature.shape)   
            else:
                feature = torch.from_numpy(features_map).float()
                feature = feature.squeeze(dim=1)
                #print("No noise:", feature.shape)
        else:
            # Handle the case where the no exam can be used (Due to ranking)
            feature = torch.zeros(1, 512, 1)

        return feature

    def get_pathology(self, patient):
        chosen_exam = self.choice_PATH.loc[self.choice_PATH['case_id'] == patient, 'chosen_exam'].values
        file_location = self.PATH.loc[self.PATH['case_id'] == patient, 'slide_location'].iloc[chosen_exam].values

        if len(file_location) > 0:
            feature_map_path = file_location[0]  # Get the file path
            
            features = np.load(feature_map_path)['arr_0']
            
            if self.train == False:
                self.noise = 0
 
            if self.noise == 1:
                mean = 0.0  # Mean of the Gaussian distribution
                std_dev = 0.01  # Standard deviation of the Gaussian distribution
                noisy_feature_map = add_gaussian_noise(features, mean, std_dev)
                feature_map = torch.from_numpy(noisy_feature_map).float()
            else:
                feature_map = torch.from_numpy(features)                            
    
            tensor = feature_map.squeeze()
        else:
            tensor = torch.zeros(1024)
            print("failed_pathology")
        
        return tensor

def add_gaussian_noise(feature_maps, mean, std_dev):
    noise = np.random.normal(mean, std_dev, size=feature_maps.shape)
    noisy_feature_maps = feature_maps + noise
    return noisy_feature_maps
