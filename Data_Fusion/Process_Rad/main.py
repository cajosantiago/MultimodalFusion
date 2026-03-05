import data_load
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import os


CT_files = 'Files/patients_with_labels_CT.csv'
MR_files = 'Files/patients_with_labels_MR.csv'

data_CT = pd.read_csv(CT_files)
image_paths_CT = data_CT['File Location'].tolist()


dataset_CT= data_load.FeatureDataset(CT_files)
data_loader_CT = DataLoader(dataset_CT, 
                         shuffle=False, num_workers= 1, 
                         pin_memory= False)
 
for batch_idx, input in enumerate(data_loader_CT):
    print("\n Batch_idx:", batch_idx)
    print("main CT",input.shape)
    
    folder_name = image_paths_CT[batch_idx] 
    file_name_save = 'GAP_feature_map.npz'
    save_feature_map = input.detach().numpy()
    
    np.savez_compressed(os.path.join(folder_name, file_name_save), save_feature_map)

    
    
data_MR = pd.read_csv(MR_files)
image_paths_MR = data_MR['File Location'].tolist()


dataset_MR= data_load.FeatureDataset(MR_files)
data_loader_MR = DataLoader(dataset_MR, 
                         shuffle=False, num_workers= 1, 
                         pin_memory= False)
 
for batch_idx, input in enumerate(data_loader_MR):
    print("\n Batch_idx:", batch_idx)
    print("main MR",input.shape)
    
    folder_name = image_paths_MR[batch_idx] 
    file_name_save = 'GAP_feature_map.npz'
    save_feature_map = input.detach().numpy()
    
    np.savez_compressed(os.path.join(folder_name, file_name_save), save_feature_map)
