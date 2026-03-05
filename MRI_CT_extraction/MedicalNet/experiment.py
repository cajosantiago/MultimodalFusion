from setting import parse_opts 
from datasets.datasets import BrainS18Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage

import sys
import os
from utils.file_process import load_lines
import numpy as np
from models import resnet

import torch.nn as nn
import torchvision.models as models


class MedicalNet(nn.Module):

  def __init__(self, path_to_weights, device):
    super(MedicalNet, self).__init__()
    
    self.model = resnet.resnet18(sample_input_D=56, sample_input_H=448, sample_input_W=448, num_seg_classes=2)
    #print(self.model)
    #Loads the state of the model and the pretrained weights
    net_dict = self.model.state_dict()
    pretrained_weights = torch.load(path_to_weights, map_location=torch.device(device))
    pretrain_dict = {
        k.replace("module.", ""): v for k, v in pretrained_weights['state_dict'].items() if k.replace("module.", "") in net_dict.keys()
      }
    net_dict.update(pretrain_dict)
    self.model.load_state_dict(net_dict)
    
    #Useless for this application
    self.fc = nn.Linear(512, 1)
    
    #which layer to extract features from
    self.layer_num = 7

  def forward(self, x):
        # Extract feature maps from a specified layer
        
        for index, layer in enumerate(self.model.children()):
            #print("\n layer:", index, layer)
            x = layer(x)
            if index == self.layer_num:
                feature_map = x
                #print("\nShape of feature map:", feature_map.shape)
                #np.save('array.npy', feature_map.numpy())
                #print("\n Shape of feature map:", feature_map)
                #print("\n Feature map:", feature_map.shape)
                #print("\n NEW")
                break

        return feature_map

if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'
    sets.num_classes = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # data tensor
    testing_data = BrainS18Dataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # generate model
    path_to_weights = '/home/csantiago/MRI_CT_extraction/MedicalNet/pretrain/resnet_18_23dataset.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = MedicalNet(path_to_weights, device=device)
    
    # Extract feature maps from the 3D DICOM images in the data loader
   
    # To get the directories where to save the feature maps
    img_direct = '/home/csantiago/Datasets/Radiology-CPTAC/full_file_location_CT.txt'
    with open(img_direct, 'r') as f:
        img_list = [line.strip() for line in f]

    folder_name = 'feature_map'
    filename_features = 'feature_map.npz'
    
    
    model.eval()
    with torch.no_grad():
      for batch_idx, input in enumerate(data_loader):
          print("\n Batch_idx:", batch_idx)  
          
          #continue
          feature_map = model(input)
          continue
          save_feature_map = feature_map.detach().numpy().astype(np.float16)
          #print("\n feature map:", save_feature_map.shape)  
          
          #Save the feature maps locally
          
          dir_name = img_list[batch_idx]
          #create folder to save np array
          dir_save = os.path.join(dir_name, folder_name)
          if not os.path.exists(dir_save):
              os.makedirs(dir_save)

          np.savez_compressed(os.path.join(dir_save, filename_features), save_feature_map)
          
          


        
    
