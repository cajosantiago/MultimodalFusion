import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models



class Multimodal_fusion(nn.Module):
    def __init__(self, args):
        super(Multimodal_fusion, self).__init__()
        self.args = args

        # Input dimensions
        self.mr_ct_dim = 512
        self.path_dim = 1024
        self.clingen_dim = 17
        # Hidden dimensions
        self.hidden_dim = 128
        self.mmhid = 128
        # Fusion methods
        self.concat = 512 #- 128
        self.num_classes = 1

        # Define fully connected layers with batch normalization
        self.fc1 = nn.Sequential(
            nn.Linear(self.mr_ct_dim, self.hidden_dim), #512 - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            nn.BatchNorm1d(self.hidden_dim)   # Batch normalization
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.mr_ct_dim, self.hidden_dim), #512 - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            nn.BatchNorm1d(self.hidden_dim)   # Batch normalization
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.clingen_dim, self.hidden_dim), #14 - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            nn.BatchNorm1d(self.hidden_dim)   # Batch normalization
        )

        self.fc4 = nn.Sequential(
            nn.Linear(self.path_dim, self.hidden_dim), # 1024 - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            nn.BatchNorm1d(self.hidden_dim)   # Batch normalization
        )

        # FOR CONCAT
        self.fuse_concat_fc = nn.Sequential(
            nn.Linear(self.concat, self.mmhid), #cat_size - 128
            nn.BatchNorm1d(self.mmhid),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.mmhid, self.mmhid), # 128 - 128
            nn.BatchNorm1d(self.mmhid)   # Batch normalization
            #nn.ReLU(),
            
        )

        # FOR MEAN
        self.fuse_mean_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.mmhid),
            nn.BatchNorm1d(self.mmhid),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.mmhid, self.mmhid),
            nn.BatchNorm1d(self.mmhid),   # Batch normalization
            nn.ReLU()
            
        )

        # 128 - 1
        self.classifier = nn.Sequential(
            nn.Linear(self.mmhid, self.num_classes) #128 - 1
        )

    def forward(self, CT, MR,  feature_path, clingen, combinations):


        #CT shape = (batch_size,512,1,1) - (batch size, embedding_size, 1, 1)
        #MR shape = (batch_size,512,1,1) - (batch size, embedding_size, 1, 1)
        #path shape = (1,1024) - (batch size, embedding_size)
        #clingen = (1,1,17) - (batch size, 1, embedding_size)

        #print(CT.shape)
        #print(MR.shape)
        # (1) Transform input to shape(batch_size, 1, embedding_size)
        CT_feat = CT.squeeze(dim=3)
        MR_feat = MR.squeeze(dim=3)

        # (2) Convert x to shape (Batch_size, embedding_size)
        CT_feat = CT_feat.reshape(-1, CT_feat.size(2)) #(1,512)
        MR_feat = MR_feat.reshape(-1, MR_feat.size(2)) #(1,512)
        CG_feat = clingen.reshape(-1, clingen.size(2)) #(1,17)
        #Path = (1, 1024)       
        CT_feat_encode = F.relu(self.fc1(CT_feat)) #(1,128)
        MR_feat_encode = F.relu(self.fc2(MR_feat)) #(1,128)
        #casts the clingen tensor to the same data type as the weight matrix of the self.fc3
        Clingen_feat_encode = F.relu(self.fc3(CG_feat.to(self.fc3[0].weight.dtype))) #(1,128)
        PATH_feat_encode = F.relu(self.fc4(feature_path)) #(1,128)

        #print("CT:", CT_feat_encode.shape, "MR", MR_feat_encode.shape,"PATH:", PATH_feat_encode.shape, "clingen:", Clingen_feat_encode.shape)

        if self.args.fusion == "mean":
            mean = masked_mean(CT_feat_encode, MR_feat_encode, PATH_feat_encode, Clingen_feat_encode, combinations)
            features = self.fuse_mean_fc(mean)
        if self.args.fusion == "cat":
            #concatenated_features = torch.cat((CT_feat_encode, Clingen_feat_encode), dim=1)
            #concatenated_features = torch.cat((CT_feat_encode, MR_feat_encode), dim=1)
            #concatenated_features = torch.cat((CT_feat_encode, MR_feat_encode, Clingen_feat_encode), dim=1)
            concatenated_features = torch.cat((CT_feat_encode, MR_feat_encode, PATH_feat_encode, Clingen_feat_encode), dim=1) #(1,512)
            features = self.fuse_concat_fc(concatenated_features)

        #print("fuse:", features.shape)

        output = self.classifier(features)

        #print("output:", output)

        return output
    

def masked_mean(CT_feat_encode, MR_feat_encode, PATH_feat_encode, Clingen_feat_encode, combinations):
    masked_means = []
    for i, patient_combinations in enumerate(combinations):
        # Apply the mask to each patient's modalities    
        CT_feat_encoded_masked = CT_feat_encode[i] * patient_combinations[0]
        MR_feat_encoded_masked = MR_feat_encode[i] * patient_combinations[1]
        PATH_feat_encoded_masked = PATH_feat_encode[i] * patient_combinations[2]
        Clingen_feat_masked = Clingen_feat_encode[i] * patient_combinations[4]  

        # Sum the masked modalities (CLINGEN is always present)
        masked_sum = CT_feat_encoded_masked + MR_feat_encoded_masked + PATH_feat_encoded_masked + Clingen_feat_masked

        # Calculate the total count of modalities included in the mask for this patient
        total_modalities_count = patient_combinations[0] + patient_combinations[1] + patient_combinations[2] + patient_combinations[4]
        

        # Calculate the masked mean for this patient
        masked_mean = masked_sum / total_modalities_count

        # Append the masked mean for this patient to the list
        masked_means.append(masked_mean)

    # Combine the masked means for all patients into a single tensor
    masked_means_tensor = torch.stack(masked_means)

    return masked_means_tensor
