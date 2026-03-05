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

class CT_classifier(nn.Module):

    def __init__(self, args):
        super(CT_classifier, self).__init__()
        self.mr_ct_dim = 512
        self.num_classes = 1
        self.hidden_dim = 128
        
        # Define fully connected layers with batch normalization
        self.fc1 = nn.Sequential(
            nn.Linear(self.mr_ct_dim, self.hidden_dim), #512 - 128
            #nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            #nn.BatchNorm1d(self.hidden_dim)   # Batch normalization
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_classes) #128 - 1
        )
    
    def forward(self, ct_input):
        # Forward pass 
        #CT shape = (batch_size,512,1,1) - (batch size, embedding_size, 1, 1)
        #print(ct_input.shape)
        ct_input = ct_input.unsqueeze(dim = 1)
        #print(ct_input.shape)
        CT_feat = ct_input.squeeze(dim=3)
        
        # (2) Convert x to shape (Batch_size, embedding_size)
        CT_feat = CT_feat.reshape(-1, CT_feat.size(2)) #(1,512)
        #print(CT_feat.shape)
        CT_feat_encode = F.relu(self.fc1(CT_feat)) #(1,128)

        ct_output = self.classifier(CT_feat_encode)
 

        return ct_output
    
    
class MR_classifier(nn.Module):

    def __init__(self, args):
        super(MR_classifier, self).__init__()
        self.mr_ct_dim = 512
        self.num_classes = 1
        self.hidden_dim = 128
        
        # Define fully connected layers with batch normalization
        self.fc2 = nn.Sequential(
            nn.Linear(self.mr_ct_dim, self.hidden_dim), #512 - 128
            #nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            #nn.BatchNorm1d(self.hidden_dim)   # Batch normalization
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_classes) #128 - 1
        )
    
    def forward(self, mr_input):
        
        mr_input = mr_input.unsqueeze(dim = 1)
        MR_feat = mr_input.squeeze(dim=3)
        # (2) Convert x to shape (Batch_size, embedding_size)
        MR_feat = MR_feat.reshape(-1, MR_feat.size(2)) #(1,512)

        MR_feat_encode = F.relu(self.fc2(MR_feat)) #(1,128)

        mr_output = self.classifier(MR_feat_encode)
        
        return mr_output
        
        
        
class Path_classifier(nn.Module):

    def __init__(self, args):
        super(Path_classifier, self).__init__()
        self.path_dim = 1024
        self.num_classes = 1
        self.hidden_dim = 128
        
        # Define fully connected layers with batch normalization
        self.fc4 = nn.Sequential(
            nn.Linear(self.path_dim, self.hidden_dim), # 1024 - 128
            #nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            #nn.BatchNorm1d(self.hidden_dim)   # Batch normalization
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_classes) #128 - 1
        )
    
    def forward(self, path_input):
        # Forward pass
        #path shape = (1,1024) - (batch size, embedding_size)
        PATH_feat_encode = F.relu(self.fc4(path_input)) #(1,128)
        path_output = self.classifier(PATH_feat_encode)


        return path_output
    
    
class Clingen_classifier(nn.Module):

    def __init__(self, args):
        super(Clingen_classifier, self).__init__()
        self.clingen_dim = 17
        self.num_classes = 1
        self.hidden_dim = 128
        
        # Define fully connected layers with batch normalization
        self.fc3 = nn.Sequential(
            nn.Linear(self.clingen_dim, self.hidden_dim), #14 - 128
            #nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            #nn.BatchNorm1d(self.hidden_dim)   # Batch normalization
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_classes) #128 - 1
        )
    
    def forward(self, clingen):
        # Forward pass
        #clingen = (1,1,17) - (batch size, 1, embedding_size)

        CG_feat = clingen.reshape(-1, clingen.size(2)) #(1,17)

        #casts the clingen tensor to the same data type as the weight matrix of the self.fc3
        Clingen_feat_encode = F.relu(self.fc3(CG_feat.to(self.fc3[0].weight.dtype))) #(1,128)

        clingen_output = self.classifier(Clingen_feat_encode)

        return clingen_output
        
        
        
        
        
        

