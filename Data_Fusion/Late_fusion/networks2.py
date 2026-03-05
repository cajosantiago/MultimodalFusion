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
        self.mr_ct_dim = 512
        self.path_dim = 1024
        self.clingen_dim = 17
        self.num_classes = 1
        self.hidden_dim = 128
        self.concat_dim = 4
        
        self.weights = nn.ParameterList()
        
        #self.weight = nn.Parameter(torch.randn(1, 4))
        # Define weights as nn.Parameters
        self.weight_fc1 = nn.Parameter(torch.randn(1, 4))
        self.weight_fc2 = nn.Parameter(torch.randn(1, 4))
        self.weight_fc3 = nn.Parameter(torch.randn(1, 4))
        self.weight_fc4 = nn.Parameter(torch.randn(1, 4))
        
        # Define fully connected layers with batch normalization
        self.fc1 = nn.Sequential(
            nn.Linear(self.mr_ct_dim, self.hidden_dim), #512 - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            nn.BatchNorm1d(self.hidden_dim),   # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes) #128 - 1
        )
        
        self.weights.append(nn.Parameter(torch.randn(1)))
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.mr_ct_dim, self.hidden_dim), #512 - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            nn.BatchNorm1d(self.hidden_dim),   # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes) #128 - 1
        )
        self.weights.append(nn.Parameter(torch.randn(1)))
        
        self.fc3 = nn.Sequential(
            nn.Linear(self.path_dim, self.hidden_dim), # 1024 - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            nn.BatchNorm1d(self.hidden_dim),   # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes) #128 - 1
        )
        
        self.weights.append(nn.Parameter(torch.randn(1)))
        
        self.fc4 = nn.Sequential(
            nn.Linear(self.clingen_dim, self.hidden_dim), #14 - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
            nn.BatchNorm1d(self.hidden_dim),   # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes) #128 - 1
        )
        
        self.weights.append(nn.Parameter(torch.randn(1)))
        """
        self.fuse_concat_fc = nn.Sequential(
            nn.Linear(self.concat_dim, self.hidden_dim, bias=False), #cat_size - 128
            nn.BatchNorm1d(self.hidden_dim),  # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias = False), # 128 - 128
            nn.BatchNorm1d(self.hidden_dim),   # Batch normalization
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes, bias = False) #128 - 1
            
        )
        """
        

        
    def forward(self, ct_input, mr_input, path_input, clingen, combinations):
        # Forward pass for each modality
        #CT shape = (batch_size,512,1,1) - (batch size, embedding_size, 1, 1)
        #MR shape = (batch_size,512,1,1) - (batch size, embedding_size, 1, 1)
        #path shape = (1,1024) - (batch size, embedding_size)
        #clingen = (1,1,17) - (batch size, 1, embedding_size)
        
        CT_feat = ct_input.squeeze(dim=3)
        MR_feat = mr_input.squeeze(dim=3)
        
        # (2) Convert x to shape (Batch_size, embedding_size)
        CT_feat = CT_feat.reshape(-1, CT_feat.size(2)) #(1,512)
        MR_feat = MR_feat.reshape(-1, MR_feat.size(2)) #(1,512)
        CG_feat = clingen.reshape(-1, clingen.size(2)) #(1,17)
 
        ct_output = self.fc1(CT_feat)
        mr_output = self.fc2(MR_feat)
        path_output = self.fc3(path_input)
        clingen_output = self.fc4(CG_feat.to(self.fc3[0].weight.dtype))
        
        CT_prob = torch.sigmoid(ct_output)
        MR_prob = torch.sigmoid(mr_output)
        Path_prob = torch.sigmoid(path_output)
        Clingen_prob = torch.sigmoid(clingen_output)
        
        # Print the weights
        print("Weights in FC1:", self.weights[0].item())
        print("Weights in FC2:", self.weights[1].item())
        print("Weights in FC3:", self.weights[2].item())
        print("Weights in FC4:", self.weights[3].item())

        outputs = []
        for i, combination in enumerate(combinations):
            #print("\n*******NEW**********")
            #print(combination)
            
            available_weights_sum = (
                self.weights[0].item() * combination[0] +
                self.weights[1].item() * combination[1] +
                self.weights[2].item() * combination[2] +
                self.weights[3].item() * combination[4]
            )
            
            #print("sum",available_weights_sum)
            
            normalized_weights = [
                self.weights[0].item() * combination[0] / available_weights_sum,
                self.weights[1].item() * combination[1] / available_weights_sum,
                self.weights[2].item() * combination[2] / available_weights_sum,
                self.weights[3].item() * combination[4] / available_weights_sum
            ]
            #print("weights",normalized_weights)
            
            weighted_outputs = [
                normalized_weights[0] * CT_prob[i],
                normalized_weights[1] * MR_prob[i],
                normalized_weights[2] * Path_prob[i],
                normalized_weights[3] * Clingen_prob[i]
            ]
            
            stacked_outputs = sum(weighted_outputs)
            
            stacked_outputs_tensor = stacked_outputs.clone().detach()

            outputs.append(stacked_outputs)
        
        outputs = torch.stack(outputs, dim=0)
        
        
        return ct_output, mr_output, path_output, clingen_output, outputs
