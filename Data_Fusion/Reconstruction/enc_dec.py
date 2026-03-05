import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Encoder Architecture
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        # Define the convolutional layers for each modality
        # Input dimensions
        self.mr_ct_dim = 512
        self.path_dim = 1024
        self.clingen_dim = 17
        # Hidden dimensions
        self.hidden_dim = 128
        self.mmhid = 128
        # Fusion methods
        self.concat = 512 #- 128
        self.dropout_prob = 0.3
    

        # Define fully connected layers with batch normalization
        self.fc1 = nn.Sequential(
            nn.Linear(self.mr_ct_dim, self.hidden_dim), #512 - 128
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.mr_ct_dim, self.hidden_dim), #512 - 128
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.clingen_dim, self.hidden_dim), #14 - 128
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128

        )

        self.fc4 = nn.Sequential(
            nn.Linear(self.path_dim, self.hidden_dim), # 1024 - 128
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), #128 - 128

        )
        
        #ADICONAR RELU
        
        # FOR CONCAT
        self.combined_fc = nn.Sequential(
            nn.Linear(self.concat, self.mmhid), #cat_size - 128
            nn.ReLU(),
            nn.Linear(self.mmhid, self.mmhid), # 128 - 128
            nn.ReLU()
        )
        
        
        
    def forward(self, CT, MR, feature_path, clingen, combinations):
        
        # Define the shapes of your tensors
        CT_feat = CT.squeeze(dim=3)
        MR_feat = MR.squeeze(dim=3)

        
        # (2) Convert x to shape (Batch_size, embedding_size)
        CT_feat = CT_feat.reshape(-1, CT_feat.size(2)) #(1,512)
        MR_feat = MR_feat.reshape(-1, MR_feat.size(2)) #(1,512)
        CG_feat = clingen.reshape(-1, clingen.size(2)) #(1,17)
        teste = CT_feat.clone()


        for i, combination in enumerate(combinations):
            flag_CT = 1 if combination[0] == 1 else 0
            flag_MR = 1 if combination[1] == 1 else 0
            flag_Path = 1 if combination[2] == 1 else 0
            modalities = ["CT", "MR", "Path"]
            dropout_prob = 0.2
            random.shuffle(modalities)
            num_modalities = sum([flag_CT, flag_MR, flag_Path])
            #print(combination, num_modalities, modalities)

            if num_modalities == 2:
                dropout_count = 0  # Keep track of how many modalities have been dropped out
                for modality in modalities:
                    if dropout_count < 1:
                        if modality == "CT" and flag_CT:
                            probability = random.uniform(0, 1)
                            if probability < dropout_prob:
                                CT_feat[i, :] = 0
                                dropout_count += 1
                        elif modality == "MR" and flag_MR:
                            probability = random.uniform(0, 1)
                            if probability < dropout_prob:
                                MR_feat[i, :] = 0
                                #print("MR")
                                dropout_count += 1
                        elif modality == "Path" and flag_Path:
                            probability = random.uniform(0, 1)
                            if probability < dropout_prob:
                                feature_path[i, :] = 0
                                #print("PT")
                                dropout_count += 1
            elif num_modalities == 3:
                dropout_count = 0  # Keep track of how many modalities have been dropped out
                for modality in modalities:
                    if dropout_count < 2:
                        if modality == "CT" and flag_CT:
                            probability = random.uniform(0, 1)
                            if probability < dropout_prob:
                                CT_feat[i, :] = 0
                                #print("CT")
                                dropout_count += 1
                        elif modality == "MR" and flag_MR:
                            probability = random.uniform(0, 1)
                            if probability < dropout_prob:
                                MR_feat[i, :] = 0
                                #print("MR")
                                dropout_count += 1
                        elif modality == "Path" and flag_Path:
                            probability = random.uniform(0, 1)
                            if probability < dropout_prob:
                                feature_path[i, :] = 0
                                #print("PATH")
                                dropout_count += 1
            else:
                # Handle the case when fewer than two modalities exist or when num_modalities is not 2 or 3
                pass

        #Path = (1, 1024)       
        CT_feat_encode = F.relu(self.fc1(CT_feat)) #(1,128)
        MR_feat_encode = F.relu(self.fc2(MR_feat)) #(1,128)
        #casts the clingen tensor to the same data type as the weight matrix of the self.fc3
        Clingen_feat_encode = F.relu(self.fc3(CG_feat.to(self.fc3[0].weight.dtype))) #(1,128)
        PATH_feat_encode = F.relu(self.fc4(feature_path)) #(1,128)


        # Concatenate features from all modalities
        combined_features = torch.cat((CT_feat_encode, MR_feat_encode, PATH_feat_encode, Clingen_feat_encode), dim=1)

        # Pass through the combined fully connected layers
        encoded_features = self.combined_fc(combined_features)
        return encoded_features



# Decoder Architecture
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        # Define fully connected layers to reconstruct each modality
        self.mr_ct_dim = 512
        self.path_dim = 1024
        self.clingen_dim = 17
        # Hidden dimensions
        self.hidden_dim = 128
        self.mmhid = 128
        # Fusion methods
        self.concat = 512 #- 128
        
        self.ct_decoder = nn.Sequential(
            nn.Linear(self.mmhid, self.mmhid),
            nn.ReLU(),
            nn.Linear(self.mmhid, self.mr_ct_dim)
           
        )
        self.mr_decoder = nn.Sequential(
            nn.Linear(self.mmhid, self.mmhid),
            nn.ReLU(),
            nn.Linear(self.mmhid, self.mr_ct_dim)
        )
        self.path_decoder = nn.Sequential(
            nn.Linear(self.mmhid, self.mmhid),
            nn.ReLU(),
            nn.Linear(self.mmhid, self.path_dim)
            
        )
        self.clingen_decoder = nn.Sequential(
            nn.Linear(self.mmhid, self.mmhid),
            nn.ReLU(),
            nn.Linear(self.mmhid, self.clingen_dim)
           
        )

    def forward(self, encoded_features):
        # Decode features for each modality
        reconstructed_ct = self.ct_decoder(encoded_features)
        reconstructed_mr = self.mr_decoder(encoded_features)
        reconstructed_path = self.path_decoder(encoded_features)
        reconstructed_clingen = self.clingen_decoder(encoded_features)

        return reconstructed_ct, reconstructed_mr, reconstructed_path, reconstructed_clingen
