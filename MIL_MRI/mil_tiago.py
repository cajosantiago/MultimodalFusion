import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

class MlpCLassifier(nn.Module):

    def __init__(self, in_features, hidden_size=256 , out_features=2, dropout=0.0):
        super(MlpCLassifier, self).__init__()
        self.drop = dropout
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_features)
        )
    """
    def __init__(self, in_features, out_features = 2, dropout=0.0):
        super(MlpCLassifier, self).__init__()
        self.drop = dropout
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2)
        )
    """
    def forward(self, x):
        """ 
        Forward pass of the MlpClassifier model.
            Input: x (Batch_size, in_features)
            Ouput: x (Batch_size, out_features)
        Note: Dropout layer was done this so the architecture can be saved and loaded without errors if
        the user uses or not dropout.  
        """
        if self.drop:
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.mlp(x)


       
class InstanceMIL(nn.Module):
    
    def __init__(self, 
                 num_classes = 2,
                 N=47, 
                 embedding_size=512, 
                 dropout=0.1,
                 pooling_type=None,
                 args=None,
                 device="cuda",
                 patch_extractor:nn.Module = None):
        
        super(InstanceMIL, self).__init__()
        self.patch_extractor = patch_extractor   
        self.deep_classifier = MlpCLassifier(in_features=embedding_size, hidden_size = 256,  out_features=num_classes, dropout=dropout)    
        self.Softmax = nn.Softmax(dim=2)
        self.N = N     
        self.num_classes = num_classes   
        #self.pooling_type = pooling_type.lower()
        self.args = args
        self.device = device
        self.patch_scores = None
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
        
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.patch_extractor(x)
    
    def save_patch_scores(self, x):
        self.patch_scores = x
        
    def get_patch_scores(self):
        return self.patch_scores
    
    def MaxPooling(self, scores):
        """ Classical MIL: The representation of a given bag is given by the maximum probability
        of 'MEL' case. If the probability of 'MEL' is higher than the probability .5 then the bag is classified as 'MEL'. 
        Args:
            scores (torch.Tensor): Scores of each instance in the bag. Shape (Batch_size, N, num_classes).
        Returns:
            torch.Tensor: Pooled scores. Shape (Batch_size, num_classes). 
        """
        pooled_probs, position = torch.max(scores, dim=1) # Get the maximum probability of the class died (1) per bag.
        
        
        return pooled_probs, position
    
                
    def forward(self, x, numb_fmap):
        '''
        Forward pass of the InstanceMIL model.
            Input: x (Batch_size, bag_size, 512, 1)
            Ouput: x (Batch_size, num_classes)
        '''
        #torch.set_printoptions(threshold=sys.maxsize)
        #np.set_printoptions(threshold=np.inf)
        #print("numb_fmap", numb_fmap)
        # Register Hook
        if x.requires_grad == True:
            x.register_hook(self.activations_hook)
        
        # (2) Transform input to shape(batch_size, N, embedding_size)
        x = x.squeeze(dim=3)
    
        # (3) Convert x to shape (Batch_size*N, embedding_size)
        x = x.reshape(-1, x.size(2))
        
        # (4) Use a deep classifier to obtain the score for each instance
        x = self.deep_classifier(x)
        # Deep classifier output shape: (12,2)
        
        # (5) Transform x to shape (Batch_size, N, num_classes)
        x = x.view(-1, self.N, self.num_classes)
        #View output shape: torch.Size([1, 12, 2])
        #print("\n SCORES:", x.shape, x)
        
        # Save the scores for each patch
        self.save_patch_scores(x)
        
       
        
        # (6) Apply softmax to the scores
        #x = self.Softmax(x)
       
        #print("\n Softmax:",x.shape, x)
        #mask = (torch.arange(self.N).unsqueeze(0).to(x.device) <= numb_fmap.unsqueeze(1).expand(x.size(0), self.N)).unsqueeze(2)
        #print("mask", mask)
        #x = x * mask
        #print("numb fmap", numb_fmap)
        #print("\n after mask:",x.shape, x)
        
        #(7) Apply pooling to obtain the bag representation
        x, position = self.MaxPooling(x[:,:numb_fmap,1])
        #print("\n POOLING:", x.shape, x)
        
        #x is the max probability of being classified as 1.
        
        #print("position", position)
        
        return x, position #(Batch_size, num_classes)
    