#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

module_path = os.path.abspath(os.path.join('../../Ant_Syn_Scraping/'))
if module_path not in sys.path:
    sys.path.append(module_path)
import model_functions_PhaseI as functions


# In[4]:


#model architecture

class Phase_I_NN(nn.Module):
    """
    This class contains the first of two neural networks to be used to determine synonymy, antonymy or irrelevance.
    Using w2v pre-trained embeddings that are then embedded into our NN using the nn.Embedding layer we are able to
    obtain the encoded embeddings of two words (pushed as a tuple) in synonym and antonym subspaces. These encodings
    are then used to calculate the synonymy and antonymy score of those two words.
    This mimics the Distiller method described by Asif Ali et al.
    """

    def __init__(self, in_dims, out_dims, common): #model1 to be added for w2v encodings
        super(Phase_I_NN, self).__init__()
        
        #embedding layer
        self.embedded = functions.glove_embedding_pre_trained_weights(common) #model1 to be added for w2v encodings after common
        
        #hidden layers
        self.hidden_layers = nn.Sequential(
        nn.Linear(50, 100), #expand
        nn.Linear(100, 300),
        nn.Softplus()
        )
        
        self.S_branch = nn.Sequential( #synonym subspace branch
        nn.Dropout(0.1), #to limit overfitting
        nn.Linear(300,100), #compress
        nn.Linear(100,50),
        nn.Softplus()
        )
        
        self.A_branch = nn.Sequential(#antonym subspace branch
        nn.Dropout(0.1), #to limit overfitting
        nn.Linear(300, 100), #compress
        nn.Linear(100,50),
        nn.Softplus()
        )
        
        #other option is to define activation function here i.e. self.Softplus = torch.nn.Softplus() and use it in the forward pass
        
        
    def forward(self, index_tuple):
        
        em_1, em_2 = self.embedded(index_tuple)[0], self.embedded(index_tuple)[1]
        
        #pass through hidden layers. For each linear layer in the hidden/branches, use the activation function to push
        out_w1 = self.hidden_layers(em_1) 
        out_w2 = self.hidden_layers(em_2)
        
        #pass each embedded data through each branch to be situated in subspaces
        S1_out = self.S_branch(out_w1)
        S2_out = self.S_branch(out_w2)
        A1_out = self.A_branch(out_w1)
        A2_out = self.A_branch(out_w2)
        
        #Need to find a way to collect encoder embeddings as well as their scoring
            
        synonymy_score = F.cosine_similarity(S1_out.view(1,-1),S2_out.view(1,-1), dim=1) #do these outside of the NN class
        antonymy_score = torch.max(F.cosine_similarity(A1_out.view(1,-1),S2_out.view(1,-1)),F.cosine_similarity(A2_out.view(1,-1),S1_out.view(1,-1), dim=1))
                              
        #return synonymy_score, antonymy_score #the encoders in each subspace
        
        return S1_out, S2_out, A1_out, A2_out, synonymy_score, antonymy_score


# In[ ]:




