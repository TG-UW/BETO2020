#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import sys

module_path = os.path.abspath(os.path.join('../../Ant_Syn_Scraping/'))
if module_path not in sys.path:
    sys.path.append(module_path)
import model_functions_PhaseI as functions


# In[1]:


def Phase_I_train_model(model, training_data_set, optimizer):
    
    train_losses = []
    syn_train_losses = []
    ant_train_losses = []
    
    train_epoch_loss = []
    syn_train_epoch_loss = []
    ant_train_epoch_loss = []
    
    train_total = 0
    
    #switch model to training mode
    model.train()
    
    syn_criterion = functions.Loss_Synonymy()
    ant_criterion = functions.Loss_Antonymy()
    
    for i, (features, labels) in enumerate(training_data_set):
        
        #features, labels = data
        
        features, labels = torch.from_numpy(np.asarray(features)).long(), torch.from_numpy(np.asarray(labels)).long()
      
        model.zero_grad() #zero out any gradients from prior loops 
        
        for u in range(len(features)):
                       
            S1_out, S2_out, A1_out, A2_out, synonymy_score, antonymy_score = model(features[u]) #gather model predictions for this loop
        
            #calculate error in the predictions
            syn_loss = syn_criterion(S1_out, S2_out, synonymy_score) #torch.mul(syn_criterion(), mul_factor) to change weight
            ant_loss = torch.mul(ant_criterion(S2_out, A1_out, antonymy_score),2)
            total_loss = syn_loss + ant_loss
            
            #save loss for this batch
            train_losses.append(total_loss.item())
            train_total+=1
        
            syn_train_losses.append(syn_loss.item())
            ant_train_losses.append(ant_loss.item())
        
        #BACKPROPAGATE LIKE A MF
        torch.autograd.backward([syn_loss, ant_loss])
        optimizer.step()
       
    #calculate and save total error for this epoch of training
    epoch_loss = sum(train_losses)/train_total
    train_epoch_loss.append(epoch_loss)
    
    syn_train_epoch_loss.append(sum(syn_train_losses)/train_total)
    ant_train_epoch_loss.append(sum(ant_train_losses)/train_total)
    
    return train_epoch_loss, syn_train_epoch_loss, ant_train_epoch_loss


# In[ ]:




