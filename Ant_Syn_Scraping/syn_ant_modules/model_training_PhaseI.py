import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import model_functions_PhaseI as functions


def Phase_I_train_model(model, training_data_set, optimizer):
    
    train_losses = []
    syn_train_losses = []
    ant_train_losses = []
    Lm_train_losses = []
    
    train_total = 0
    
    #switch model to training mode
    model.train()
    
    syn_criterion = functions.Loss_Synonymy()
    ant_criterion = functions.Loss_Antonymy()
    Lm_criterion = functions.Loss_Labels()
    
    for features, labels in training_data_set:
                
        features = torch.from_numpy(np.asarray(features)).long()
        labels = torch.from_numpy(np.asarray(labels)).long()
        
        batch_syn_loss = torch.zeros(len(features))
        batch_ant_loss = torch.zeros(len(features))
        batch_Lm_loss = torch.zeros(len(features))
      
        model.zero_grad() #zero out any gradients from prior loops 
                                           
        #gather model predictions for this loop
        S1_out, S2_out, A1_out, A2_out, synonymy_score, antonymy_score = model(features)
                        
        #calculate error in the predictions. torch.mul() to weight a loss type
        syn_loss = syn_criterion(S1_out, S2_out, synonymy_score)
#         print(syn_loss)
        ant_loss = ant_criterion(S2_out, A1_out, antonymy_score)
#         print(ant_loss)
        Lm_loss = Lm_criterion(S1_out, synonymy_score, antonymy_score)
#         print(Lm_loss)
            
        total_loss = syn_loss + ant_loss + Lm_loss
            
        #save total loss for this batch
        train_losses.append(total_loss.item())
        train_total+=1
        
        #store batch losses in lists
        syn_train_losses.append(syn_loss)
        ant_train_losses.append(ant_loss)
        Lm_train_losses.append(Lm_loss)
        
        #BACKPROPAGATE LIKE A MF
        torch.autograd.backward([syn_loss, ant_loss, Lm_loss])
        optimizer.step()
       
    #calculate total error for this epoch of training
    train_epoch_loss = sum(train_losses)/train_total
    syn_train_epoch_loss = sum(syn_train_losses)/train_total
    ant_train_epoch_loss = sum(ant_train_losses)/train_total
    Lm_train_epoch_loss = sum(Lm_train_losses)/train_total
    
    return train_epoch_loss, syn_train_epoch_loss, ant_train_epoch_loss, Lm_train_epoch_loss

