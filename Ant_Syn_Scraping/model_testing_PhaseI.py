#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timeit import default_timer as timer

import os
import sys

module_path = os.path.abspath(os.path.join('../../Ant_Syn_Scraping/'))
if module_path not in sys.path:
    sys.path.append(module_path)
import model_functions_PhaseI as functions


# In[4]:


def Phase_I_eval_model(model, testing_data_set, optimizer):
    #evaluate the model
    model.eval()
    
    syn_criterion = functions.Loss_Synonymy()
    ant_criterion = functions.Loss_Antonymy()

    #don't update nodes during evaluation b/c not training
    with torch.no_grad():
        test_losses = []
        syn_test_losses = []
        ant_test_losses = []
        
        syn_test_acc_list = []
        ant_test_acc_list = []
        
        test_total = 0
        ant_el_count = 0
        ant_correct = 0
        syn_el_count = 0
        syn_correct = 0
        
        #for confusion matrix
        
        syn_predictions = []
        ant_predictions = []
        
        syn_true = []
        ant_true = []

        for i, (inputs, labels) in enumerate(testing_data_set):
        
            inputs, labels = torch.from_numpy(np.asarray(inputs)).long(), torch.from_numpy(np.asarray(labels)).long()
        
            for u in range(len(inputs)):
            
                S1_out, S2_out, A1_out, A2_out, synonymy_score, antonymy_score = model(inputs[u])
                
                #calculate loss per batch of testing data
                syn_test_loss = syn_criterion(S1_out, S2_out, synonymy_score) #torch.mul() (see training) to vary loss function weights
                #print test losses
                ant_test_loss = ant_criterion(S2_out, A1_out, antonymy_score)
            
                test_loss = syn_test_loss + ant_test_loss
            
                test_losses.append(test_loss.item())
                syn_test_losses.append(syn_test_loss.item())
                ant_test_losses.append(ant_test_loss.item())
                test_total += 1 
        
                #accuracy function
                
                x = synonymy_score.item()
                y = labels[u,0].item()
                    
                if y == 1 and y*0.8 <= x: #include elif portions to full accuracy. W/O we only test synonym accuracy.
                    syn_correct += 1
                    syn_el_count += 1
                
                #elif y == 0 and -0.8 < x < 0.8:
                 #   syn_correct += 1
                  #  syn_el_count +=1
                
                #elif y == -1 and y*0.8 >= x:
                 #   syn_correct += 1
                  #  syn_el_count += 1

                else:
                    syn_el_count += 1
            
                a = antonymy_score.item() 
                b = labels[u,1].item()
                
                if b == 1 and b*0.8 <= a: #include elif portions to full accuracy. W/O we only test antonym accuracy.
                    ant_correct += 1
                    ant_el_count += 1
                
                #elif b == 0 and -0.8 < b < 0.8:
                 #   ant_correct += 1 
                  #  ant_el_count += 1
                    
                #elif b == -1 and b*0.8 >= a:
                 #   ant_correct += 1
                  #  ant_el_count += 1

                else:
                    ant_el_count += 1
      
        
                syn_acc = (syn_correct/syn_el_count) * 100
                syn_test_acc_list.append(syn_acc)
        
                ant_acc = (ant_correct/ant_el_count) * 100
                ant_test_acc_list.append(ant_acc)
                
                syn_predictions.append(synonymy_score.item())
                ant_predictions.append(antonymy_score.item())
                
                syn_true.append(labels[u,0].item())
                ant_true.append(labels[u,1].item())

        test_epoch_loss = sum(test_losses)/test_total
        syn_test_epoch_loss = sum(syn_test_losses)/test_total
        ant_test_epoch_loss = sum(ant_test_losses)/test_total
        
        syn_epoch_acc = sum(syn_test_acc_list)/test_total
        ant_epoch_acc = sum(ant_test_acc_list)/test_total


        print(f"Total Epoch Testing Loss is: {test_epoch_loss}")
        print(f"Total Epoch Antonym Testing Accuracy is: {ant_epoch_acc}")
        print(f"Total Epoch Synonym Testing Accuracy is: {syn_epoch_acc}")
       
        
    
    return test_epoch_loss, syn_test_epoch_loss, ant_test_epoch_loss, syn_epoch_acc, ant_epoch_acc, syn_true, syn_predictions, ant_true, ant_predictions


# In[ ]:




