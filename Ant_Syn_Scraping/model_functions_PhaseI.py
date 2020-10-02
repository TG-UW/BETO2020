#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

import os


# In[ ]:


class Loss_Synonymy(nn.Module):
    """
    This class contains a loss function that uses the sum of ReLu loss to make predictions for the encoded embeddings
    in the synonym subspace. A lower and higher bound for synonymy are to be determined. Need to better understand the
    equation found in the Asif Ali et al. paper.
    """  
    def __init__ (self):
        super(Loss_Synonymy, self).__init__()
        
    def forward(self, S1_out, S2_out, synonymy_score):
        
        result_list = torch.zeros(S2_out.size(0))
        element_count = 0
        
        error_1 = torch.zeros(1,1)
        error_2 = torch.zeros(1,1)
            
        for x, a, b in zip(synonymy_score, S1_out, S2_out): #x=synonymy_score, a=S1_out, b=S2_out
    
            
            if torch.ge(x, torch.tensor(0.8)) == True:
                error = F.relu(torch.add(torch.tensor(1), torch.neg(torch.tanh(torch.dist(a, b, 2))))) #assumed Euclidean Distance
                error_1 = torch.add(error_1,error)
                
            elif torch.lt(x, torch.tensor(0.8)) == True:
                error = F.relu(torch.add(torch.tensor(1), torch.tanh(torch.dist(a, b, 2))))
                error_2 = torch.add(error_2,error)
                
        
        error_total = torch.add(error_1, error_2)
        
        result_list[element_count] = error_total
        element_count += 1
        
        result = result_list.mean()
        
        return result


# In[ ]:


class Loss_Antonymy(nn.Module):
    """
    This class contains a loss function that uses the sum of ReLu loss to make predictions for the encoded embeddings
    in the antonym subspace. A lower and higher bound for antonymy are to be determined. Need to better understand the
    equation found in the Asif Ali et al. paper.
    """
    
    def __init__(self):
        super(Loss_Antonymy, self).__init__()
       
    def forward(self, S2_out, A1_out, antonymy_score): 
        
        result_list = torch.zeros(S2_out.size(0))
        element_count = 0
    
        error_1 = torch.zeros(1,1)
        error_2 = torch.zeros(1,1)
        
        for x, a, b in zip(antonymy_score, A1_out, S2_out): #x=antonymy_score, a=A1_out, b=S2_out (to ensure trans-transitivity)
            
            if torch.ge(x, torch.tensor(0.8)) == True:
                error = F.relu(torch.add(torch.tensor(1), torch.neg(torch.tanh(torch.dist(a, b, 2)))))
                error_1 = torch.add(error_1,error)
                
            elif torch.lt(x, torch.tensor(0.8)) == True:
                error = F.relu(torch.add(torch.tensor(1), torch.tanh(torch.dist(a, b, 2))))
                error_2 = torch.add(error_2, error)
                 
        error_total = torch.add(error_1, error_2)
        
        error_total.retain_grad()
        result_list[element_count] = error_total
        element_count += 1
        
        result = result_list.mean()
        
        return result


# In[ ]:


class Loss_Labels(nn.Module):
    """
    This class is the last portion of the general loss function. Here the predicted synonymy and antonymy scores
    are concatenated and compared to the concatenated labeled synonymy and antonymy scores
    """
    def __init__(self):
        super(Loss_Labels, self).__init__()
       
    def forward(self, S1_out, synonymy_score, antonymy_score):
        
        result_list = torch.zeros(S1_out.size(0))
        element_count = 0
        
        for x, y in zip(synonymy_score, antonymy_score):
            
            #print(synonymy_score)
            #print(antonymy_score)
            error = torch.nn.functional.log_softmax(torch.stack((x,y),dim=0), dim=0)
            error = torch.argmax(error).float()
            #error.squeeze()
            
            error.requires_grad = True
            error.retain_grad()
            result_list[element_count] = error
            element_count += 1
            result = torch.neg(result_list.mean())
    
        return result


# In[1]:


#feeding the model pretrained weights

class w2v_embedding_pre_trained_weights(nn.Module):
    """
    This class contains the pre-training of the Phase_I_NN neural network weights using a list of words from which
    a list of weights can be obtained. It is then converted that can then be embedded using the from_pretrained() 
    function into the NN model
    """
    def __init__(self, words, model):
        super(w2v_embedding_pre_trained_weights, self).__init__()
    
        for i in range(len(words)):
            words[i] = model.wv.__getitem__(words[i]).tolist()
    
        weight = torch.tensor(words)
    
        self.embedding = nn.Embedding.from_pretrained(weight)
    
    def forward(self, index):
        
        index_vector = self.embedding(torch.LongTensor(index))
        
        return index_vector

class glove_embedding_pre_trained_weights(nn.Module):
    """
    This class contains the pre-training of the Phase_I_NN neural network weights using a list of words from which a list of weights can be obtained from a downloaded GloVe embedding dictionary 
    """
    def __init__(self, words):
        super(glove_embedding_pre_trained_weights, self).__init__()
        
        data = '/Users/Thomas/Desktop/glove.6B'
        os.chdir(data)
        embeddings_dict = {}
        with open("glove.6B.50d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
    
        for i in range(len(words)):
            words[i] = embeddings_dict[words[i]].tolist()
    
        weight = torch.tensor(words)
    
        self.embedding = nn.Embedding.from_pretrained(weight)
    
    def forward(self, index):
        
        index_vector = self.embedding(torch.LongTensor(index))
        
        return index_vector


# In[ ]:


#function to find the best fit for our learning rate and epochs
def fit(model, lr, epochs):

    #define the optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
    #empty list to hold loss per epoch
    train_epoch_losses = []
    syn_train_epoch_losses = []
    ant_train_epoch_losses = []
    label_train_epoch_losses = []

    test_epoch_losses = []
    syn_test_epoch_losses = []
    ant_test_epoch_losses = []
    label_test_epoch_losses = []


    syn_test_epoch_accuracies = []
    ant_test_epoch_accuracies = []
    test_epoch_accuracies = []

    #pce_test_epoch_r2 = []
    #voc_test_epoch_r2 = []
    #jsc_test_epoch_r2 = []
    #ff_test_epoch_r2 = []
    #test_epoch_r2s = []

    save_epochs = np.arange(0, num_epochs, 5)

    for epoch in range(num_epochs):
        print('On epoch ', epoch)
    
        #save_dir = "/Users/Thomas/Desktop/BETO2020-Remote/Ant_Syn_Scraping/"
        #model_name = "Phase_I_II_NN"
        #model_path = save_dir+model_name+'*.pt'
        #if epoch < 10:
            #save_path = save_dir + model_name + '_epoch0' + str(epoch) + '.pt'
        #else:
            #save_path = save_dir + model_name + '_epoch' + str(epoch) + '.pt'
        
#     if glob.glob(model_path) != []:
#         model_states = glob.glob(model_path)
#         model_states = sorted(model_states)
#         previous_model = model_states[-1]
        
#         model, optimizer = nuts.load_trained_model(previous_model, model, optimizer)
    
        train_epoch_loss, syn_train_epoch_loss, ant_train_epoch_loss, label_train_epoch_loss = Phase_I_train_model(model = model, training_data_set = training_data_set, optimizer = optimizer)
        train_epoch_losses.append(train_epoch_loss)
        syn_train_epoch_losses.append(syn_train_epoch_loss)
        ant_train_epoch_losses.append(ant_train_epoch_loss)
        label_train_epoch_losses.append(label_train_epoch_loss)


        test_epoch_loss, syn_test_epoch_loss, ant_test_epoch_loss, label_test_epoch_loss, syn_epoch_acc, ant_epoch_acc = Phase_I_eval_model(model = model, testing_data_set = testing_data_set, optimizer = optimizer)

        test_epoch_losses.append(test_epoch_loss)
        syn_test_epoch_losses.append(syn_test_epoch_loss)
        ant_test_epoch_losses.append(ant_test_epoch_loss)
        label_test_epoch_losses.append(label_test_epoch_loss)

        #tot_tst_loss = sum(test_epoch_loss, syn_test_epoch_loss, ant_test_epoch_loss, label_test_epoch_loss)
        #test_epoch_losses.append(tot_tst_loss)

        syn_test_epoch_accuracies.append(syn_epoch_acc)
        ant_test_epoch_accuracies.append(ant_epoch_acc)

        tot_test_acc = (syn_epoch_acc + ant_epoch_acc)
        test_epoch_accuracies.append(tot_test_acc)

        print('Finished epoch ', epoch)

    best_loss_indx = test_epoch_losses.index(min(test_epoch_losses))
    best_acc_indx = test_epoch_accuracies.index(min(test_epoch_accuracies))

    fit_results = {
        'lr': lr,
        'best_loss_epoch': best_loss_indx,
        'best_acc_epoch': best_acc_indx,
        #'best_r2_epoch': best_r2_indx,
        'syn_loss': syn_test_epoch_losses,
        'ant_loss': ant_test_epoch_losses,
        'label_loss': label_test_epoch_losses,
        'test_losses': test_epoch_losses,        
        'syn_acc': syn_test_epoch_accuracies,
        'ant_acc': ant_test_epoch_accuracies,
        'test_accs': test_epoch_accuracies,
        #'pce_r2': pce_test_epoch_r2,
        #'voc_r2': voc_test_epoch_r2,
        #'jsc_r2': jsc_test_epoch_r2,
        #'ff_r2': ff_test_epoch_r2,
        #'test_r2s': test_epoch_r2s,
        'train_syn_loss': syn_train_epoch_losses,
        'train_ant_loss': ant_train_epoch_losses,
        'train_label_loss': label_train_epoch_losses,
    }

    return fit_results


# In[ ]:


#plotting our graphs from the function to find the ideal lr and epoch
def plot_fit_results(fit_dict):
    lr = float(fit_dict['lr'])
    best_loss_epoch = int(fit_dict['best_loss_epoch'])
    best_acc_epoch = int(fit_dict['best_acc_epoch'])
    #best_r2_epoch = int(fit_dict['best_r2_epoch'])
    
    test_loss = [float(i) for i in fit_dict['test_losses']]
    syn_loss = [float(i) for i in fit_dict['syn_loss']]
    ant_loss = [float(i) for i in fit_dict['ant_loss']]
    label_loss = [float(i) for i in fit_dict['label_loss']]
    
    test_acc = [float(i) for i in fit_dict['test_accs']]
    syn_acc = [float(i) for i in fit_dict['syn_acc']]
    ant_acc = [float(i) for i in fit_dict['ant_acc']]
    
    #test_r2 = [float(i) for i in fit_dict['test_r2s']]
    #pce_r2 = [float(i) for i in fit_dict['pce_r2']]
    #voc_r2 = [float(i) for i in fit_dict['voc_r2']]
    #jsc_r2 = [float(i) for i in fit_dict['jsc_r2']]
    #ff_r2 = [float(i) for i in fit_dict['ff_r2']]
    
    train_syn_loss = [float(i) for i in fit_dict['train_syn_loss']]
    train_ant_loss = [float(i) for i in fit_dict['train_ant_loss']]
    train_label_loss = [float(i) for i in fit_dict['train_label_loss']]

    epochs = np.arange(0, (len(test_loss)), 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 6))
    ax1.plot(epochs, pce_loss, c = 'r', label = 'syn loss')
    ax1.plot(epochs, voc_loss, c = 'g', label = 'ant loss')
    ax1.plot(epochs, jsc_loss, c = 'b', label = 'label loss')
    ax1.plot(epochs, test_loss, c = 'c', label = 'total loss')
    ax1.plot(epochs, train_pce_loss, c = 'r', linestyle = '-.', label = 'syn train loss')
    ax1.plot(epochs, train_voc_loss, c = 'g', linestyle = '-.', label = 'ant train loss')
    ax1.plot(epochs, train_jsc_loss, c = 'b', linestyle = '-.', label = 'label train loss')
    ax1.scatter(best_loss_epoch, min(test_loss), alpha = 0.8, s = 64, c = 'turquoise')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Custom Error Loss')
    ax1.legend(loc = 'best')
    ax1.set_title(f'Custom Loss with lr = {lr}')

    ax2.plot(epochs, pce_acc, c = 'r', label = 'syn acc')
    ax2.plot(epochs, voc_acc, c = 'g', label = 'ant acc')
    ax2.plot(epochs, test_acc, c = 'b', label = 'total acc')
    ax2.scatter(best_acc_epoch, min(test_acc), alpha = 0.8, s = 64, c = 'turquoise')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Mean Absolute Percent Accuracy')
    ax2.legend(loc = 'best')
    ax2.set_title(f'Accuracies with lr = {lr}')

    #ax3.plot(epochs, pce_r2, c = 'r', label = 'pce R$^2$')
    #ax3.plot(epochs, voc_r2, c = 'g', label = 'voc R$^2$')
    #ax3.plot(epochs, jsc_r2, c = 'b', label = 'jsc R$^2$')
    #ax3.plot(epochs, ff_r2, c = 'c', label = 'ff R$^2$')
    #ax3.plot(epochs, test_r2, c = 'k', label = 'total R$^2$')
    #ax3.scatter(best_r2_epoch, max(test_r2), alpha = 0.8, s = 64, c = 'turquoise')
    #ax3.set_xlabel('Epochs')
    #ax3.set_ylabel('R$^2$')
    #ax3.legend(loc = 'best')
    #ax3.set_title(f'R$^2$ with lr = {lr}')
    
    plt.tight_layout()
    plt.show()


# In[ ]:


#Network Utilities

def init_weights(model):
    
    classname = model.__class__.__name__
    
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        
    elif classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)


# In[ ]:


#def plot_error_accuracy
 #   fig, ax = plt.subplots(figsize = (8,6))

  #  epochs = np.arange(1, (num_epochs+1), 1)
#
 #   plt.plot(epochs, train_epoch_losses, c = 'k', label = 'training error')
  ## plt.legend(loc = 'upper right')
    #plt.title("Total Training & Testing Error")
    #ax.set_xlabel('Epoch')
    #ax.set_ylabel('Total Custom Loss')
    #plt.show()

    #fig, ax = plt.subplots(figsize = (8,6))
    #plt.plot(epochs, syn_test_epoch_accuracies, c = 'k', label = 'syn accuracy')
    #plt.plot(epochs, ant_test_epoch_accuracies, c = 'r', label = 'ant accuracy')
    #plt.legend(loc = 'lower right')
    #plt.title("Phase I Labeling Accuracy")
    #ax.set_xlabel('Epoch')
    #ax.set_ylabel('Accuracy')
    #plt.show()

