#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from gensim.models import Word2Vec as wv

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import math


# In[ ]:





# In[ ]:





# In[ ]:


#dataset creation
#train dataset

class Phase_I_Train_Dataset(Dataset):
    
    def __init__(self):
        
        data = pd.read_json('Phase_I_Train.json', dtype = np.float32)
        self.len = data.shape[0]
        
        data_x = list(zip(data['word 1 index'], data['word 2 index'])) #creating a list of tuples where [w1,w2] and [ss, as]
        data_y = list(zip(data['syn score'], data['ant score']))
        
        #split into x_data our features and y_data our targets
        self.x_data = torch.tensor(data_x)
        self.y_data = torch.tensor(data_y)
        
    def __len__(self):
        
        return self.len
    
    def __getitem__(self, index):
        
        return self.x_data[index], self.y_data[index]


# In[ ]:


#test dataset

class Phase_I_Test_Dataset(Dataset):
    
    def __init__(self):
        
        data = pd.read_json('Phase_I_Test.json', dtype = np.float32)
        self.len = data.shape[0]
        
        data_x = list(zip(data['word 1 index'], data['word 2 index'])) #creating a list of tuples where [w1,w2] and [ss, as]
        data_y = list(zip(data['syn score'], data['ant score']))
            
        #split into x_data our features and y_data our targets
        self.x_data = torch.tensor(data_x)
        self.y_data = torch.tensor(data_y)
        
    def __len__(self):
        
        return self.len
    
    def __getitem__(self, index):
        
        return self.x_data[index], self.y_data[index]

