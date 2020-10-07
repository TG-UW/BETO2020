import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset

#dataset creation
#train dataset

class Phase_I_Train_Dataset(Dataset):
    
    def __init__(self):
        
        data = pd.read_json('./data/Phase_I_Train.json', dtype = np.float32)
        self.len = data.shape[0]
        
        #creating a list of tuples where [w1,w2] and [ss, as]
        data_x = list(zip(data['word 1 index'], data['word 2 index']))
        data_y = list(zip(data['syn score'], data['ant score']))
        
        #split into x_data our features and y_data our targets
        #F.soft_max expects 'float' predictions and 'long' labels
        self.x_data = torch.tensor(data_x)
        self.x_data = self.x_data.type(torch.float)
        
        self.y_data = torch.tensor(data_y)
        self.y_data = self.y_data.type(torch.long)
        
    def __len__(self):
        
        return self.len
    
    def __getitem__(self, index):
        
        return self.x_data[index], self.y_data[index]


#test dataset

class Phase_I_Test_Dataset(Dataset):
    
    def __init__(self):
        
        data = pd.read_json('./data/Phase_I_Test.json', dtype = np.float32)
        self.len = data.shape[0]
        
        #creating a list of tuples where [w1,w2] and [ss, as]
        data_x = list(zip(data['word 1 index'], data['word 2 index']))
        data_y = list(zip(data['syn score'], data['ant score']))
            
        #split into x_data our features and y_data our targets
        #F.soft_max expects 'float' predictions and 'long' labels
        self.x_data = torch.tensor(data_x)
        self.x_data = self.x_data.type(torch.float)
        
        self.y_data = torch.tensor(data_y)
        self.y_data = self.y_data.type(torch.long)
        
    def __len__(self):
        
        return self.len
    
    def __getitem__(self, index):
        
        return self.x_data[index], self.y_data[index]

