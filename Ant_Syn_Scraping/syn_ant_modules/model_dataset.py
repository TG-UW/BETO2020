import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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


class DistillerDataset(Dataset):
    """
    Dataset class for Phase I (Distiller), which encodes word pairs into
    synonym and antonym subspaces. Takes in pandas.DataFrames() of the word
    pairs, their labels, and vocabulary indices. It then returns a 
    torch.utils.data.Dataset()
    """
    
    def __init__(self, word_pairs, labels, indices, path):
        
        if os.path.exists(path) == False: #first time using dataset or train-test-split
            
            self.word_pairs = word_pairs
            self.labels = labels
            self.indices = indices
            self.index_pairs = pd.DataFrame(columns = ['word 1', 'word 2'])

            pbar = tqdm(total = len(self.word_pairs), position = 0)

            for i in range(len(self.word_pairs)):

                word1 = self.word_pairs['word 1'].iloc[i]
                word2 = self.word_pairs['word 2'].iloc[i]

                index1 = self.get_index(word1)
                index2 = self.get_index(word2)

                self.index_pairs.loc[i] = pd.Series({'word 1':index1, 'word 2':index2}) 

                pbar.update()
                
            self.index_pairs.to_json(path)
            self.word_pairs.to_json(path[:-5]+'_words.json')
            self.labels.to_json(path[:-5]+'_labels.json')
                
        else: #okay to use previously created train-test-split
            
            self.index_pairs = pd.read_json(path)
            self.word_pairs = pd.read_json(path[:-5]+'_words.json')
            self.labels = pd.read_json(path[:-5]+'_labels.json')
        
    def __len__(self):
        
        self.len = self.data.shape[0]
        
        return self.len
    
    
    def __getitem__(self, key):
        
        index1 = self.index_pairs['word 1'].iloc[key]
        index2 = self.index_pairs['word 2'].iloc[key]
        
        index_pair = (index1, index2)
        
        return index_pair, self.labels[key]
    
    
    def get_index(self, word):
        
        index = self.indices.loc[self.indices['word'] == word]
        
        return index
        

def generate_indices(word_pairs_df):
    """
    Helper function to generate a common set of indices for a
    given list of word pairs.
    """
    
    indices = pd.DataFrame(columns = ['index', 'word'])

    index = 0

    pbar = tqdm(total = len(word_pairs_df), position = 0)

    for i in range(len(word_pairs_df)):

        word1 = word_pairs_df['word 1'].iloc[i]
        word2 = word_pairs_df['word 2'].iloc[i]

        if word1 not in indices['word']:
            indices.loc[index] = pd.Series({'index':index, 'word':word1})
            index+=1
        else:
            pass

        if word2 not in indices['word']:
            indices.loc[index] = pd.Series({'index':index, 'word':word2})
            index+=1
        else:
            pass

        pbar.update()
        
    return indices