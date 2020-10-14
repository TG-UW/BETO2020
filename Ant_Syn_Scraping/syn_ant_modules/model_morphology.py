import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import model_functions_PhaseI as functions

#model architecture

class Phase_I_NN(nn.Module):
    """
    This class contains the first of two neural networks to be used to determine synonymy,
    antonymy or irrelevance. Using w2v pre-trained embeddings that are then embedded into
    our NN using the nn.Embedding layer we are able to obtain the encoded embeddings of two
    words (pushed as a tuple) in synonym and antonym subspaces. These encodings are then used
    to calculate the synonymy and antonymy score of those two words. 
    
    This mimics the Distiller method described by Asif Ali et al.
    """

    def __init__(self, in_dims, vocab_length, common, w2v_model):
        super(Phase_I_NN, self).__init__()
        
        if common != None:
            #embedding layer
            self.embedded = functions.w2v_embedding_pre_trained_weights(common, w2v_model)
            
        else:
            #take in word index and generate random embedding
            self.embedded = nn.Embedding(num_embeddings = vocab_length,
                                         embedding_dim = in_dims)
        
        #hidden layers
        self.hidden_layers = nn.Sequential(
        nn.Linear(in_dims, 800), #expand
        nn.Linear(800, 1000),
#         nn.Softplus()
        )
        
        #synonym subspace branch
        self.S_branch = nn.Sequential(
#         nn.Dropout(0.1), #to limit overfitting
        nn.Linear(1000,500), #compress
        nn.Linear(500,200),
#         nn.Softplus()
        )
        
        #antonym subspace branch
        self.A_branch = nn.Sequential(
#         nn.Dropout(0.1), #to limit overfitting
        nn.Linear(1000,500), #compress
        nn.Linear(500,200),
#         nn.Softplus()
        )
        
    def forward(self, index_tuples):
                        
        #for every word pair in the training batch, get (pre-trained embeddings) from
        #the index. em_1 is tensor of first words, em_2 is tensor of second words
        em_1 = self.embedded(index_tuples[:,0])
        em_2 = self.embedded(index_tuples[:,1])
        
        #pass through hidden layers
        out_w1 = self.hidden_layers(em_1) 
        out_w2 = self.hidden_layers(em_2)
                
        #pass each embedded word-pair through each branch to be situated in subspaces
        S1_out = self.S_branch(out_w1)
        S2_out = self.S_branch(out_w2)
        A1_out = self.A_branch(out_w1)
        A2_out = self.A_branch(out_w2)
                
        #calculate synonymy and antonymy scores
        synonymy_score = F.cosine_similarity(S1_out, S2_out, dim = 1)
        synonymy_score = synonymy_score.view(-1, 1)
        antonymy_score = torch.max(F.cosine_similarity(A1_out, S2_out, dim = 1),
                                   F.cosine_similarity(A2_out, S1_out, dim = 1))
        antonymy_score = antonymy_score.view(-1, 1)
                                      
        return S1_out, S2_out, A1_out, A2_out, synonymy_score, antonymy_score



class Phase_II_NN(nn.Module):
    """
    This NN takes in the synonymy and antonymy scores distilled by Phase_I_NN and uses
    them to predict synonymy and antonymy classification for each word pair.
    
    0 = irrelevant pair
    1 = synonymous pair
    2 = antonymous pair
    """
    def __init__(self, s1_out, s2_out, a1_out, a2_out):
        super(Phase_II_NN).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(2, 100),
            nn.Linear(100, 10),
            nn.Linear(10, 3),
            nn.Softplus()
        )
        
        
    def forward(self, syn_score, ant_score):
        
        tensor = torch.cat(syn_score, ant_score, dim = 1)
        classification = self.classifier(tensor)
        
        return classification
        
        