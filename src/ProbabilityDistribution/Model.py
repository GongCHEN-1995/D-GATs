import numpy as np
import copy
import json
from typing import Optional, Tuple, List

import torch
from torch import Tensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.parameter import Parameter
from src.FineTuning.Model import GAT

class DenseLayer(nn.Module):
    def __init__(self, input_dim=128, output_dim=128, activation='relu', leakyrelu=0.1):
        super(DenseLayer, self).__init__()
        self.model_type = 'DenseLayer'
        self.linear1 = Linear(input_dim, output_dim)
        self.linear2 = Linear(output_dim, output_dim)
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
        elif activation == 'softplus':
            self.activation = nn.Softplus()
    def forward(self, inputs:Tensor):
        output = self.linear2(self.activation(self.linear1(inputs)))
        return output
    
class MP(nn.Module):
    def __init__(self, nb_MP, input_dim=128):
        super(MP, self).__init__()
        self.model_type = 'MP'
        self.linear1 = Linear(input_dim, nb_MP)
            
    def forward(self, inputs:Tensor):
        output = self.linear1(inputs)
        return output
    
class PB_Prediction(nn.Module):
    def __init__(self, nb_PD, input_dim=128):
        super(PB_Prediction, self).__init__()
        self.model_type = 'PB_Prediction'
        self.linear1 = Linear(input_dim, nb_PD)
            
    def forward(self, inputs:Tensor):
        output = self.linear1(inputs)
        return output
    
class MolecularProperties(nn.Module):
    def __init__(self, nb_node_features, nb_edge_features, nb_MP, nb_PD, output_dim=128, d_model=128, dropout=0.1, num_heads=1, nb_layer=1, activation='relu', leakyrelu=0.1):
        super(MolecularProperties, self).__init__()
        self.model_type = 'MolecularProperties'
        self.GAT = GAT(nb_node_features, nb_edge_features, d_model=d_model, dropout=dropout, num_heads=num_heads, nb_layer=nb_layer, activation=activation, leakyrelu=leakyrelu)
        self.DenseLayer = DenseLayer((nb_layer+1)*d_model, output_dim, activation=activation, leakyrelu=leakyrelu)
        self.MP = MP(nb_MP, output_dim)  
        self.PB_mu = PB_Prediction(nb_PD, output_dim)
        self.PB_sigma = PB_Prediction(nb_PD, output_dim)
        self.sigma_activation = nn.Softplus()
        
    def forward(self, inputs):
        _, CLS = self.GAT(inputs['nodes_features'], inputs['edges_features'])
        CLS = self.DenseLayer(CLS)
        outputs_MP = self.MP(CLS) # molecular properties / features
        outputs_mu = self.PB_mu(CLS) 
        outputs_sigma = self.PB_sigma(CLS) #+ 1e-10 # should be positive and cannot be zero!
        
        return outputs_MP, outputs_mu, self.sigma_activation(outputs_sigma)

def D_GAT(dataset, mol, config_file_path):
    nb_node_features, nb_edge_features, nb_MP, nb_PD = mol[0].mdoel_needed_info()  
    mol_config = json.load(open(config_file_path,'r'))
    d_model = mol_config["d_model"]
    output_dim = mol_config["output_dim"]
    num_heads = mol_config["num_heads"]
    activation = mol_config["activation"]
    PreTraining_model_path = mol_config["PreTraining_model_path"]
    dropout = mol_config["dropout"]
    nb_layer = mol_config["nb_layer"]
    
    model = MolecularProperties(nb_node_features, nb_edge_features-2, nb_MP, nb_PD, output_dim=output_dim, d_model=d_model, dropout=dropout, num_heads=num_heads, nb_layer=nb_layer,activation=activation,leakyrelu=0.1)
    try:
        model.load_state_dict(torch.load(PreTraining_model_path))
        print('Load FineTuning mdoel: ', PreTraining_model_path)
    except:
        Pretrained_model = torch.load(PreTraining_model_path)
        model_dict =  model.state_dict()  
        state_dict = {k:v for k,v in Pretrained_model.items() if k in model_dict and v.shape==model_dict[k].shape}  
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print('Load PreTraining mdoel: ', PreTraining_model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    print('Model prepared!')
    
    best_score = [[1e20, 1e20] for i in range(3)]
    return model, best_score

class PD_NLL_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(PD_NLL_Loss, self).__init__()
        self.weight = weight.view(1, 1, -1) if weight else None # weight.shape = 1, 1, num_of_properties
        self.size_average = size_average
        
    def forward(self, mu, sigma, PD):
        # mu.shape      = bsz, num_of_properties
        # sigma.shape   = bsz, num_of_properties
        # PD.shape      = bsz, exp_times, num_of_properties
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        loss = torch.log(sigma) + (PD - mu).pow(2) / 2 / sigma.pow(2)
        if self.weight:
            loss = loss * self.weight
        
        loss = loss.reshape(-1)
        if self.size_average:
            loss = torch.mean(loss)
        else:
            loss = torch.sum(loss)
        return loss