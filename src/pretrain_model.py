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
    
from src.model import GAT

class Predictor(nn.Module):
    def __init__(self, nb_output, d_model_in=128, d_model_hidden=128, activation='relu', leakyrelu=0.1):
        super(Predictor, self).__init__()
        self.model_type = 'Predictor'
        self.linear1 = Linear(d_model_in, d_model_hidden)
        self.linear2 = Linear(d_model_hidden, nb_output)
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
            
    def forward(self, inputs:Tensor):
        output = self.linear2(self.activation(self.linear1(inputs)))
        return output
        
class PreTraingModel(nn.Module):
    def __init__(self, nb_node_features, nb_edge_features, nb_MP, nb_ele, d_model=128, dropout=0.1, num_heads=1, nb_layer=1, activation='relu', leakyrelu=0.1):
        super(PreTraingModel, self).__init__()
        self.model_type = 'PreTraingModel'
        self.d_model = d_model
        self.GAT = GAT(nb_node_features, nb_edge_features, d_model=d_model, dropout=dropout, num_heads=num_heads, nb_layer=nb_layer, activation=activation, leakyrelu=leakyrelu)
        self.MP = Predictor(nb_MP, (nb_layer+1)*d_model, d_model, activation, leakyrelu)
        self.ring = Predictor(1, d_model, d_model, activation, leakyrelu)
        self.aromatic = Predictor(1, d_model, d_model, activation, leakyrelu)
        self.element = Predictor(nb_ele, d_model, d_model, activation, leakyrelu)
        self.degree = Predictor(7, d_model, d_model, activation, leakyrelu)
        self.hybridization = Predictor(8, d_model, d_model, activation, leakyrelu)
        self.chirality = Predictor(4, d_model, d_model, activation, leakyrelu)
        self.H = Predictor(5, d_model, d_model, activation, leakyrelu)
        self.formal_charge = Predictor(1, d_model, d_model, activation, leakyrelu)
        self.radical_electrons = Predictor(1, d_model, d_model, activation, leakyrelu)
            
    def forward(self, inputs):
        outputs = {}
        node_x, CLS = self.GAT(inputs['nodes_features'], inputs['edges_features'])
        outputs['MP'] = self.MP(CLS)
        node_x = node_x.reshape(-1,self.d_model)
        selected_nodes = torch.index_select(node_x, 0, inputs['node_index'])
        outputs['ring'] = self.ring(selected_nodes)
        outputs['aromatic'] = self.aromatic(selected_nodes)
        outputs['element'] = self.element(selected_nodes)
        outputs['degree'] = self.degree(selected_nodes)
        outputs['hybridization'] = self.hybridization(selected_nodes)
        outputs['chirality'] = self.chirality(selected_nodes)
        outputs['H'] = self.H(selected_nodes)
        outputs['formal charge'] = self.formal_charge(selected_nodes)
        outputs['radical electrons'] = self.radical_electrons(selected_nodes)
        return outputs

def D_GAT(mol, config_file_path):
    nb_node_features, nb_edge_features, nb_MP, nb_ele = mol[0].mdoel_needed_info() 
    mol_config = json.load(open(config_file_path,'r'))
    d_model = mol_config["d_model"]
    dropout = mol_config["dropout"]
    num_heads = mol_config["num_heads"]
    nb_layer = mol_config["nb_layer"]
    activation = mol_config["activation"]
    PreTraining_model_path = mol_config["PreTraining_model_path"]
    
    model = PreTraingModel(nb_node_features, nb_edge_features-2, nb_MP, nb_ele, d_model=d_model, dropout=dropout, num_heads=num_heads, nb_layer=nb_layer,activation=activation,leakyrelu=0.1)
    if PreTraining_model_path:
        model.load_state_dict(torch.load(PreTraining_model_path))
        print('Load PreTraining model: ', PreTraining_model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    print('Model prepared!')
    
    best_score = np.zeros((3,17))
    best_score[1,0] = 1e10

    return model, best_score