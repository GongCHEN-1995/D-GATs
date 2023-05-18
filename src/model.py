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

# module for Graph Attention Network
class EDGEAttention(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, activation='relu', leakyrelu=0.1, bias=True):
        super(EDGEAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == self.d_model, "d_model must be divisible by num_heads"
        
        self.in_proj_weight1 = Parameter(torch.empty(d_model, 3 * d_model))

        if bias:
            self.in_proj_bias1 = Parameter(torch.empty(3 * d_model))
        else:
            self.register_parameter('in_proj_bias1', None)
            
        self._reset_parameters()
        
        self.dropout = nn.Dropout(dropout)
        self.linear1_edge = Linear(d_model, d_model)
        self.linear2_edge = Linear(d_model, d_model)
        self.norm1_edge = nn.modules.normalization.LayerNorm(d_model)
        self.norm2_edge = nn.modules.normalization.LayerNorm(d_model)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
            
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight1)
        if self.in_proj_bias1 is not None:
            nn.init.constant_(self.in_proj_bias1, 0.)
    
    def _MHA_linear(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        output = inputs.matmul(weight)
        if bias is not None:
            output += bias
        return output

    def forward(self, edge_x: Tensor, edge_mask: Tensor) -> Tensor:
        edge_x = edge_x.transpose(0,1) #edge_len, bsz, d_model
        bsz = edge_x.size(1)
        edge_len = edge_x.size(0)
        scaling = float(self.head_dim) ** -0.5

        # self-attention fot edge_x
        q_edge, k_edge, v_edge = self._MHA_linear(edge_x, self.in_proj_weight1, self.in_proj_bias1).chunk(3, dim=-1) 

        q_edge = q_edge.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,edge_len,head_dim
        k_edge = k_edge.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,edge_len,head_dim
        v_edge = v_edge.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,edge_len,head_dim

        attn_output_weights = torch.bmm(q_edge, k_edge.transpose(1, 2)) #bsz*num_heads, edge_len, edge_len
        attn_output_weights = attn_output_weights * scaling
        attn_output_weights = attn_output_weights.reshape(bsz, self.num_heads, edge_len, edge_len)
        attn_output_weights = attn_output_weights.masked_fill(edge_mask.unsqueeze(1),float("-inf"))
        attn_output_weights = attn_output_weights.view(bsz * self.num_heads, edge_len, edge_len) #bsz*num_heads, edge_len, edge_len
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)  #bsz*num_heads, edge_len, edge_len
        attn_output_weights =  self.dropout(attn_output_weights)
        
        attn_output = torch.bmm(attn_output_weights, v_edge)  # size = bsz*num_heads, edge_len, head_dim  
        edge_x2 = attn_output.transpose(0, 1).reshape(edge_len, bsz, self.d_model)
        
        edge_x = edge_x + self.dropout(edge_x2)
        edge_x = self.norm1_edge(edge_x)
        edge_x2 = self.linear2_edge(self.dropout(self.activation(self.linear1_edge(edge_x))))
        edge_x = edge_x + self.dropout(edge_x2)
        edge_x = self.norm2_edge(edge_x)
        
        return edge_x.transpose(0,1)
     
# module for Graph Attention Network
class NODEAttention(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, activation='relu', leakyrelu=0.1, bias=True):
        super(NODEAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == self.d_model, "d_model must be divisible by num_heads"
        
        self.in_proj_weight1 = Parameter(torch.empty(d_model, 3 * d_model))
        self.in_proj_weight2 = Parameter(torch.empty(d_model, 2 * d_model))

        if bias:
            self.in_proj_bias1 = Parameter(torch.empty(3 * d_model))  
            self.in_proj_bias2 = Parameter(torch.empty(2 * d_model))
        else:
            self.register_parameter('in_proj_bias1', None)
            self.register_parameter('in_proj_bias2', None)
            
        self._reset_parameters()
        
        self.dropout = nn.Dropout(dropout)
        self.linear1_node = Linear(d_model, d_model)
        self.linear2_node = Linear(d_model, d_model)
        self.norm1_node = nn.modules.normalization.LayerNorm(d_model)
        self.norm2_node = nn.modules.normalization.LayerNorm(d_model)
        self.ReadOut = ReadOut(d_model, dropout, num_heads, activation, leakyrelu, bias)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
            
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight1)
        nn.init.xavier_uniform_(self.in_proj_weight2)
        if self.in_proj_bias1 is not None:
            nn.init.constant_(self.in_proj_bias1, 0.)
            nn.init.constant_(self.in_proj_bias2, 0.)
    
    def _MHA_linear(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        output = inputs.matmul(weight)
        if bias is not None:
            output += bias
        return output

    def forward(self, node_x: Tensor, edge_x: Tensor, CLS:Tensor, node_mask: Tensor, CLS_mask:Tensor) -> Tensor:
        node_x = node_x.transpose(0,1) #node_len, bsz, d_model
        edge_x = edge_x.transpose(0,1) #edge_len, bsz, d_model
        bsz = node_x.size(1)
        node_len = node_x.size(0)
        edge_len = edge_x.size(0)
        scaling = float(self.head_dim) ** -0.5
        
        # attention fot node_x
        q_node, k_node, v_node = self._MHA_linear(node_x, self.in_proj_weight1, self.in_proj_bias1).chunk(3, dim=-1) 
        k_edge, v_edge = self._MHA_linear(edge_x, self.in_proj_weight2, self.in_proj_bias2).chunk(2, dim=-1) 
        
        
        q_node = q_node.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,node_len,head_dim
        k_node = torch.cat([k_node,k_edge],dim=0)
        k_node = k_node.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,node_len+edge_len,head_dim
        v_node = torch.cat([v_node,v_edge],dim=0)
        v_node = v_node.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,node_len+edge_len,head_dim

        attn_output_weights = torch.bmm(q_node, k_node.transpose(1, 2)) #bsz*num_heads, node_len, node_len+edge_len
        attn_output_weights = attn_output_weights * scaling
        attn_output_weights = attn_output_weights.reshape(bsz, self.num_heads, node_len, -1)
        attn_output_weights = attn_output_weights.masked_fill(node_mask.unsqueeze(1),float("-inf"))
        attn_output_weights = attn_output_weights.view(bsz * self.num_heads, node_len, -1) #bsz*num_heads, node_len, node_len+edge_len
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)  #bsz*num_heads, node_len, node_len+edge_len
        attn_output_weights = self.dropout(attn_output_weights)
        
        attn_output = torch.bmm(attn_output_weights, v_node)  # size = bsz*num_heads, node_len, head_dim  
        node_x2 = attn_output.transpose(0, 1).reshape(node_len, bsz, self.d_model)
        
        node_x = node_x + self.dropout(node_x2)
        node_x = self.norm1_node(node_x)
        node_x2 = self.linear2_node(self.dropout(self.activation(self.linear1_node(node_x))))
        node_x = node_x + self.dropout(node_x2)
        node_x = self.norm2_node(node_x).transpose(0,1)
        
        CLS = self.ReadOut(node_x, CLS, CLS_mask)
        return node_x, CLS

# module for Graph Attention Network
class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, activation='relu', leakyrelu=0.1):
        super(GraphAttentionLayer, self).__init__()
        
        self.node_attention = NODEAttention(d_model, dropout, num_heads, activation, leakyrelu)
        self.edge_attention = EDGEAttention(d_model, dropout, num_heads, activation, leakyrelu)

    def forward(self, node_x: Tensor, edge_x: Tensor, CLS:Tensor, node_mask: Tensor, edge_mask: Tensor, CLS_mask:Tensor) -> Tensor:
        edge_x = self.edge_attention(edge_x, edge_mask)
        node_x, CLS = self.node_attention(node_x, edge_x, CLS, node_mask, CLS_mask)
        
        return node_x, edge_x, CLS

class ReadOut(nn.Module):
    def __init__(self, d_model=128, dropout=0.1, num_heads=1, activation='relu', leakyrelu=0.1, bias=True):
        super(ReadOut, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == self.d_model, "d_model must be divisible by num_heads"
        self.in_proj_weight = Parameter(torch.empty(d_model, 2 * d_model))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(2 * d_model))
        else:
            self.register_parameter('in_proj_bias', None)
            
        self._reset_parameters()
        
        self.dropout = nn.Dropout(dropout)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)
        self.norm1 = nn.modules.normalization.LayerNorm(d_model)
        self.norm2 = nn.modules.normalization.LayerNorm(d_model)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
            
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
    
    def _MHA_linear(self, inputs: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        output = inputs.matmul(weight)
        if bias is not None:
            output += bias
        return output

    def forward(self, node_x: Tensor, CLS:Tensor, CLS_mask:Tensor) -> Tensor:
        node_x = torch.cat([CLS.reshape(-1,1,self.d_model), node_x], dim=1)
        node_x = node_x.transpose(0,1) #node_len+1, bsz, d_model
        bsz = node_x.size(1)
        node_len = node_x.size(0) # in fact, here node_len = node_len+1
        scaling = float(self.head_dim) ** -0.5

        # self-attention fot edge_x
        k_node, v_node = self._MHA_linear(node_x, self.in_proj_weight, self.in_proj_bias).chunk(2, dim=-1) 
        
        q_node =    CLS.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,1,head_dim
        k_node = k_node.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,node_len,head_dim
        v_node = v_node.reshape(-1, bsz*self.num_heads, self.head_dim).transpose(0, 1) #bsz*num_heads,node_len,head_dim

        attn_output_weights = torch.bmm(q_node, k_node.transpose(1, 2)) #bsz*num_heads, 1, node_len
        attn_output_weights = attn_output_weights * scaling
        attn_output_weights = attn_output_weights.reshape(bsz, self.num_heads, 1, node_len)
        attn_output_weights = attn_output_weights.masked_fill(CLS_mask,float("-inf"))
        attn_output_weights = attn_output_weights.view(bsz * self.num_heads, 1, node_len) #bsz*num_heads, 1, node_len
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)  #bsz*num_heads, 1, node_len
        attn_output_weights =  self.dropout(attn_output_weights)
        
        attn_output = torch.bmm(attn_output_weights, v_node)  # size = bsz*num_heads, 1, head_dim  
        CLS2 = attn_output.transpose(0, 1).reshape(bsz, self.d_model)
        
        CLS = CLS + self.dropout(CLS2)
        CLS = self.norm1(CLS)
        CLS2 = self.linear2(self.dropout(self.activation(self.linear1(CLS))))
        CLS = CLS + self.dropout(CLS2)
        CLS = self.norm2(CLS)
        
        return CLS
    
class GAT(nn.Module):
    def __init__(self, nb_node_features, nb_edge_features, d_model=128, dropout=0.1, num_heads=1, nb_layer=1, activation='relu', leakyrelu=0.1):
        super(GAT, self).__init__()
        self.d_model = d_model
        self.model_type = 'GAT'
        self.nb_layer = nb_layer
        
        self.node_attention = NODEAttention(d_model=d_model, dropout=dropout, num_heads=num_heads, activation=activation, leakyrelu=leakyrelu)
        
        # transform features to vectors
        self.CLS = Parameter(torch.empty(1, d_model))
        nn.init.xavier_uniform_(self.CLS)
        self.node_embedding = Linear(nb_node_features, d_model, bias=True)
        self.node_W1 = Linear(d_model, d_model)
        self.node_W2 = Linear(d_model, d_model)
        self.edge_embedding = Linear(nb_edge_features, d_model, bias=True)
        self.edge_W = Linear(d_model, d_model)
        self.norm_node1 = nn.modules.normalization.LayerNorm(d_model)
        self.norm_node2 = nn.modules.normalization.LayerNorm(d_model)
        self.norm_node3 = nn.modules.normalization.LayerNorm(d_model)
        self.norm1_edge = nn.modules.normalization.LayerNorm(d_model)
        self.norm2_edge = nn.modules.normalization.LayerNorm(d_model)
        
        self.layers = self._get_clones(GraphAttentionLayer(d_model=d_model, dropout=dropout, num_heads=num_heads, activation=activation, leakyrelu=leakyrelu), nb_layer)

    def _get_clones(self, module, N):
        return nn.modules.container.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    def forward(self, node_features:Tensor, edge_features:Tensor)->Tuple[Tensor,Tensor]:
        node_len = node_features.size(1)
        edge_len = edge_features.size(1)
        bsz = edge_features.size(0)

        CLS_mask = torch.cat([torch.zeros(bsz,1).to(node_features.device), node_features[:,:,0]], dim=1).unsqueeze(1).unsqueeze(1)
        CLS_mask = CLS_mask > 0
        CLS = self.CLS.expand(bsz, self.d_model)
        edges_all = edge_features[:,:,:2].cpu().numpy().astype(int).tolist()
        for i in range(len(edges_all)):
            for j in range(len(edges_all[i])):
                if edges_all[i][j] == [0,0]:
                    del edges_all[i][j:]
                    break
  
        node_x = self.node_embedding(node_features)
        node_x = self.norm_node1(node_x)# + Envs_x)
        node_x2 = self.norm_node2(self.node_W1(node_x))
        node_x3 = self.norm_node3(self.node_W2(node_x))
        
        edge_x = self.norm1_edge(self.edge_embedding(edge_features[:,:,2:])) # bsz,edge_len,d_model
        edge_x = edge_x.reshape(-1, self.d_model).repeat(1,2).reshape(bsz, 2*edge_len, self.d_model)
        
        edge_mask_all = np.zeros((bsz,2*edge_len,2*edge_len))
        node_mask_all = np.zeros((bsz,node_len,node_len+2*edge_len))
        
        for i in range(bsz):
            edges = edges_all[i]
                
            dict_edge_node = {i:[] for i in range(node_len)}
            for j in range(len(edges)):
                dict_edge_node[edges[j][0]].append(2*j+1)
                dict_edge_node[edges[j][1]].append(2*j)
                
            edge_mask = np.identity(2*edge_len)
            edge2_index2 = []
            edge2_index3 = []
            for j in range(len(edges)):
                mask_index = copy.deepcopy(dict_edge_node[edges[j][0]])
                mask_index.remove(2*j+1)
                edge_mask[2*j,mask_index] = 1
                mask_index = copy.deepcopy(dict_edge_node[edges[j][1]])
                mask_index.remove(2*j)
                edge_mask[2*j+1,mask_index] = 1
    
                edge2_index2 += [edges[j][0], edges[j][1]]
                edge2_index3 += [edges[j][1], edges[j][0]]

            for j in range(node_len):
                node_mask_all[i,j,[node_len+p for p in dict_edge_node[j]]] = 1
                node_mask_all[i,j,j] = 1
            edge_mask_all[i,:,:] = edge_mask
            edge2_index2 = torch.LongTensor(edge2_index2).to(node_features.device)  
            edge2_index3 = torch.LongTensor(edge2_index3).to(node_features.device)
            edge_x[i,:2*len(edges),:] += torch.index_select(node_x2[i,:,:], 0, edge2_index2)
            edge_x[i,:2*len(edges),:] += torch.index_select(node_x3[i,:,:], 0, edge2_index3)
        
        edge_x = self.norm2_edge(self.edge_W(edge_x))
        edge_mask_all = ~torch.LongTensor(edge_mask_all).to(torch.bool).to(node_features.device)
        node_mask_all = ~torch.LongTensor(node_mask_all).to(torch.bool).to(node_features.device)
        
        node_x, CLS = self.node_attention(node_x, edge_x, CLS, node_mask_all, CLS_mask)
        all_CLS = CLS
        for mod in self.layers:
            node_x, edge_x, CLS = mod(node_x, edge_x, CLS, node_mask_all, edge_mask_all, CLS_mask)
            all_CLS = torch.cat([all_CLS, CLS], dim=1)
        return node_x, all_CLS

class MP(nn.Module):
    def __init__(self, nb_output, input_dim=128, d_model=128, activation='relu', leakyrelu=0.1):
        super(MP, self).__init__()
        self.model_type = 'MP'
        self.linear1 = Linear(input_dim, nb_output)
        if activation == "relu":
            self.activation = F.relu
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(leakyrelu)
            
    def forward(self, inputs:Tensor):
        output = self.linear1(inputs)
        return output
    
class MolecularProperties(nn.Module):
    def __init__(self, nb_node_features, nb_edge_features, nb_MP, d_model=128, dropout=0.1, num_heads=1, nb_layer=1, activation='relu', leakyrelu=0.1):
        super(MolecularProperties, self).__init__()
        self.model_type = 'MolecularProperties'
        self.GAT = GAT(nb_node_features, nb_edge_features, d_model=d_model, dropout=dropout, num_heads=num_heads, nb_layer=nb_layer, activation=activation, leakyrelu=leakyrelu)
        self.MP = MP(nb_MP, (nb_layer+1)*d_model, d_model)
            
    def forward(self, inputs):
        _, CLS = self.GAT(inputs['nodes_features'], inputs['edges_features'])
        outputs = self.MP(CLS)
        return outputs

def D_GAT(dataset, mol, config_file_path):
    nb_node_features, nb_edge_features, nb_MP = mol[0].mdoel_needed_info()  
    mol_config = json.load(open(config_file_path,'r'))
    d_model = mol_config["d_model"]
    num_heads = mol_config["num_heads"]
    activation = mol_config["activation"]
    PreTraining_model_path = mol_config["PreTraining_model_path"]
    if 'dropout' + dataset in mol_config:
        dropout = mol_config["dropout" + dataset]
    else:
        dropout = mol_config["dropout"]
    if 'nb_layer' + dataset in mol_config:
        nb_layer = mol_config["nb_layer" + dataset]
    else:
        nb_layer = mol_config["nb_layer"]
    
    model = MolecularProperties(nb_node_features, nb_edge_features-2, nb_MP, d_model=d_model, dropout=dropout, num_heads=num_heads, nb_layer=nb_layer,activation=activation,leakyrelu=0.1)
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
    
    best_score = [0, 1e5, 0, 0, 0]
    return model, best_score