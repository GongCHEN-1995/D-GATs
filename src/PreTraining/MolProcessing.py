import numpy as np
import pandas as pd
import functools
import subprocess
import random
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') #hiding the warning messages

import torch

class Molecule():
    def __init__(self, smiles_y, bool_random=True, rate=0, max_len=100, max_ring=15, print_info=False):
        self.max_len = max_len # maximum number of atoms in a molecule
        self.max_ring = max_ring
        self.smiles, self.targets = smiles_y
        if self.targets == []:
            self.targets = (np.ones(3)*(1e5)).tolist()
        self.nb_MP = len(self.targets)
        self.exist = True
        self.process_mol_with_RDKit()
        if self.exist and rate > 0:
            self.mask_nodes(rate)
            
    def mdoel_needed_info(self):
        return self.nb_node_features, self.nb_edge_features, self.nb_MP, self.num_ele
    
    def process_mol_with_RDKit(self, bool_random=True):
        mol = Chem.MolFromSmiles(self.smiles, sanitize=True)
        if mol is None:
            self.exist = False
            return None
        nodes_features = []
        edges = []
        edges_features = []

        nb_atoms = len(mol.GetAtoms())
        if nb_atoms <= self.max_len:
            special_max_len = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
            for i in special_max_len:
                if nb_atoms <= i:
                    self.max_len = min(self.max_len, i)
                    break
                
        node_len = self.max_len
        edge_len = self.max_len + self.max_ring
        all_ele = ['PAD', 'MASK', 'UNK', 'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn',\
                   'Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I',\
                   'Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl',\
                   'Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']
        ele2index = {j : i for i,j in enumerate(all_ele)}
        num_ele = len(all_ele)
                
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        mapping = np.arange(len(mol.GetAtoms()))
        if bool_random:
            np.random.shuffle(mapping)
        reverse_mapping = np.zeros(len(mapping)).astype(int)
        for i in range(len(mapping)):
            reverse_mapping[mapping[i]] = i
        mapping = mapping.tolist()
        reverse_mapping = reverse_mapping.tolist()
        if len(atoms) <= node_len and len(atoms) >= 10:
            for i in range(len(atoms)): 
                atom = atoms[reverse_mapping[i]]
                node_features = [0] * (num_ele+28)
                node_features[ele2index[atom.GetSymbol() if atom.GetSymbol() in all_ele else 'UNK']] = 1 # atomic numebr [all_ele]
                node_features[num_ele+(atom.GetDegree() if atom.GetDegree()<=5 else 6)] = 1              # degree of atom [0~5, 6=other]
                node_features[int(atom.GetHybridization())+num_ele+7] = 1   # hybridization type [unspecified,s,sp,sp2,sp3,sp3d,sp3d2,other]
                node_features[int(atom.GetChiralTag())+num_ele+15] = 1      # chirality [CHI_UNSPECIFIED,CHI_TETRAHEDRAL_CW,CHI_TETRAHEDRAL_CCW,CHI_OTHER]
                num_H = atom.GetTotalNumHs() if atom.GetTotalNumHs() < 4 else 4
                node_features[num_H+num_ele+19] = 1                         # number of H atoms [0,1,2,3,4]
                node_features[num_ele+24] = int(atom.IsInRing())             # whether in ring
                node_features[num_ele+25] = int(atom.GetIsAromatic())        # whether aromatic  
                node_features[num_ele+26] = atom.GetFormalCharge()          # formal charge
                node_features[num_ele+27] = atom.GetNumRadicalElectrons()   # radical electrons
                nodes_features.append(node_features)
            node_features = [0] * (num_ele+28)
            node_features[ele2index['PAD']] = 1
            for _ in range(node_len - len(atoms)):
                nodes_features.append(node_features)

            for bond in bonds:
                edge = [mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]]
                edges.append(edge)
                edge_features = [0] * 17                       # First two places are indices of connected nodes, third place is used as [MASKED EDGE]
                edge_features[0] = edge[0]
                edge_features[1] = edge[1]
                bond_type = int(bond.GetBondType()) if (int(bond.GetBondType())<=3 or int(bond.GetBondType())==12) else 0
                if bond_type == 12:
                    bond_type = 4
                edge_features[bond_type+2] = 1                  # bond type [OTHERS,SINGLE,DOUBLE,TRIPLE,AROMATIC]
                edge_features[7] = int(bond.GetIsConjugated())  # whether conjugation
                edge_features[8] = int(bond.IsInRing())         # whether in the ring
                edge_features[int(bond.GetStereo())+9] = 1      # stereo type [STEREONONE,STEREOANY,STEREOZ,STEREOE,STEREOCIS,STEREOTRANS]
#                 edge_features[-2] is for 'PAD' and edge_features[-1] is for 'MASK'
                edges_features.append(edge_features)
            edge_features = [0] * 17
            edge_features[-2] = 1 # PAD
            if edge_len < len(bonds):
                print('Too Much Bonds!!! ', self.smiles)
                self.exist = False
                return None
            for _ in range(edge_len - len(bonds)):
                edges_features.append(edge_features)
        else:
            self.exist = False
            return None

        self.num_ele = num_ele
        self.mol = mol
        self.nb_atoms = len(atoms)
        self.nodes_features = np.array(nodes_features)
        self.edges_features = np.array(edges_features)
        self.nb_node_features = len(node_features)
        self.nb_edge_features = len(edge_features)
        self.edges = edges
        
    def get_inputs_features(self):
        return self.nodes_features, self.edges_features
    
    def get_mask_data(self):
        return self.mask_node, self.node_targets
        
    def get_edges(self):
        return self.edges
    
    def find_CLS(self, x):
        index = []
        for i,j in enumerate(x):
            if j == 1:
                index.append(i)
        assert len(index) == 1, "Multi-label for Classification or no result!!!"
        return index[0]
                
    def find_mask_targets(self, i):
        target = []
        num_ele = self.num_ele
        node_features = self.nodes_features[i]
        target.append(node_features[num_ele+24])                           # whether in ring
        target.append(node_features[num_ele+25])                           # whether aromatic  
        target.append(self.find_CLS(node_features[:num_ele]))              # element
        target.append(self.find_CLS(node_features[num_ele:num_ele+7]))     # degree of atom [0~5, 6=other]
        target.append(self.find_CLS(node_features[num_ele+7:num_ele+15]))  # hybridization type [unspecified,s,sp,sp2,sp3,sp3d,sp3d2,other]
        target.append(self.find_CLS(node_features[num_ele+15:num_ele+19])) # chirality [CHI_UNSPECIFIED,CHI_TETRAHEDRAL_CW,CHI_TETRAHEDRAL_CCW,CHI_OTHER]
        target.append(self.find_CLS(node_features[num_ele+19:num_ele+24])) # number of H atoms [0,1,2,3,4]
        target.append(node_features[num_ele+26])                           # formal charge
        target.append(node_features[num_ele+27])                           # radical electrons
        return target
    
    def generate_random_node(self):
        num_ele = self.num_ele
        output = np.zeros(self.nodes_features.shape[1])
        output[random.randint(2, num_ele-1)] = 1            # atomic numebr [all_ele]
        output[num_ele+random.randint(0, 6)] = 1            # degree of atom [0~5, 6=other]
        output[random.randint(0, 7)+num_ele+7] = 1          # hybridization type [unspecified,s,sp,sp2,sp3,sp3d,sp3d2,other]
        output[random.randint(0, 4)+num_ele+15] = 1         # chirality [CHI_UNSPECIFIED,CHI_TETRAHEDRAL_CW,CHI_TETRAHEDRAL_CCW,CHI_OTHER]
        output[random.randint(0, 4)+num_ele+19] = 1         # number of H atoms [0,1,2,3,4]
        output[num_ele+24] = random.randint(0, 1)           # whether in ring
        output[num_ele+15] = random.randint(0, 1)           # whether aromatic  
        output[num_ele+26] = random.randint(-1,1)           # formal charge
        output[num_ele+27] = 0                              # radical electrons
        return output
    
    def generate_random_edge(self):
        output = np.zeros(self.edges_features.shape[1]-2)
        output[random.randint(0, 4)] = 1     # bond type [OTHERS,SINGLE,DOUBLE,TRIPLE,AROMATIC]
        output[5] = random.randint(0, 1)     # whether conjugation
        output[6] = random.randint(0, 1)     # whether in ring
        output[7+random.randint(0, 5)] = 1   # stereo type [STEREONONE,STEREOANY,STEREOZ,STEREOE,STEREOCIS,STEREOTRANS]

        return np.array(output)
        
    def mask_nodes(self, rate):
        edges = self.edges
        mask_node = np.arange(self.nb_atoms)
        np.random.shuffle(mask_node)
        mask_node = mask_node[:int(np.ceil(rate*self.nb_atoms))]
        node_targets = []
        for i in mask_node:
            node_targets.append(self.find_mask_targets(i))

        mask_node_feature = [0] * self.nodes_features.shape[1]
        mask_node_feature[1] = 1
        for i in mask_node:
            if random.random() < 0.8: # 80% possibility to mask
                self.nodes_features[i,:] = mask_node_feature
            else: # 80% possibility to use wrong inputs
                self.nodes_features[i,:] = self.generate_random_node()
            
        mask_edge = []
        for i in range(len(edges)):
            if edges[i][0] in mask_node or edges[i][1] in mask_node:
                mask_edge.append(i)
        mask_edge_feature = [0] * (self.edges_features.shape[1] - 2)
        mask_edge_feature[-1] = 1
        for i in mask_node:
            if random.random() < 0.8: # 80% possibility to mask
                self.edges_features[i,2:] = mask_edge_feature
            else: # 80% possibility to use wrong inputs
                self.edges_features[i,2:] = self.generate_random_edge()
        self.mask_node = mask_node
        self.mask_edge = mask_edge
        self.node_targets = np.array(node_targets)
        
def para_process_mol(i, all_smiles_y, rate, max_len, nb_time_processing=10):
    mol = []
    for j in range(i*nb_time_processing, min((i+1)*nb_time_processing,len(all_smiles_y))):
        one_mol = Molecule(all_smiles_y[j], True, rate, max_len)
        if one_mol.exist:
            mol.append(one_mol)
    return i, mol

def load_data(dataset):
    print('Read and process the collected data...')
    if dataset == '250k':
        file = pd.read_csv('./data/' + dataset + '.csv', header=0)
    else:
        file = pd.read_csv('./data/' + dataset + 'Scaffold.csv', header=0)
        
    task_name = ['logP', 'qed', 'SAS'] if dataset == '250k' else []
    if 'smiles' in file:
        smi_name = 'smiles'
    else:
        smi_name = 'mol'
        
    print('----------------------------------------')
    print('Dataset: ', dataset)
    print('example: ', file.iloc[0],'...')
    print('number of molecules:', file.shape[0])
        
    all_smiles_y = []
    file = file.where(pd.notnull(file), -1)
    for i in range(file.shape[0]):
        targets = []
        for j in task_name:
            targets.append(file[j][i])
        all_smiles_y.append([file[smi_name][i],targets])
    return all_smiles_y

def Read_mol_data():
    nb_cpu = 10
    rate = 0.2
    nb_time_processing = 20

    all_smiles_y = []
    all_smiles_y += load_data('250k')
    all_smiles_y += load_data('MUV')
    all_smiles_y += load_data('HIV')
    all_smiles_y += load_data('Tox21')
    all_smiles_y += load_data('Lipo')
    all_smiles_y += load_data('ToxCast')
    all_smiles_y += load_data('ESOL')
    all_smiles_y += load_data('FreeSolv')
    all_smiles_y += load_data('SIDER')
    all_smiles_y += load_data('BBBP')
    all_smiles_y += load_data('BACE')
    all_smiles_y += load_data('ClinTox')
    all_smiles_y += load_data('PCBA')[-40000:]
    
    max_len = 60
    mol = []
    nb = 100
    nb_part = int(np.ceil(len(all_smiles_y)/nb))
    for j in range(nb):
        part_smiles_y = all_smiles_y[j*nb_part:(j+1)*nb_part]
        with Pool(processes=nb_cpu) as pool:
            for results in pool.imap(functools.partial(para_process_mol, all_smiles_y=part_smiles_y, rate=rate, max_len=max_len, nb_time_processing=nb_time_processing),range(int(np.ceil(len(part_smiles_y)/nb_time_processing)))):
                i, mol_part = results
                mol += mol_part
        print(j+1, '/', nb, 'finished!')
    random.shuffle(mol)
    train_index = int(np.ceil(len(mol) * 0.8))
    val_index = int(np.ceil(len(mol) * 0.9))
    mol_train = mol[:train_index]
    mol_val = mol[train_index:val_index]
    mol_test = mol[val_index:]
    return mol_train, mol_val, mol_test

def PreProcess(mol):
    nb_node_features, nb_edge_features, nb_MP, nb_ele = mol[0].mdoel_needed_info() 
    targets = {'MP':[], 'ring':[], 'aromatic':[], 'element':[], 'degree':[], 'hybridization':[], 'chirality':[], 'H':[], 'formal charge':[], 'radical electrons':[]}
    inputs = {'nodes_features':[], 'edges_features':[], 'node_index':[]}
    for i in range(len(mol)):
        targets['MP'].append(mol[i].targets)
        # for inputs
        nodes_features, edges_features = mol[i].get_inputs_features()
        node_len = nodes_features.shape[0]
        inputs['nodes_features'] += [nodes_features.tolist()]
        inputs['edges_features'] += [edges_features.tolist()]

        mask_node, node_targets = mol[i].get_mask_data()
        assert node_targets.shape[1] == 9
        inputs['node_index'] += (mask_node + i * node_len).tolist()
        targets['ring'] += node_targets[:,0].tolist()
        targets['aromatic'] += node_targets[:,1].tolist()
        targets['element'] += node_targets[:,2].tolist()
        targets['degree'] += node_targets[:,3].tolist()
        targets['hybridization'] += node_targets[:,4].tolist()
        targets['chirality'] += node_targets[:,5].tolist()
        targets['H'] += node_targets[:,6].tolist()
        targets['formal charge'] += node_targets[:,7].tolist()
        targets['radical electrons'] += node_targets[:,8].tolist()
    for name in inputs:
        if 'features' in name:
            inputs[name] = torch.Tensor(inputs[name])
        else:
            inputs[name] = torch.LongTensor(inputs[name])
    for name in targets:
        if name in ['MP']:
            targets[name] = torch.Tensor(targets[name]).reshape(-1,nb_MP)
        elif name in ['ring', 'aromatic', 'formal charge', 'radical electrons']:
            targets[name] = torch.Tensor(targets[name]).reshape(-1,1)
        else:
            targets[name] = torch.LongTensor(targets[name]).reshape(-1)
    return inputs, targets

def Generate_dataloader(mol_train, mol_val, mol_test):
    train_dataloader = []
    val_dataloader = []
    test_dataloader = []
    special_max_len = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    for i in special_max_len:
        new_mol_train = []
        for j in mol_train:
            if j.max_len == i:
                new_mol_train.append(j)
        new_mol_val = []
        for j in mol_val:
            if j.max_len == i:
                new_mol_val.append(j)
        new_mol_test = []
        for j in mol_test:
            if j.max_len == i:
                new_mol_test.append(j)

        if i <= 15:
            bsz = 1024
        elif i <= 30:
            bsz = 512
        elif i <= 45:
            bsz = 350
        elif i <= 60:
            bsz = 256
        else:
            bsz = 32
        num_train_dataloader = int(np.ceil(len(new_mol_train)/bsz))
        num_val_dataloader = int(np.ceil(len(new_mol_val)/bsz))
        num_test_dataloader = int(np.ceil(len(new_mol_test)/bsz))

        for j in range(num_train_dataloader):
            train_dataloader.append(PreProcess(new_mol_train[j * bsz:(j+1)*bsz]))
        for j in range(num_val_dataloader):
            val_dataloader.append(PreProcess(new_mol_val[j * bsz:(j+1)*bsz]))
        for j in range(num_test_dataloader):
            test_dataloader.append(PreProcess(new_mol_test[j * bsz:(j+1)*bsz]))
     
    return train_dataloader, val_dataloader, test_dataloader