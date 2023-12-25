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

def get_mol_max_length(all_smiles_y):
    max_len = []
    for i in all_smiles_y:
        try:
            nb_atom = len(Chem.MolFromSmiles(i[0]).GetAtoms())
        except:
            nb_atom = 0
        max_len.append(nb_atom)
            
    return max(max_len) 

def load_data(dataset, task_name=None, bool_scale=True, print_info=True):
    if task_name == None:
        if dataset == 'PD':
            task_name = ['feat_SolubilityInDMF(mol/L)', 'feat_TopDistance(Å)','feat_BottomDistance(Å)', 'feat_LeftRightDistanceMax(Å)','feat_LeftRightDistanceMin(Å)', 'feat_TopBottomDistanceRatio','feat_LeftRightDistanceMaxMinRatio','feat_ElectronicEnergyNeutralHighEnergy(Hartree)','feat_ElectronicEnergyNeutralLowEnergy(Hartree)','feat_ElectronicEnergyDifferenceNeutral(kJ/mol)','feat_DipoleMomentNeutralHighEnergy(Debye)','feat_DipoleMomentNeutralLowEnergy(Debye)','feat_HOMONeutralHighEnergy(Hartree)','feat_HOMONeutralLowEnergy(Hartree)','feat_LUMONeutralHighEnergy(Hartree)','feat_LUMONeutralLowEnergy(Hartree)','feat_HOMO-LUMOGapNeutralHighEnergy(Hartree)','feat_HOMO-LUMOGapNeutralLowEnergy(Hartree)','feat_N1ESPChargeNeutralHighEnergy','feat_N1ESPChargeNeutralLowEnergy','feat_H1ESPChargeNeutralHighEnergy','feat_H1ESPChargeNeutralLowEnergy','feat_C2ESPChargeNeutralHighEnergy','feat_C2ESPChargeNeutralLowEnergy','feat_N3ESPChargeNeutralHighEnergy','feat_N3ESPChargeNeutralLowEnergy','feat_C4ESPChargeNeutralHighEnergy','feat_C4ESPChargeNeutralLowEnergy','feat_C5ESPChargeNeutralHighEnergy','feat_C5ESPChargeNeutralLowEnergy','feat_N1H1ESPChargeNeutralHighEnergy','feat_N1H1ESPChargeNeutralLowEnergy','feat_ElectronicEnergyCharged(Hartree)','feat_ElectronicEnergyDifferenceNeutralHighEnergyCharged(kJ/mol)','feat_ElectronicEnergyDifferenceNeutralLowEnergyCharged(kJ/mol)','feat_DipoleMomentCharged(Debye)', 'feat_HOMOCharged(Hartree)','feat_LUMOCharged(Hartree)', 'feat_HOMO-LUMOGapCharged(Hartree)','feat_N1N3ESPChargeChargedMax', 'feat_C2ESPChargeCharged','feat_N1N3ESPChargeChargedMin', 'feat_C4ESPChargeChargedN1N3Max','feat_C5ESPChargeChargedN1N3Max']
            PD_name = ['rxn_MetalConcentration(mol/L)','rxn_LinkerConcentration(mol/L)', 'rxn_TotalConcentration(mol/L)','rxn_TotalConcentrationParticle(mol/L)','rxn_ConcentrationLinkerOverSolubility', 'rxn_LinkerMetalRatio','rxn_Log10LinkerMetalRatio', 'rxn_Temperature(C)','rxn_DielectricConstant', 'rxn_Time(h)']
            smi_name = 'LinkerCanonicalSmiles'
        else:
            raise RuntimeError('Unrecognized Database!!!')
            
    print('Read and process the collected data...')
    file = pd.read_csv('./data/' + dataset + '.csv', header=0)

    if print_info:
        print('----------------------------------------')
        print('Dataset: ', dataset)
        print('Example: ')
        print(file.iloc[0])
        print('Number of molecules:', file.shape[0])
        
    def record_and_calculate(smi, PD, target):
        PD = np.array(PD)
        theta = []
        for i in range(PD.shape[1]):
            theta.append([np.mean(PD[:,i]), np.std(PD[:,i])])
        return [smi, target, theta]
        
    all_smiles_y = []
    file = file.where(pd.notnull(file), -1)
    
    # decide the scale
    scale_fea = np.ones(len(task_name))
    scale_PD= np.ones(len(PD_name))
    if bool_scale:
        file_max = file.max()
        for i,j in enumerate(task_name):
            scale_fea[i] = file_max[j]
        for i,j in enumerate(PD_name):
            scale_PD[i] = file_max[j]

    target = []
    smi = ''
    for i in range(file.shape[0]):
        if file[smi_name][i] == smi:
            PD.append([file[j][i]/scale_PD[p] for p,j in enumerate(PD_name)])
            if i == file.shape[0] - 1:
                all_smiles_y.append([smi, target, PD])
        else:
            if target:
                all_smiles_y.append([smi, target, PD])
            smi = file[smi_name][i]
            PD = []
            target = [file[j][i]/scale_fea[p] for p,j in enumerate(task_name)]
            PD.append([file[j][i]/scale_PD[p] for p,j in enumerate(PD_name)])       
    return all_smiles_y, scale_fea, scale_PD

class Molecule():
    def __init__(self, smiles_y, dataset, bool_random=True, max_len=100, max_ring=15, print_info=False):
        self.max_len = max_len # maximum number of atoms in a molecule
        self.max_ring = max_ring
        self.smiles, self.targets, self.PD = smiles_y
        self.nb_MP = len(self.targets)
        self.nb_PD = len(self.PD[0])
        self.exist = True
        self.dataset = dataset # name of dataset
        self.process_mol_with_RDKit()
       
    def mdoel_needed_info(self):
        return self.nb_node_features, self.nb_edge_features, self.nb_MP, self.nb_PD
    
    def process_mol_with_RDKit(self, bool_random=True):
        mol = Chem.MolFromSmiles(self.smiles, sanitize=True)
        if mol is None:
            self.exist = False
            print('Bad smiles to generate mol!!!', self.smiles)
            return None
        nodes_features = []
        edges = []
        edges_features = []
        
        nb_atoms = len(mol.GetAtoms())
                    
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
        if len(atoms) <= node_len:
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

                edge_features = [0] * 17                        # First two places are indices of connected nodes, third place is used as [MASKED EDGE]
                edge_features[0] = edge[0]
                edge_features[1] = edge[1]
                bond_type = int(bond.GetBondType()) if (int(bond.GetBondType())<=3 or int(bond.GetBondType())==12) else 0
                if bond_type == 12:
                    bond_type = 4
                edge_features[bond_type+2] = 1                  # bond type [OTHERS,SINGLE,DOUBLE,TRIPLE,AROMATIC]
                edge_features[7] = int(bond.GetIsConjugated())  # whether conjugation
                edge_features[8] = int(bond.IsInRing())         # whether in the ring
                edge_features[int(bond.GetStereo())+9] = 1      # stereo type [STEREONONE,STEREOANY,STEREOZ,STEREOE,STEREOCIS,STEREOTRANS]

                edges_features.append(edge_features)
            edge_features = [0] * 17
            edge_features[-1] = 1
            if edge_len < len(bonds):
                print('Too Much Bonds!!! ', self.smiles)
                self.exist = False
                return None
            for _ in range(edge_len - len(bonds)):
                edges_features.append(edge_features)
        else:
            self.exist = False
#             print('Bad molecule to generate features!!!', self.smiles)
            return None
#             raise RuntimeError('Bad molecule to generate features!!!')

        self.mol = mol
        self.nb_atoms = len(atoms)
        self.nodes_features = np.array(nodes_features)
        self.edges_features = np.array(edges_features)
        self.nb_node_features = len(node_features)
        self.nb_edge_features = len(edge_features)
        self.edges = edges
        
    def get_inputs_features(self):
        return self.nodes_features, self.edges_features
    
    def get_edges(self):
        return self.edges
    
def para_process_mol(i, all_smiles_y, dataset, max_len, nb_time_processing=10):
    mol = []
    for j in range(i*nb_time_processing, min((i+1)*nb_time_processing,len(all_smiles_y))):
        one_mol = Molecule(all_smiles_y[j], dataset, False, max_len)
        if one_mol.exist:
            mol.append(one_mol)
    return i, mol

def Read_mol_data(dataset, bool_scale=False, task_name=None):
    max_len = {}
    nb_cpu = 1
    nb_time_processing = 20
    all_smiles_y, scale_fea, scale_PD = load_data(dataset, task_name, bool_scale)
    
    if dataset in max_len:
        train_max_len, val_max_len, test_max_len = max_len[dataset]
    else:
        max_len = get_mol_max_length(all_smiles_y)
        train_max_len = val_max_len = test_max_len = max_len
        
    all_targets = []
    for i in all_smiles_y:
        all_targets.append(i[1])
    mean = torch.Tensor(np.mean(all_targets, axis=0))
    std = torch.Tensor(np.std(all_targets, axis=0))

    num_train = 7 #int(len(all_smiles_y) * 0.8)
    num_val= 1 #int(len(all_smiles_y) * 0.1)
    train_smiles_y = all_smiles_y[:num_train]
    val_smiles_y = all_smiles_y[num_train:num_train + num_val]
    test_smiles_y = all_smiles_y[num_train + num_val:]
    random.shuffle(train_smiles_y)
    
    mol_train = []
    mol_val = []
    mol_test = []
    
    if len(train_smiles_y) > 2e5:
        nb = 100
    elif len(train_smiles_y) > 1e4:
        nb = 10
    else:
        nb = 1
    nb_part = int(np.ceil(len(train_smiles_y)/nb))
    for j in range(nb):
        part_smiles_y = train_smiles_y[j*nb_part:(j+1)*nb_part]
        with Pool(processes=nb_cpu) as pool:
            for results in pool.imap(functools.partial(para_process_mol, all_smiles_y=part_smiles_y, dataset=dataset, max_len=train_max_len, nb_time_processing=nb_time_processing),range(int(np.ceil(len(part_smiles_y)/nb_time_processing)))):
                i, mol_part = results
                mol_train += mol_part
                if nb == 1 and (i+1) % (nb_cpu*50) == 0:
                    print(i+1,'/',int(np.ceil(len(part_smiles_y)/nb_time_processing)),' finished')
        print(j+1,'/',nb,' finished!')
    random.shuffle(mol_train)  
    print('Training dataset finished')

    if len(val_smiles_y) > 2e4:
        nb = 10
    else:
        nb = 1
    nb_part = int(np.ceil(len(val_smiles_y)/nb))
    for j in range(nb):
        part_smiles_y = val_smiles_y[j*nb_part:(j+1)*nb_part]
        with Pool(processes=nb_cpu) as pool:
            for results in pool.imap(functools.partial(para_process_mol, all_smiles_y=part_smiles_y, dataset=dataset, max_len=val_max_len, nb_time_processing=nb_time_processing),range(int(np.ceil(len(part_smiles_y)/nb_time_processing)))):
                i, mol_part = results
                mol_val += mol_part
                if nb == 1 and (i+1) % (nb_cpu*25) == 0:
                    print(i+1,'/',int(np.ceil(len(part_smiles_y)/nb_time_processing)),' finished')
        if nb > 1:
            print(j+1,'/',nb,' finished!')
    print('Val dataset finished')

    nb_part = int(np.ceil(len(test_smiles_y)/nb))
    for j in range(nb):
        part_smiles_y = test_smiles_y[j*nb_part:(j+1)*nb_part]
        with Pool(processes=nb_cpu) as pool:
            for results in pool.imap(functools.partial(para_process_mol, all_smiles_y=part_smiles_y, dataset=dataset, max_len=test_max_len, nb_time_processing=nb_time_processing),range(int(np.ceil(len(part_smiles_y)/nb_time_processing)))):
                i, mol_part = results
                mol_test += mol_part
                if nb == 1 and (i+1) % (nb_cpu*25) == 0:
                    print(i+1,'/',int(np.ceil(len(part_smiles_y)/nb_time_processing)),' finished')
        if nb > 1:
            print(j+1,'/',nb,' finished!')
    print('Test dataset finished')
    return mol_train, mol_val, mol_test, mean, std, scale_fea, scale_PD

def PreProcess(mol):
    targets = []
    PD = []
    inputs = {'nodes_features':[], 'edges_features':[]}
    for i in range(len(mol)):
        # for targets
        targets.append(mol[i].targets)
        PD.append(mol[i].PD)
        # for inputs
        nodes_features, edges_features = mol[i].get_inputs_features()
        node_len = nodes_features.shape[0]
        edge_len = edges_features.shape[0]
        inputs['nodes_features'] += [nodes_features.tolist()]
        inputs['edges_features'] += [edges_features.tolist()]
        
    targets = torch.Tensor(targets)
    PD = torch.Tensor(PD)
    for name in inputs:
        if 'features' in name:
            inputs[name] = torch.Tensor(inputs[name])
    
    return inputs, targets, PD

def Generate_dataloader(dataset, mol_train, mol_val, mol_test):
    train_dataloader = []
    val_dataloader = []
    test_dataloader = []
    
    bsz = 1
    num_train_dataloader = int(np.ceil(len(mol_train)/bsz))
    num_val_dataloader = int(np.ceil(len(mol_val)/bsz))
    num_test_dataloader = int(np.ceil(len(mol_test)/bsz))

    for i in range(num_train_dataloader):
        train_dataloader.append(PreProcess(mol_train[i * bsz:(i+1)*bsz]))

    for i in range(num_val_dataloader):
        val_dataloader.append(PreProcess(mol_val[i * bsz: (i+1) * bsz]))

    for i in range(num_test_dataloader):
        test_dataloader.append(PreProcess(mol_test[i * bsz: (i+1) * bsz]))
    return train_dataloader, val_dataloader,test_dataloader