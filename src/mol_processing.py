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

def get_mol_max_length(dataset):
    all_smiles_y = load_data(dataset, print_info=False)
    num_train = int(len(all_smiles_y) * 0.8)
    num_val = int(len(all_smiles_y) * 0.1)
    train_smiles_y = all_smiles_y[:num_train]
    val_smiles_y = all_smiles_y[num_train:num_train+num_val]
    test_smiles_y = all_smiles_y[num_train+num_val:]

    aa = []
    bb = []
    cc = []
    for i in train_smiles_y:
        try:
            nb_atom = len(Chem.MolFromSmiles(i[0]).GetAtoms())
        except:
            nb_atom = 0
        aa.append(nb_atom)
    for i in val_smiles_y:
        try:
            nb_atom = len(Chem.MolFromSmiles(i[0]).GetAtoms())
        except:
            nb_atom = 0
        bb.append(nb_atom)
    for i in test_smiles_y:
        try:
            nb_atom = len(Chem.MolFromSmiles(i[0]).GetAtoms())
        except:
            nb_atom = 0
        cc.append(nb_atom)

    dd = np.zeros(10) # Number of mol with different number of atoms [0, 0-25, 25-50, 50-75, 75-100, 100-150, 150-200, 200-250, >250]
    for i in aa:
        if i == 0:
            dd[0] += 1
        elif i <= 25:
            dd[1] += 1
        elif i <= 50:
            dd[2] += 1
        elif i <= 75:
            dd[3] += 1
        elif i <= 100:
            dd[4] += 1
        elif i <= 150:
            dd[5] += 1
        elif i <= 200:
            dd[6] += 1
        elif i <= 250:
            dd[7] += 1
        else:
            dd[8] += 1
            
    return max(max(aa), max(bb), max(cc)) 

def load_data(dataset, task_name=None, print_info=True):
    if task_name == None:
        if dataset == 'BACE':
            task_name = ['Class', 'pIC50']
        elif dataset == 'BBBP':
            task_name = ['p_np']
        elif dataset == 'SIDER':
            task_name = ['Hepatobiliary disorders','Metabolism and nutrition disorders','Product issues','Eye disorders','Investigations','Musculoskeletal and connective tissue disorders','Gastrointestinal disorders','Social circumstances','Immune system disorders','Reproductive system and breast disorders','Neoplasms benign, malignant and unspecified (incl cysts and polyps)','General disorders and administration site conditions','Endocrine disorders','Surgical and medical procedures','Vascular disorders','Blood and lymphatic system disorders','Skin and subcutaneous tissue disorders','Congenital, familial and genetic disorders','Infections and infestations','Respiratory, thoracic and mediastinal disorders','Psychiatric disorders','Renal and urinary disorders','Pregnancy, puerperium and perinatal conditions','Ear and labyrinth disorders','Cardiac disorders','Nervous system disorders','Injury, poisoning and procedural complications']
        elif dataset == 'ToxCast':
            task_name = ['ACEA_T47D_80hr_Negative','ACEA_T47D_80hr_Positive','APR_HepG2_CellCycleArrest_24h_dn','APR_HepG2_CellCycleArrest_24h_up','APR_HepG2_CellCycleArrest_72h_dn','APR_HepG2_CellLoss_24h_dn','APR_HepG2_CellLoss_72h_dn','APR_HepG2_MicrotubuleCSK_24h_dn','APR_HepG2_MicrotubuleCSK_24h_up','APR_HepG2_MicrotubuleCSK_72h_dn','APR_HepG2_MicrotubuleCSK_72h_up','APR_HepG2_MitoMass_24h_dn','APR_HepG2_MitoMass_24h_up','APR_HepG2_MitoMass_72h_dn','APR_HepG2_MitoMass_72h_up','APR_HepG2_MitoMembPot_1h_dn','APR_HepG2_MitoMembPot_24h_dn','APR_HepG2_MitoMembPot_72h_dn','APR_HepG2_MitoticArrest_24h_up','APR_HepG2_MitoticArrest_72h_up','APR_HepG2_NuclearSize_24h_dn','APR_HepG2_NuclearSize_72h_dn','APR_HepG2_NuclearSize_72h_up','APR_HepG2_OxidativeStress_24h_up','APR_HepG2_OxidativeStress_72h_up','APR_HepG2_StressKinase_1h_up','APR_HepG2_StressKinase_24h_up','APR_HepG2_StressKinase_72h_up','APR_HepG2_p53Act_24h_up','APR_HepG2_p53Act_72h_up','APR_Hepat_Apoptosis_24hr_up','APR_Hepat_Apoptosis_48hr_up','APR_Hepat_CellLoss_24hr_dn','APR_Hepat_CellLoss_48hr_dn','APR_Hepat_DNADamage_24hr_up','APR_Hepat_DNADamage_48hr_up','APR_Hepat_DNATexture_24hr_up','APR_Hepat_DNATexture_48hr_up','APR_Hepat_MitoFxnI_1hr_dn','APR_Hepat_MitoFxnI_24hr_dn','APR_Hepat_MitoFxnI_48hr_dn','APR_Hepat_NuclearSize_24hr_dn','APR_Hepat_NuclearSize_48hr_dn','APR_Hepat_Steatosis_24hr_up','APR_Hepat_Steatosis_48hr_up','ATG_AP_1_CIS_dn','ATG_AP_1_CIS_up','ATG_AP_2_CIS_dn','ATG_AP_2_CIS_up','ATG_AR_TRANS_dn','ATG_AR_TRANS_up','ATG_Ahr_CIS_dn','ATG_Ahr_CIS_up','ATG_BRE_CIS_dn','ATG_BRE_CIS_up','ATG_CAR_TRANS_dn','ATG_CAR_TRANS_up','ATG_CMV_CIS_dn','ATG_CMV_CIS_up','ATG_CRE_CIS_dn','ATG_CRE_CIS_up','ATG_C_EBP_CIS_dn','ATG_C_EBP_CIS_up','ATG_DR4_LXR_CIS_dn','ATG_DR4_LXR_CIS_up','ATG_DR5_CIS_dn','ATG_DR5_CIS_up','ATG_E2F_CIS_dn','ATG_E2F_CIS_up','ATG_EGR_CIS_up','ATG_ERE_CIS_dn','ATG_ERE_CIS_up','ATG_ERRa_TRANS_dn','ATG_ERRg_TRANS_dn','ATG_ERRg_TRANS_up','ATG_ERa_TRANS_up','ATG_E_Box_CIS_dn','ATG_E_Box_CIS_up','ATG_Ets_CIS_dn','ATG_Ets_CIS_up','ATG_FXR_TRANS_up','ATG_FoxA2_CIS_dn','ATG_FoxA2_CIS_up','ATG_FoxO_CIS_dn','ATG_FoxO_CIS_up','ATG_GAL4_TRANS_dn','ATG_GATA_CIS_dn','ATG_GATA_CIS_up','ATG_GLI_CIS_dn','ATG_GLI_CIS_up','ATG_GRE_CIS_dn','ATG_GRE_CIS_up','ATG_GR_TRANS_dn','ATG_GR_TRANS_up','ATG_HIF1a_CIS_dn','ATG_HIF1a_CIS_up','ATG_HNF4a_TRANS_dn','ATG_HNF4a_TRANS_up','ATG_HNF6_CIS_dn','ATG_HNF6_CIS_up','ATG_HSE_CIS_dn','ATG_HSE_CIS_up','ATG_IR1_CIS_dn','ATG_IR1_CIS_up','ATG_ISRE_CIS_dn','ATG_ISRE_CIS_up','ATG_LXRa_TRANS_dn','ATG_LXRa_TRANS_up','ATG_LXRb_TRANS_dn','ATG_LXRb_TRANS_up','ATG_MRE_CIS_up','ATG_M_06_TRANS_up','ATG_M_19_CIS_dn','ATG_M_19_TRANS_dn','ATG_M_19_TRANS_up','ATG_M_32_CIS_dn','ATG_M_32_CIS_up','ATG_M_32_TRANS_dn','ATG_M_32_TRANS_up','ATG_M_61_TRANS_up','ATG_Myb_CIS_dn','ATG_Myb_CIS_up','ATG_Myc_CIS_dn','ATG_Myc_CIS_up','ATG_NFI_CIS_dn','ATG_NFI_CIS_up','ATG_NF_kB_CIS_dn','ATG_NF_kB_CIS_up','ATG_NRF1_CIS_dn','ATG_NRF1_CIS_up','ATG_NRF2_ARE_CIS_dn','ATG_NRF2_ARE_CIS_up','ATG_NURR1_TRANS_dn','ATG_NURR1_TRANS_up','ATG_Oct_MLP_CIS_dn','ATG_Oct_MLP_CIS_up','ATG_PBREM_CIS_dn','ATG_PBREM_CIS_up','ATG_PPARa_TRANS_dn','ATG_PPARa_TRANS_up','ATG_PPARd_TRANS_up','ATG_PPARg_TRANS_up','ATG_PPRE_CIS_dn','ATG_PPRE_CIS_up','ATG_PXRE_CIS_dn','ATG_PXRE_CIS_up','ATG_PXR_TRANS_dn','ATG_PXR_TRANS_up','ATG_Pax6_CIS_up','ATG_RARa_TRANS_dn','ATG_RARa_TRANS_up','ATG_RARb_TRANS_dn','ATG_RARb_TRANS_up','ATG_RARg_TRANS_dn','ATG_RARg_TRANS_up','ATG_RORE_CIS_dn','ATG_RORE_CIS_up','ATG_RORb_TRANS_dn','ATG_RORg_TRANS_dn','ATG_RORg_TRANS_up','ATG_RXRa_TRANS_dn','ATG_RXRa_TRANS_up','ATG_RXRb_TRANS_dn','ATG_RXRb_TRANS_up','ATG_SREBP_CIS_dn','ATG_SREBP_CIS_up','ATG_STAT3_CIS_dn','ATG_STAT3_CIS_up','ATG_Sox_CIS_dn','ATG_Sox_CIS_up','ATG_Sp1_CIS_dn','ATG_Sp1_CIS_up','ATG_TAL_CIS_dn','ATG_TAL_CIS_up','ATG_TA_CIS_dn','ATG_TA_CIS_up','ATG_TCF_b_cat_CIS_dn','ATG_TCF_b_cat_CIS_up','ATG_TGFb_CIS_dn','ATG_TGFb_CIS_up','ATG_THRa1_TRANS_dn','ATG_THRa1_TRANS_up','ATG_VDRE_CIS_dn','ATG_VDRE_CIS_up','ATG_VDR_TRANS_dn','ATG_VDR_TRANS_up','ATG_XTT_Cytotoxicity_up','ATG_Xbp1_CIS_dn','ATG_Xbp1_CIS_up','ATG_p53_CIS_dn','ATG_p53_CIS_up','BSK_3C_Eselectin_down','BSK_3C_HLADR_down','BSK_3C_ICAM1_down','BSK_3C_IL8_down','BSK_3C_MCP1_down','BSK_3C_MIG_down','BSK_3C_Proliferation_down','BSK_3C_SRB_down','BSK_3C_Thrombomodulin_down','BSK_3C_Thrombomodulin_up','BSK_3C_TissueFactor_down','BSK_3C_TissueFactor_up','BSK_3C_VCAM1_down','BSK_3C_Vis_down','BSK_3C_uPAR_down','BSK_4H_Eotaxin3_down','BSK_4H_MCP1_down','BSK_4H_Pselectin_down','BSK_4H_Pselectin_up','BSK_4H_SRB_down','BSK_4H_VCAM1_down','BSK_4H_VEGFRII_down','BSK_4H_uPAR_down','BSK_4H_uPAR_up','BSK_BE3C_HLADR_down','BSK_BE3C_IL1a_down','BSK_BE3C_IP10_down','BSK_BE3C_MIG_down','BSK_BE3C_MMP1_down','BSK_BE3C_MMP1_up','BSK_BE3C_PAI1_down','BSK_BE3C_SRB_down','BSK_BE3C_TGFb1_down','BSK_BE3C_tPA_down','BSK_BE3C_uPAR_down','BSK_BE3C_uPAR_up','BSK_BE3C_uPA_down','BSK_CASM3C_HLADR_down','BSK_CASM3C_IL6_down','BSK_CASM3C_IL6_up','BSK_CASM3C_IL8_down','BSK_CASM3C_LDLR_down','BSK_CASM3C_LDLR_up','BSK_CASM3C_MCP1_down','BSK_CASM3C_MCP1_up','BSK_CASM3C_MCSF_down','BSK_CASM3C_MCSF_up','BSK_CASM3C_MIG_down','BSK_CASM3C_Proliferation_down','BSK_CASM3C_Proliferation_up','BSK_CASM3C_SAA_down','BSK_CASM3C_SAA_up','BSK_CASM3C_SRB_down','BSK_CASM3C_Thrombomodulin_down','BSK_CASM3C_Thrombomodulin_up','BSK_CASM3C_TissueFactor_down','BSK_CASM3C_VCAM1_down','BSK_CASM3C_VCAM1_up','BSK_CASM3C_uPAR_down','BSK_CASM3C_uPAR_up','BSK_KF3CT_ICAM1_down','BSK_KF3CT_IL1a_down','BSK_KF3CT_IP10_down','BSK_KF3CT_IP10_up','BSK_KF3CT_MCP1_down','BSK_KF3CT_MCP1_up','BSK_KF3CT_MMP9_down','BSK_KF3CT_SRB_down','BSK_KF3CT_TGFb1_down','BSK_KF3CT_TIMP2_down','BSK_KF3CT_uPA_down','BSK_LPS_CD40_down','BSK_LPS_Eselectin_down','BSK_LPS_Eselectin_up','BSK_LPS_IL1a_down','BSK_LPS_IL1a_up','BSK_LPS_IL8_down','BSK_LPS_IL8_up','BSK_LPS_MCP1_down','BSK_LPS_MCSF_down','BSK_LPS_PGE2_down','BSK_LPS_PGE2_up','BSK_LPS_SRB_down','BSK_LPS_TNFa_down','BSK_LPS_TNFa_up','BSK_LPS_TissueFactor_down','BSK_LPS_TissueFactor_up','BSK_LPS_VCAM1_down','BSK_SAg_CD38_down','BSK_SAg_CD40_down','BSK_SAg_CD69_down','BSK_SAg_Eselectin_down','BSK_SAg_Eselectin_up','BSK_SAg_IL8_down','BSK_SAg_IL8_up','BSK_SAg_MCP1_down','BSK_SAg_MIG_down','BSK_SAg_PBMCCytotoxicity_down','BSK_SAg_PBMCCytotoxicity_up','BSK_SAg_Proliferation_down','BSK_SAg_SRB_down','BSK_hDFCGF_CollagenIII_down','BSK_hDFCGF_EGFR_down','BSK_hDFCGF_EGFR_up','BSK_hDFCGF_IL8_down','BSK_hDFCGF_IP10_down','BSK_hDFCGF_MCSF_down','BSK_hDFCGF_MIG_down','BSK_hDFCGF_MMP1_down','BSK_hDFCGF_MMP1_up','BSK_hDFCGF_PAI1_down','BSK_hDFCGF_Proliferation_down','BSK_hDFCGF_SRB_down','BSK_hDFCGF_TIMP1_down','BSK_hDFCGF_VCAM1_down','CEETOX_H295R_11DCORT_dn','CEETOX_H295R_ANDR_dn','CEETOX_H295R_CORTISOL_dn','CEETOX_H295R_DOC_dn','CEETOX_H295R_DOC_up','CEETOX_H295R_ESTRADIOL_dn','CEETOX_H295R_ESTRADIOL_up','CEETOX_H295R_ESTRONE_dn','CEETOX_H295R_ESTRONE_up','CEETOX_H295R_OHPREG_up','CEETOX_H295R_OHPROG_dn','CEETOX_H295R_OHPROG_up','CEETOX_H295R_PROG_up','CEETOX_H295R_TESTO_dn','CLD_ABCB1_48hr','CLD_ABCG2_48hr','CLD_CYP1A1_24hr','CLD_CYP1A1_48hr','CLD_CYP1A1_6hr','CLD_CYP1A2_24hr','CLD_CYP1A2_48hr','CLD_CYP1A2_6hr','CLD_CYP2B6_24hr','CLD_CYP2B6_48hr','CLD_CYP2B6_6hr','CLD_CYP3A4_24hr','CLD_CYP3A4_48hr','CLD_CYP3A4_6hr','CLD_GSTA2_48hr','CLD_SULT2A_24hr','CLD_SULT2A_48hr','CLD_UGT1A1_24hr','CLD_UGT1A1_48hr','NCCT_HEK293T_CellTiterGLO','NCCT_QuantiLum_inhib_2_dn','NCCT_QuantiLum_inhib_dn','NCCT_TPO_AUR_dn','NCCT_TPO_GUA_dn','NHEERL_ZF_144hpf_TERATOSCORE_up','NVS_ADME_hCYP19A1','NVS_ADME_hCYP1A1','NVS_ADME_hCYP1A2','NVS_ADME_hCYP2A6','NVS_ADME_hCYP2B6','NVS_ADME_hCYP2C19','NVS_ADME_hCYP2C9','NVS_ADME_hCYP2D6','NVS_ADME_hCYP3A4','NVS_ADME_hCYP4F12','NVS_ADME_rCYP2C12','NVS_ENZ_hAChE','NVS_ENZ_hAMPKa1','NVS_ENZ_hAurA','NVS_ENZ_hBACE','NVS_ENZ_hCASP5','NVS_ENZ_hCK1D','NVS_ENZ_hDUSP3','NVS_ENZ_hES','NVS_ENZ_hElastase','NVS_ENZ_hFGFR1','NVS_ENZ_hGSK3b','NVS_ENZ_hMMP1','NVS_ENZ_hMMP13','NVS_ENZ_hMMP2','NVS_ENZ_hMMP3','NVS_ENZ_hMMP7','NVS_ENZ_hMMP9','NVS_ENZ_hPDE10','NVS_ENZ_hPDE4A1','NVS_ENZ_hPDE5','NVS_ENZ_hPI3Ka','NVS_ENZ_hPTEN','NVS_ENZ_hPTPN11','NVS_ENZ_hPTPN12','NVS_ENZ_hPTPN13','NVS_ENZ_hPTPN9','NVS_ENZ_hPTPRC','NVS_ENZ_hSIRT1','NVS_ENZ_hSIRT2','NVS_ENZ_hTrkA','NVS_ENZ_hVEGFR2','NVS_ENZ_oCOX1','NVS_ENZ_oCOX2','NVS_ENZ_rAChE','NVS_ENZ_rCNOS','NVS_ENZ_rMAOAC','NVS_ENZ_rMAOAP','NVS_ENZ_rMAOBC','NVS_ENZ_rMAOBP','NVS_ENZ_rabI2C','NVS_GPCR_bAdoR_NonSelective','NVS_GPCR_bDR_NonSelective','NVS_GPCR_g5HT4','NVS_GPCR_gH2','NVS_GPCR_gLTB4','NVS_GPCR_gLTD4','NVS_GPCR_gMPeripheral_NonSelective','NVS_GPCR_gOpiateK','NVS_GPCR_h5HT2A','NVS_GPCR_h5HT5A','NVS_GPCR_h5HT6','NVS_GPCR_h5HT7','NVS_GPCR_hAT1','NVS_GPCR_hAdoRA1','NVS_GPCR_hAdoRA2a','NVS_GPCR_hAdra2A','NVS_GPCR_hAdra2C','NVS_GPCR_hAdrb1','NVS_GPCR_hAdrb2','NVS_GPCR_hAdrb3','NVS_GPCR_hDRD1','NVS_GPCR_hDRD2s','NVS_GPCR_hDRD4.4','NVS_GPCR_hH1','NVS_GPCR_hLTB4_BLT1','NVS_GPCR_hM1','NVS_GPCR_hM2','NVS_GPCR_hM3','NVS_GPCR_hM4','NVS_GPCR_hNK2','NVS_GPCR_hOpiate_D1','NVS_GPCR_hOpiate_mu','NVS_GPCR_hTXA2','NVS_GPCR_p5HT2C','NVS_GPCR_r5HT1_NonSelective','NVS_GPCR_r5HT_NonSelective','NVS_GPCR_rAdra1B','NVS_GPCR_rAdra1_NonSelective','NVS_GPCR_rAdra2_NonSelective','NVS_GPCR_rAdrb_NonSelective','NVS_GPCR_rNK1','NVS_GPCR_rNK3','NVS_GPCR_rOpiate_NonSelective','NVS_GPCR_rOpiate_NonSelectiveNa','NVS_GPCR_rSST','NVS_GPCR_rTRH','NVS_GPCR_rV1','NVS_GPCR_rabPAF','NVS_GPCR_rmAdra2B','NVS_IC_hKhERGCh','NVS_IC_rCaBTZCHL','NVS_IC_rCaDHPRCh_L','NVS_IC_rNaCh_site2','NVS_LGIC_bGABARa1','NVS_LGIC_h5HT3','NVS_LGIC_hNNR_NBungSens','NVS_LGIC_rGABAR_NonSelective','NVS_LGIC_rNNR_BungSens','NVS_MP_hPBR','NVS_MP_rPBR','NVS_NR_bER','NVS_NR_bPR','NVS_NR_cAR','NVS_NR_hAR','NVS_NR_hCAR_Antagonist','NVS_NR_hER','NVS_NR_hFXR_Agonist','NVS_NR_hFXR_Antagonist','NVS_NR_hGR','NVS_NR_hPPARa','NVS_NR_hPPARg','NVS_NR_hPR','NVS_NR_hPXR','NVS_NR_hRAR_Antagonist','NVS_NR_hRARa_Agonist','NVS_NR_hTRa_Antagonist','NVS_NR_mERa','NVS_NR_rAR','NVS_NR_rMR','NVS_OR_gSIGMA_NonSelective','NVS_TR_gDAT','NVS_TR_hAdoT','NVS_TR_hDAT','NVS_TR_hNET','NVS_TR_hSERT','NVS_TR_rNET','NVS_TR_rSERT','NVS_TR_rVMAT2','OT_AR_ARELUC_AG_1440','OT_AR_ARSRC1_0480','OT_AR_ARSRC1_0960','OT_ER_ERaERa_0480','OT_ER_ERaERa_1440','OT_ER_ERaERb_0480','OT_ER_ERaERb_1440','OT_ER_ERbERb_0480','OT_ER_ERbERb_1440','OT_ERa_EREGFP_0120','OT_ERa_EREGFP_0480','OT_FXR_FXRSRC1_0480','OT_FXR_FXRSRC1_1440','OT_NURR1_NURR1RXRa_0480','OT_NURR1_NURR1RXRa_1440','TOX21_ARE_BLA_Agonist_ch1','TOX21_ARE_BLA_Agonist_ch2','TOX21_ARE_BLA_agonist_ratio','TOX21_ARE_BLA_agonist_viability','TOX21_AR_BLA_Agonist_ch1','TOX21_AR_BLA_Agonist_ch2','TOX21_AR_BLA_Agonist_ratio','TOX21_AR_BLA_Antagonist_ch1','TOX21_AR_BLA_Antagonist_ch2','TOX21_AR_BLA_Antagonist_ratio','TOX21_AR_BLA_Antagonist_viability','TOX21_AR_LUC_MDAKB2_Agonist','TOX21_AR_LUC_MDAKB2_Antagonist','TOX21_AR_LUC_MDAKB2_Antagonist2','TOX21_AhR_LUC_Agonist','TOX21_Aromatase_Inhibition','TOX21_AutoFluor_HEK293_Cell_blue','TOX21_AutoFluor_HEK293_Media_blue','TOX21_AutoFluor_HEPG2_Cell_blue','TOX21_AutoFluor_HEPG2_Cell_green','TOX21_AutoFluor_HEPG2_Media_blue','TOX21_AutoFluor_HEPG2_Media_green','TOX21_ELG1_LUC_Agonist','TOX21_ERa_BLA_Agonist_ch1','TOX21_ERa_BLA_Agonist_ch2','TOX21_ERa_BLA_Agonist_ratio','TOX21_ERa_BLA_Antagonist_ch1','TOX21_ERa_BLA_Antagonist_ch2','TOX21_ERa_BLA_Antagonist_ratio','TOX21_ERa_BLA_Antagonist_viability','TOX21_ERa_LUC_BG1_Agonist','TOX21_ERa_LUC_BG1_Antagonist','TOX21_ESRE_BLA_ch1','TOX21_ESRE_BLA_ch2','TOX21_ESRE_BLA_ratio','TOX21_ESRE_BLA_viability','TOX21_FXR_BLA_Antagonist_ch1','TOX21_FXR_BLA_Antagonist_ch2','TOX21_FXR_BLA_agonist_ch2','TOX21_FXR_BLA_agonist_ratio','TOX21_FXR_BLA_antagonist_ratio','TOX21_FXR_BLA_antagonist_viability','TOX21_GR_BLA_Agonist_ch1','TOX21_GR_BLA_Agonist_ch2','TOX21_GR_BLA_Agonist_ratio','TOX21_GR_BLA_Antagonist_ch2','TOX21_GR_BLA_Antagonist_ratio','TOX21_GR_BLA_Antagonist_viability','TOX21_HSE_BLA_agonist_ch1','TOX21_HSE_BLA_agonist_ch2','TOX21_HSE_BLA_agonist_ratio','TOX21_HSE_BLA_agonist_viability','TOX21_MMP_ratio_down','TOX21_MMP_ratio_up','TOX21_MMP_viability','TOX21_NFkB_BLA_agonist_ch1','TOX21_NFkB_BLA_agonist_ch2','TOX21_NFkB_BLA_agonist_ratio','TOX21_NFkB_BLA_agonist_viability','TOX21_PPARd_BLA_Agonist_viability','TOX21_PPARd_BLA_Antagonist_ch1','TOX21_PPARd_BLA_agonist_ch1','TOX21_PPARd_BLA_agonist_ch2','TOX21_PPARd_BLA_agonist_ratio','TOX21_PPARd_BLA_antagonist_ratio','TOX21_PPARd_BLA_antagonist_viability','TOX21_PPARg_BLA_Agonist_ch1','TOX21_PPARg_BLA_Agonist_ch2','TOX21_PPARg_BLA_Agonist_ratio','TOX21_PPARg_BLA_Antagonist_ch1','TOX21_PPARg_BLA_antagonist_ratio','TOX21_PPARg_BLA_antagonist_viability','TOX21_TR_LUC_GH3_Agonist','TOX21_TR_LUC_GH3_Antagonist','TOX21_VDR_BLA_Agonist_viability','TOX21_VDR_BLA_Antagonist_ch1','TOX21_VDR_BLA_agonist_ch2','TOX21_VDR_BLA_agonist_ratio','TOX21_VDR_BLA_antagonist_ratio','TOX21_VDR_BLA_antagonist_viability','TOX21_p53_BLA_p1_ch1','TOX21_p53_BLA_p1_ch2','TOX21_p53_BLA_p1_ratio','TOX21_p53_BLA_p1_viability','TOX21_p53_BLA_p2_ch1','TOX21_p53_BLA_p2_ch2','TOX21_p53_BLA_p2_ratio','TOX21_p53_BLA_p2_viability','TOX21_p53_BLA_p3_ch1','TOX21_p53_BLA_p3_ch2','TOX21_p53_BLA_p3_ratio','TOX21_p53_BLA_p3_viability','TOX21_p53_BLA_p4_ch1','TOX21_p53_BLA_p4_ch2','TOX21_p53_BLA_p4_ratio','TOX21_p53_BLA_p4_viability','TOX21_p53_BLA_p5_ch1','TOX21_p53_BLA_p5_ch2','TOX21_p53_BLA_p5_ratio','TOX21_p53_BLA_p5_viability','Tanguay_ZF_120hpf_AXIS_up','Tanguay_ZF_120hpf_ActivityScore','Tanguay_ZF_120hpf_BRAI_up','Tanguay_ZF_120hpf_CFIN_up','Tanguay_ZF_120hpf_CIRC_up','Tanguay_ZF_120hpf_EYE_up','Tanguay_ZF_120hpf_JAW_up','Tanguay_ZF_120hpf_MORT_up','Tanguay_ZF_120hpf_OTIC_up','Tanguay_ZF_120hpf_PE_up','Tanguay_ZF_120hpf_PFIN_up','Tanguay_ZF_120hpf_PIG_up','Tanguay_ZF_120hpf_SNOU_up','Tanguay_ZF_120hpf_SOMI_up','Tanguay_ZF_120hpf_SWIM_up','Tanguay_ZF_120hpf_TRUN_up','Tanguay_ZF_120hpf_TR_up','Tanguay_ZF_120hpf_YSE_up']
        elif dataset == 'ClinTox':
            task_name = ['FDA_APPROVED','CT_TOX']
        elif dataset == 'Tox21':
            task_name = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD','NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        elif dataset == 'HIV':
            task_name = ['HIV_active']
        elif dataset == 'MUV':
            task_name = ["MUV-466","MUV-548","MUV-600","MUV-644","MUV-652","MUV-689","MUV-692","MUV-712","MUV-713","MUV-733","MUV-737","MUV-810","MUV-832","MUV-846","MUV-852","MUV-858","MUV-859"]
        elif dataset == 'PCBA':
            task_name = ['PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457', 'PCBA-1458', 'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469', 'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 'PCBA-1634', 'PCBA-1688', 'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242', 'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546', 'PCBA-2549', 'PCBA-2551', 'PCBA-2662', 'PCBA-2675', 'PCBA-2676', 'PCBA-411', 'PCBA-463254', 'PCBA-485281', 'PCBA-485290', 'PCBA-485294', 'PCBA-485297', 'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 'PCBA-485349', 'PCBA-485353', 'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 'PCBA-493208', 'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339', 'PCBA-504444', 'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842', 'PCBA-504845', 'PCBA-504847', 'PCBA-504891', 'PCBA-540276', 'PCBA-540317', 'PCBA-588342', 'PCBA-588453', 'PCBA-588456', 'PCBA-588579', 'PCBA-588590', 'PCBA-588591', 'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 'PCBA-602233', 'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171', 'PCBA-624173', 'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288', 'PCBA-624291', 'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635', 'PCBA-651644', 'PCBA-651768', 'PCBA-651965', 'PCBA-652025', 'PCBA-652104', 'PCBA-652105', 'PCBA-652106', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979', 'PCBA-720504', 'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 'PCBA-720553', 'PCBA-720579', 'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709', 'PCBA-720711', 'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881', 'PCBA-883', 'PCBA-884', 'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899', 'PCBA-902', 'PCBA-903', 'PCBA-904', 'PCBA-912', 'PCBA-914', 'PCBA-915', 'PCBA-924', 'PCBA-925', 'PCBA-926', 'PCBA-927', 'PCBA-938', 'PCBA-995']
        elif dataset == 'FreeSolv':
            task_name = ['expt']
        elif dataset == 'Lipo':
            task_name = ['exp']
        elif dataset == 'ESOL':
            task_name = ['measured log solubility in mols per litre']
        elif dataset == 'QM7':
            task_name = ["u0_atom"]
        elif dataset == 'QM8':
            task_name = ["E1-CC2","E2-CC2","f1-CC2","f2-CC2","E1-PBE0","E2-PBE0","f1-PBE0","f2-PBE0","E1-PBE0.1","E1-CAM","E2-CAM","f1-CAM","f2-CAM"]
        elif dataset == 'QM9':
            task_name = ["mu","alpha","homo","lumo","gap","r2","zpve","u0","u298","h298","g298","cv"]
        
        
    print('Read and process the collected data...')
    file = pd.read_csv('./data/' + dataset + 'Scaffold.csv', header=0)

    if 'smiles' in file:
        smi_name = 'smiles'
    else:
        smi_name = 'mol'
        
    if print_info:
        print('----------------------------------------')
        print('Dataset: ', dataset)
        print('Example: ')
        print(file.iloc[0])
        print('Number of molecules:', file.shape[0])
        
    all_smiles_y = []
    file = file.where(pd.notnull(file), -1)
    for i in range(file.shape[0]):
        target = []
        for j in task_name:
            target.append(file[j][i])
        all_smiles_y.append([file[smi_name][i],target])
    return all_smiles_y

class Molecule():
    def __init__(self, smiles_y, dataset, bool_random=True, max_len=100, max_ring=15, print_info=False):
        self.max_len = max_len # maximum number of atoms in a molecule
        self.max_ring = max_ring
        self.smiles, self.targets = smiles_y
        self.nb_MP = len(self.targets)
        self.exist = True
        self.dataset = dataset # name of dataset
        self.process_mol_with_RDKit()
       
    def mdoel_needed_info(self):
        return self.nb_node_features, self.nb_edge_features, self.nb_MP
    
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
        if self.dataset in ['PCBA', 'SIDER', 'HIV'] and nb_atoms <= self.max_len:
            special_max_len = [24, 50, 75, 100, 150, 200, 250, 300, 350, 500]
            for i in special_max_len:
                if nb_atoms <= i:
                    self.max_len = min(i, self.max_len)
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

def Read_mol_data(dataset, task_name=None, target_type='classification'):
    assert target_type in ['classification', 'regression']
    max_len = {'Tox21':[100, 135, 125], 'HIV': [225, 150, 185], 'SIDER':[500, 500, 500], 'MUV':[45, 50, 45], 'BBBP':[135, 125, 100], 'BACE':[100, 60, 100], 'ClinTox':[125, 140, 95], 'ToxCast':[125, 100, 110], 'PCBA':[350, 350, 350], 'ESOL':[55, 40, 45], 'FreeSolv':[25, 25, 25], 'Lipo':[115, 100, 65], 'QM7':[7, 7, 7], 'QM8':[8, 8, 8], 'QM9':[9, 9, 9]}

    nb_cpu = 10
    nb_time_processing = 20
    all_smiles_y = load_data(dataset, task_name)
    
    if dataset in max_len:
        train_max_len, val_max_len, test_max_len = max_len[dataset]
    else:
        max_len = get_mol_max_length(dataset)
        train_max_len = val_max_len = test_max_len = max_len
        
    if target_type == 'classification':
        mean = std = None
    else:
        all_targets = []
        for i in all_smiles_y:
            all_targets.append(i[1])
        mean = torch.Tensor(np.mean(all_targets, axis=0))
        std = torch.Tensor(np.mean(all_targets, axis=0))
            
    pos_times_HIV = 20
    pos_times_ClinTox = 12 
    pos_times_MUV = 2

    num_train = int(len(all_smiles_y) * 0.8)
    num_val= int(len(all_smiles_y) * 0.1)
    train_smiles_y = all_smiles_y[:num_train]
    val_smiles_y = all_smiles_y[num_train:num_train+num_val]
    test_smiles_y = all_smiles_y[num_train+num_val:]

    new_train_smiles_y = []
    mol_train = []
    mol_val = []
    mol_test = []
    if dataset in ['HIV']:
        for i in range(len(train_smiles_y)):
            if train_smiles_y[i][1][0] == 0:
                new_train_smiles_y.append(train_smiles_y[i])
            else:
                for _ in range(pos_times_HIV):
                    new_train_smiles_y.append(train_smiles_y[i])
    elif dataset in ['ClinTox']:
        for i in range(len(train_smiles_y)):
            if train_smiles_y[i][1][0] == 1:
                new_train_smiles_y.append(train_smiles_y[i])
            else:
                for _ in range(pos_times_ClinTox):
                    new_train_smiles_y.append(train_smiles_y[i])
    elif dataset in ['MUV']:
        for i in range(len(train_smiles_y)):
            if 1 in train_smiles_y[i][1]:
                for _ in range(pos_times_MUV):
                    new_train_smiles_y.append(train_smiles_y[i])
            else:
                new_train_smiles_y.append(train_smiles_y[i])
    else:
        for i in range(len(train_smiles_y)):
            new_train_smiles_y.append(train_smiles_y[i])

    random.shuffle(new_train_smiles_y)

    if len(new_train_smiles_y) > 2e5:
        nb = 100
    elif len(new_train_smiles_y) > 1e4:
        nb = 10
    else:
        nb = 1
    nb_part = int(np.ceil(len(new_train_smiles_y)/nb))
    for j in range(nb):
        part_smiles_y = new_train_smiles_y[j*nb_part:(j+1)*nb_part]
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
    return mol_train, mol_val, mol_test, mean, std

def PreProcess(mol):
    targets = []
    inputs = {'nodes_features':[], 'edges_features':[]}
    for i in range(len(mol)):
        # for targets
        targets.append(mol[i].targets)
        
        # for inputs
        nodes_features, edges_features = mol[i].get_inputs_features()
        node_len = nodes_features.shape[0]
        edge_len = edges_features.shape[0]
        inputs['nodes_features'] += [nodes_features.tolist()]
        inputs['edges_features'] += [edges_features.tolist()]
        
    targets = torch.Tensor(targets)
    for name in inputs:
        if 'features' in name:
            inputs[name] = torch.Tensor(inputs[name])
    
    return inputs, targets

def Generate_dataloader(dataset, mol_train, mol_val, mol_test):
    train_dataloader = []
    val_dataloader = []
    test_dataloader = []
    if dataset in ['SIDER', 'PCBA', 'HIV']:
        special_max_len = [24, 50, 75, 100, 150, 200, 250, 300, 350, 500]

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

            if i <= 350:
                bsz = 32
            else:
                bsz = 8
            num_train_dataloader = int(np.ceil(len(new_mol_train)/bsz))
            num_val_dataloader = int(np.ceil(len(new_mol_val)/bsz))
            num_test_dataloader = int(np.ceil(len(new_mol_test)/bsz))

            for j in range(num_train_dataloader):
                inputs, targets = PreProcess(new_mol_train[j * bsz:(j+1)*bsz])
                train_dataloader.append([inputs, targets])

            for j in range(num_val_dataloader):
                inputs, targets = PreProcess(new_mol_val[j * bsz: (j+1) * bsz])
                val_dataloader.append([inputs, targets])

            for j in range(num_test_dataloader):
                inputs, targets = PreProcess(new_mol_test[j * bsz: (j+1) * bsz])
                test_dataloader.append([inputs, targets])
        random.shuffle(train_dataloader)
    else:
        bsz = 32
        num_train_dataloader = int(np.ceil(len(mol_train)/bsz))
        num_val_dataloader = int(np.ceil(len(mol_val)/bsz))
        num_test_dataloader = int(np.ceil(len(mol_test)/bsz))

        for i in range(num_train_dataloader):
            inputs, targets = PreProcess(mol_train[i * bsz:(i+1)*bsz])
            train_dataloader.append([inputs, targets])

        for i in range(num_val_dataloader):
            inputs, targets = PreProcess(mol_val[i * bsz: (i+1) * bsz])
            val_dataloader.append([inputs, targets])

        for i in range(num_test_dataloader):
            inputs, targets = PreProcess(mol_test[i * bsz: (i+1) * bsz])
            test_dataloader.append([inputs, targets])
    return train_dataloader, val_dataloader,test_dataloader