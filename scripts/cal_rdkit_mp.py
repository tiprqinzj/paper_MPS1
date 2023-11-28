from multiprocessing import Pool
import os, subprocess, time, math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import GraphDescriptors, Descriptors, MolSurf
from rdkit.Chem.AllChem import GetMACCSKeysFingerprint, GetMorganFingerprintAsBitVect

def cal_time(since):
    now = time.time()
    s = now - since

    if s > 3600:
        h = math.floor(s / 3600)
        m = math.floor((s - h * 3600) / 60)
        s = s - h * 3600 - m * 60
        out = '{}h {}m {:.0f}s'.format(h, m, s)
    else:
        m = math.floor(s / 60)
        s = s - m * 60
        out = '{}m {:.0f}s'.format(m, s)
    return out

def get_RDKit2D_name():
    
    name_trad2D = [
        'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
        'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
        'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
        'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
        'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'EState_VSA11',
        'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
        'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'VSA_EState10',
        'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
        'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
        'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
        'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10',
        'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
        'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
        'ExactMolWt', 'MolWt', 'HeavyAtomMolWt', 'MolLogP', 'MolMR', 'TPSA', 'Ipc', 'qed',
        'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha',
        'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
        'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
        'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge',
        'HeavyAtomCount', 'RingCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
        'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
        'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons',
        'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
        'NumValenceElectrons'
    ]

    name_fr = [
        'fr_Al_COO',
        'fr_Al_OH',
        'fr_Al_OH_noTert',
        'fr_ArN',
        'fr_Ar_COO',
        'fr_Ar_N',
        'fr_Ar_NH',
        'fr_Ar_OH',
        'fr_COO',
        'fr_COO2',
        'fr_C_O',
        'fr_C_O_noCOO',
        'fr_C_S',
        'fr_HOCCN',
        'fr_Imine',
        'fr_NH0',
        'fr_NH1',
        'fr_NH2',
        'fr_N_O',
        'fr_Ndealkylation1',
        'fr_Ndealkylation2',
        'fr_Nhpyrrole',
        'fr_SH',
        'fr_aldehyde',
        'fr_alkyl_carbamate',
        'fr_alkyl_halide',
        'fr_allylic_oxid',
        'fr_amide',
        'fr_amidine',
        'fr_aniline',
        'fr_aryl_methyl',
        'fr_azide',
        'fr_azo',
        'fr_barbitur',
        'fr_benzene',
        'fr_benzodiazepine',
        'fr_bicyclic',
        'fr_diazo',
        'fr_dihydropyridine',
        'fr_epoxide',
        'fr_ester',
        'fr_ether',
        'fr_furan',
        'fr_guanido',
        'fr_halogen',
        'fr_hdrzine',
        'fr_hdrzone',
        'fr_imidazole',
        'fr_imide',
        'fr_isocyan',
        'fr_isothiocyan',
        'fr_ketone',
        'fr_ketone_Topliss',
        'fr_lactam',
        'fr_lactone',
        'fr_methoxy',
        'fr_morpholine',
        'fr_nitrile',
        'fr_nitro',
        'fr_nitro_arom',
        'fr_nitro_arom_nonortho',
        'fr_nitroso',
        'fr_oxazole',
        'fr_oxime',
        'fr_para_hydroxylation',
        'fr_phenol',
        'fr_phenol_noOrthoHbond',
        'fr_phos_acid',
        'fr_phos_ester',
        'fr_piperdine',
        'fr_piperzine',
        'fr_priamide',
        'fr_prisulfonamd',
        'fr_pyridine',
        'fr_quatN',
        'fr_sulfide',
        'fr_sulfonamd',
        'fr_sulfone',
        'fr_term_acetylene',
        'fr_tetrazole',
        'fr_thiazole',
        'fr_thiocyan',
        'fr_thiophene',
        'fr_unbrch_alkane',
        'fr_urea'
    ]
    
    return name_trad2D + name_fr


def cal_RDKit2D(mol):
    
    BCUT_ls = [
        Descriptors.BCUT2D_MWHI(mol), Descriptors.BCUT2D_MWLOW(mol),
        Descriptors.BCUT2D_CHGHI(mol), Descriptors.BCUT2D_CHGLO(mol),
        Descriptors.BCUT2D_LOGPHI(mol), Descriptors.BCUT2D_LOGPLOW(mol),
        Descriptors.BCUT2D_MRHI(mol), Descriptors.BCUT2D_MRLOW(mol)
    ]

    Graph_ls = [
        GraphDescriptors.BalabanJ(mol), GraphDescriptors.BertzCT(mol),
        GraphDescriptors.Chi0(mol), GraphDescriptors.Chi0n(mol), GraphDescriptors.Chi0v(mol),
        GraphDescriptors.Chi1(mol), GraphDescriptors.Chi1n(mol), GraphDescriptors.Chi1v(mol),
        GraphDescriptors.Chi2n(mol), GraphDescriptors.Chi2v(mol),
        GraphDescriptors.Chi3n(mol), GraphDescriptors.Chi3v(mol),
        GraphDescriptors.Chi4n(mol), GraphDescriptors.Chi4v(mol)
    ]

    EState_ls = [
        Descriptors.EState_VSA1(mol), Descriptors.EState_VSA2(mol), 
        Descriptors.EState_VSA3(mol), Descriptors.EState_VSA4(mol),
        Descriptors.EState_VSA5(mol), Descriptors.EState_VSA6(mol),
        Descriptors.EState_VSA7(mol), Descriptors.EState_VSA8(mol),
        Descriptors.EState_VSA9(mol), Descriptors.EState_VSA10(mol),
        Descriptors.EState_VSA11(mol),
        Descriptors.VSA_EState1(mol), Descriptors.VSA_EState2(mol),
        Descriptors.VSA_EState3(mol), Descriptors.VSA_EState4(mol),
        Descriptors.VSA_EState5(mol), Descriptors.VSA_EState6(mol),
        Descriptors.VSA_EState7(mol), Descriptors.VSA_EState8(mol),
        Descriptors.VSA_EState9(mol), Descriptors.VSA_EState10(mol)
    ]
    MolSurf_ls = [
        MolSurf.PEOE_VSA1(mol), MolSurf.PEOE_VSA2(mol), MolSurf.PEOE_VSA3(mol),
        MolSurf.PEOE_VSA4(mol), MolSurf.PEOE_VSA5(mol), MolSurf.PEOE_VSA6(mol),
        MolSurf.PEOE_VSA7(mol), MolSurf.PEOE_VSA8(mol), MolSurf.PEOE_VSA9(mol),
        MolSurf.PEOE_VSA10(mol), MolSurf.PEOE_VSA11(mol), MolSurf.PEOE_VSA12(mol),
        MolSurf.PEOE_VSA13(mol), MolSurf.PEOE_VSA14(mol),
        MolSurf.SMR_VSA1(mol), MolSurf.SMR_VSA2(mol), MolSurf.SMR_VSA3(mol),
        MolSurf.SMR_VSA4(mol), MolSurf.SMR_VSA5(mol), MolSurf.SMR_VSA6(mol),
        MolSurf.SMR_VSA7(mol), MolSurf.SMR_VSA8(mol), MolSurf.SMR_VSA9(mol),
        MolSurf.SMR_VSA10(mol),
        MolSurf.SlogP_VSA1(mol), MolSurf.SlogP_VSA2(mol), MolSurf.SlogP_VSA3(mol),
        MolSurf.SlogP_VSA4(mol), MolSurf.SlogP_VSA5(mol), MolSurf.SlogP_VSA6(mol),
        MolSurf.SlogP_VSA7(mol), MolSurf.SlogP_VSA8(mol), MolSurf.SlogP_VSA9(mol),
        MolSurf.SlogP_VSA10(mol), MolSurf.SlogP_VSA11(mol), MolSurf.SlogP_VSA12(mol),
    ]

    Property_ls = [
        Descriptors.ExactMolWt(mol),
        Descriptors.MolWt(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol),
        Descriptors.TPSA(mol),
        Descriptors.Ipc(mol),
        Descriptors.qed(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        Descriptors.Kappa3(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.MaxAbsEStateIndex(mol),
        Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MaxEStateIndex(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinAbsEStateIndex(mol),
        Descriptors.MinAbsPartialCharge(mol),
        Descriptors.MinEStateIndex(mol),
        Descriptors.MinPartialCharge(mol)
    ]
    
    Count_ls = [
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.NOCount(mol),
        Descriptors.NumAliphaticCarbocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticCarbocycles(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumSaturatedCarbocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumValenceElectrons(mol)
    ]

    fr_ls = [
        Descriptors.fr_Al_COO(mol),
        Descriptors.fr_Al_OH(mol),
        Descriptors.fr_Al_OH_noTert(mol),
        Descriptors.fr_ArN(mol),
        Descriptors.fr_Ar_COO(mol),
        Descriptors.fr_Ar_N(mol),
        Descriptors.fr_Ar_NH(mol),
        Descriptors.fr_Ar_OH(mol),
        Descriptors.fr_COO(mol),
        Descriptors.fr_COO2(mol),
        Descriptors.fr_C_O(mol),
        Descriptors.fr_C_O_noCOO(mol),
        Descriptors.fr_C_S(mol),
        Descriptors.fr_HOCCN(mol),
        Descriptors.fr_Imine(mol),
        Descriptors.fr_NH0(mol),
        Descriptors.fr_NH1(mol),
        Descriptors.fr_NH2(mol),
        Descriptors.fr_N_O(mol),
        Descriptors.fr_Ndealkylation1(mol),
        Descriptors.fr_Ndealkylation2(mol),
        Descriptors.fr_Nhpyrrole(mol),
        Descriptors.fr_SH(mol),
        Descriptors.fr_aldehyde(mol),
        Descriptors.fr_alkyl_carbamate(mol),
        Descriptors.fr_alkyl_halide(mol),
        Descriptors.fr_allylic_oxid(mol),
        Descriptors.fr_amide(mol),
        Descriptors.fr_amidine(mol),
        Descriptors.fr_aniline(mol),
        Descriptors.fr_aryl_methyl(mol),
        Descriptors.fr_azide(mol),
        Descriptors.fr_azo(mol),
        Descriptors.fr_barbitur(mol),
        Descriptors.fr_benzene(mol),
        Descriptors.fr_benzodiazepine(mol),
        Descriptors.fr_bicyclic(mol),
        Descriptors.fr_diazo(mol),
        Descriptors.fr_dihydropyridine(mol),
        Descriptors.fr_epoxide(mol),
        Descriptors.fr_ester(mol),
        Descriptors.fr_ether(mol),
        Descriptors.fr_furan(mol),
        Descriptors.fr_guanido(mol),
        Descriptors.fr_halogen(mol),
        Descriptors.fr_hdrzine(mol),
        Descriptors.fr_hdrzone(mol),
        Descriptors.fr_imidazole(mol),
        Descriptors.fr_imide(mol),
        Descriptors.fr_isocyan(mol),
        Descriptors.fr_isothiocyan(mol),
        Descriptors.fr_ketone(mol),
        Descriptors.fr_ketone_Topliss(mol),
        Descriptors.fr_lactam(mol),
        Descriptors.fr_lactone(mol),
        Descriptors.fr_methoxy(mol),
        Descriptors.fr_morpholine(mol),
        Descriptors.fr_nitrile(mol),
        Descriptors.fr_nitro(mol),
        Descriptors.fr_nitro_arom(mol),
        Descriptors.fr_nitro_arom_nonortho(mol),
        Descriptors.fr_nitroso(mol),
        Descriptors.fr_oxazole(mol),
        Descriptors.fr_oxime(mol),
        Descriptors.fr_para_hydroxylation(mol),
        Descriptors.fr_phenol(mol),
        Descriptors.fr_phenol_noOrthoHbond(mol),
        Descriptors.fr_phos_acid(mol),
        Descriptors.fr_phos_ester(mol),
        Descriptors.fr_piperdine(mol),
        Descriptors.fr_piperzine(mol),
        Descriptors.fr_priamide(mol),
        Descriptors.fr_prisulfonamd(mol),
        Descriptors.fr_pyridine(mol),
        Descriptors.fr_quatN(mol),
        Descriptors.fr_sulfide(mol),
        Descriptors.fr_sulfonamd(mol),
        Descriptors.fr_sulfone(mol),
        Descriptors.fr_term_acetylene(mol),
        Descriptors.fr_tetrazole(mol),
        Descriptors.fr_thiazole(mol),
        Descriptors.fr_thiocyan(mol),
        Descriptors.fr_thiophene(mol),
        Descriptors.fr_unbrch_alkane(mol),
        Descriptors.fr_urea(mol),
    ]
    
    des_ls = BCUT_ls + Graph_ls + EState_ls + MolSurf_ls + Property_ls + Count_ls + fr_ls
    
    return des_ls


def split_sdf(sdf_file, split_n):

    # count total mols in sdf
    N = 0
    with open(sdf_file) as f:
        for line in f:
            if line == '$$$$\n':
                N += 1
    
    # split_n should be less than N
    if split_n >= N: split_n = 1

    # assign number of mols in each split
    n_each = math.ceil(N / split_n)
    prefix_ls = [sdf_file[:-4] + '_{:0>3d}'.format(i) for i in range(1, split_n + 1)]

    # split original sdf to splitted sdf
    with open(sdf_file) as fr:
        mol_count, file_tag, temp_string = 0, 0, ''

        for line in fr:
            temp_string += line

            if line == '$$$$\n':
                mol_count += 1
            
            if mol_count == n_each:
                with open(prefix_ls[file_tag] + '.sdf', 'w') as fw:
                    fw.write(temp_string)
                
                mol_count = 0
                file_tag += 1
                temp_string = ''
    
    if temp_string != '':
        with open(prefix_ls[file_tag] + '.sdf', 'w') as fw:
            fw.write(temp_string)
    
    return prefix_ls

def cal_rdkit_with_errors(prefix, mode):

    ''' mode: 'MACCS', 'ECFP4', 'RDKit2D'
    '''

    # load sdf file
    suppl = Chem.SDMolSupplier(prefix + '.sdf')

    # calculate
    if mode == 'MACCS':
        name_ls = ['MACCS_' + str(i) for i in range(0, 167)]
        des_ar = np.zeros(shape=(len(suppl), 167))
        for i, mol in enumerate(suppl):
            fp = GetMACCSKeysFingerprint(mol)
            des_ar[i] = fp

    elif mode == 'ECFP4':
        name_ls = ['ECFP4_{}'.format(i) for i in range(1, 1025)]
        des_ar = np.zeros(shape=(len(suppl), 1024))
        for i, mol in enumerate(suppl):
            fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            des_ar[i] = fp

    elif mode == 'RDKit2D':
        name_ls = get_RDKit2D_name()
        des_ar = np.zeros(shape=(len(suppl), len(name_ls)))
        for i, mol in enumerate(suppl):
            des_ls = cal_RDKit2D(mol)
            des_ar[i] = np.array(des_ls)

    else:
        return None

    # save
    out_csv = '{}_{}_cal.csv'.format(prefix, mode)
    pd.DataFrame(des_ar, columns=name_ls).to_csv(out_csv, index=None)

    # remove splitted sdf
    os.remove(prefix + '.sdf')

def pack_cal_rdkit_with_errors(params_ls):
    return cal_rdkit_with_errors(params_ls[0], params_ls[1])


def combine_and_remove(prefix_ls, mode, out_prefix):

    # combine all _cal csv
    fobj_cal = []
    for i, prefix in enumerate(prefix_ls):
        with open('{}_{}_cal.csv'.format(prefix, mode)) as fr:
            if i == 0:
                fobj_cal += fr.readlines()
            else:
                fobj_cal += fr.readlines()[1:]
        
        os.remove('{}_{}_cal.csv'.format(prefix, mode))
    
    # save fobj_cal to csv
    with open('{}_{}_combine.csv'.format(out_prefix, mode), 'w') as fw:
        for line in fobj_cal: fw.write(line)
    
    # load witherrors
    df_combine = pd.read_csv('{}_{}_combine.csv'.format(out_prefix, mode))
    os.remove('{}_{}_combine.csv'.format(out_prefix, mode))
    
    # judge errors
    if mode == 'RDKit2D':
        failed_ls = []
        for i in range(df_combine.shape[0]):
            if df_combine.loc[i].isna().sum() > 0: failed_ls.append(i)
        
        # save failed idx
        with open('{}_{}_failedID.txt'.format(out_prefix, mode), 'w') as fw:
            for idx in failed_ls: fw.write('{}\n'.format(idx))
    
        if len(failed_ls) > 0:
            df_combine = df_combine.drop(index=failed_ls).reset_index(drop=True)
        
    # save
    df_combine.to_csv('{}_{}.csv.gz'.format(out_prefix, mode), index=None, compression='gzip')


def main(sdf_file, mode, split_n):
    ''' mode: 'MACCS', 'ECFP4', 'RDKit2D'
    '''

    time_start = time.time()
    
    # split by sdf file to get prefix ls
    prefix_ls = split_sdf(sdf_file, split_n)

    # make params list
    params_ls = []
    for prefix in prefix_ls:
        params_ls.append((prefix, mode))

    # start calculate
    pool = Pool(processes=split_n)
    pool.map(pack_cal_rdkit_with_errors, params_ls)
    pool.close()
    pool.join()

    # combine and save
    out_prefix = sdf_file[:-4]
    combine_and_remove(prefix_ls, mode, out_prefix)

    print('Done. Time ({})'.format(cal_time(time_start)))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Calculate RDKit fingerprints/descriptors, Multi thread')

    parser.add_argument('--cur_folder', type=str, default=os.getcwd(), metavar='', help='working folder, default pwd')
    parser.add_argument('--sdf_file', type=str, default='data2D.sdf', metavar='', help='default data2D.sdf')
    parser.add_argument('--mode', type=str, metavar='', help='MACCS, ECFP4, RDKit2D')
    parser.add_argument('--split_n', type=int, default=1, metavar='', help='multiple subprocess, default 1')

    # unpack args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    sdf_file = args.sdf_file
    mode = args.mode
    split_n = args.split_n
    
    # working folder
    os.chdir(cur_folder)

    # execute
    main(sdf_file, mode, split_n)
