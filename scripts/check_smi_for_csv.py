import os
import pandas as pd
from rdkit import RDLogger, Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Descriptors import ExactMolWt


def check_symbol(mol, element_mode):
    ''' element_mode: strict, common
    '''

    if element_mode == 'strict':
        common_symbols = ['C', 'H', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I']
    elif element_mode == 'common':
        common_symbols = ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si']
    else:
        common_symbols = ['C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si']

    all_symbols = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in all_symbols:
            all_symbols.append(atom.GetSymbol())
    
    if 'C' not in all_symbols:
        return False
    
    others = []
    for s in all_symbols:
        if s not in common_symbols:
            others.append(s)
    
    if len(others) > 0:
        return False
    else:
        return True


def check_mw(mol, MIN=200, MAX=800):

    MW = ExactMolWt(mol)

    if MW < MIN or MW > MAX:
        return False
    else:
        return True


def check_and_generate2D(mol):

    molblock = Chem.MolToMolBlock(mol)
    mol2 = Chem.MolFromMolBlock(molblock)

    if mol2 == None:
        return False
    else:
        return molblock

def mol_info(mol):

    inchikey = Chem.MolToInchiKey(mol)
    smi_aromatic = Chem.MolToSmiles(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    smi_kekulize = Chem.MolToSmiles(mol)

    return smi_aromatic, smi_kekulize, inchikey


def pipe_check_smi(smi, min_mw, max_mw, element_mode):

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    #
    if ' ' in smi or smi == '':
        return (False, 'Failed because SMILES is empty or space in it')
    
    # step one: check by Chem.MolFromSmiles
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None: pass
        else: return (False, 'Failed in Chem.MolFromSmiles')
    except:
        return (False, 'Failed in Chem.MolFromSmiles')
    
    # step two: remove frag
    try:
        remover = rdMolStandardize.FragmentRemover()
        mol = remover.remove(mol)
        mol = rdMolStandardize.FragmentParent(mol)
    except:
        return (False, 'Failed in remove fragment')

    # step three: check symbol
    if check_symbol(mol, element_mode) == False:
        return (False, 'Failed because unexpected symbols')
    
    # step four: check MW
    if check_mw(mol, min_mw, max_mw) == False:
        return (False, 'Failed because unsuitable MW, only MW in range [{}, {}] will be remained'.format(min_mw, max_mw))
    
    # step five: try to generate molblock
    molblock = check_and_generate2D(mol)
    if molblock == False:
        return (False, 'Failed in Chem.MolToMolBlock')
    
    # step six: generate InChiKey
    smi_aromatic, smi_kekulize, inchikey = mol_info(mol)
    if len(inchikey) < 27 or inchikey == None:
        return (False, 'Failed to generate InChiKey')

    # success
    return (True, [smi_aromatic, smi_kekulize, inchikey])


def main(cur_folder, in_file, out_file, logfile, smi_title, name_title, min_mw, max_mw, element_mode):

    # change working folder
    os.chdir(cur_folder)
    
    # load names and smiles
    df_input = pd.read_csv(in_file)
    names_ls = df_input.loc[:, name_title].tolist()
    smiles_ls = df_input.loc[:, smi_title].tolist()
    
    # check
    success_id, aro_smiles, keku_smiles, inchikeys = [], [], [], []
    failed_id, failed_names, failed_reason = [], [], []
    
    for i, smi in enumerate(smiles_ls):
        checked, result = pipe_check_smi(smi, min_mw, max_mw, element_mode)

        if checked == False:
            failed_id.append(i)
            failed_names.append(names_ls[i])
            failed_reason.append(result)
        else:
            success_id.append(i)
            aro_smiles.append(result[0])
            keku_smiles.append(result[1])
            inchikeys.append(result[2])

    dict_out = {
        'Aromatic_SMILES': aro_smiles,
        'Kekulized_SMILES': keku_smiles,
        'InChiKey': inchikeys,
    }

    # change original columns if existed
    for col in ['Aromatic_SMILES', 'Kekulized_SMILES', 'InChiKey']:
        if col in df_input.columns:
            df_input.rename(columns={col: '{}_old'.format(col)})

    # concat result
    df_out = pd.concat([df_input.loc[success_id].reset_index(drop=True),
                        pd.DataFrame(dict_out)], axis=1)
    
    # save result and log
    df_out.to_csv(out_file, index=None)

    # save failed reason
    with open(logfile, 'a') as f:
        f.write('Check SMILES for {}:\n\n'.format(in_file))

        if len(failed_id) > 0:
            for i, idx in enumerate(failed_id):
                f.write('{}, {}, {}\n'.format(failed_names[i], smiles_ls[idx], failed_reason[i]))
        else:
            f.write('All SMILES passed.\n')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Check SMILES from CSV file')

    parser.add_argument('--cur_folder', type=str, default='folder/', help='SHOULD end with /')
    parser.add_argument('--in_file', type=str, default='input.csv', help='input.csv')
    parser.add_argument('--out_file', type=str, default='checked.csv', help='checked.csv')
    parser.add_argument('--logfile', type=str, default='checked.log', help='checked.log')
    parser.add_argument('--smi_title', type=str, default='SMILES', help='')
    parser.add_argument('--name_title', type=str, default='Name', help='')
    parser.add_argument('--min_mw', type=float, default=200, help='')
    parser.add_argument('--max_mw', type=float, default=800, help='')
    parser.add_argument('--element_mode', type=str, default='common', help='strict or common')


    # unpack args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    in_file = args.in_file
    out_file = args.out_file
    logfile = args.logfile
    smi_title = args.smi_title
    name_title = args.name_title
    min_mw = args.min_mw
    max_mw = args.max_mw
    element_mode = args.element_mode

    # execute
    main(cur_folder, in_file, out_file, logfile, smi_title, name_title, min_mw, max_mw, element_mode)
