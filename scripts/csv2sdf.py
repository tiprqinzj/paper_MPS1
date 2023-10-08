import os
import pandas as pd
from rdkit import Chem


def gen2Dsdf(smiles_ls, name_ls, out_sdf):

    # init
    molblock_string = ''

    for name, smi in zip(name_ls, smiles_ls):
        mol = Chem.MolFromSmiles(smi)
        molblock = Chem.MolToMolBlock(mol)
        molblock_string += '{}{}$$$$\n'.format(name, molblock)
        
    # save SDF
    with open(out_sdf, 'w') as f:
        f.write(molblock_string)


def main(in_csv, smi_title, name_title, out_sdf):
    ''' in_csv should be checked
    '''
    df_input = pd.read_csv(in_csv)
    smiles_ls = df_input.loc[:, smi_title].tolist()
    names_ls = df_input.loc[:, name_title].tolist()
    gen2Dsdf(smiles_ls, names_ls, out_sdf)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Check SMILES and save SDF and figures')

    parser.add_argument('--cur_folder', type=str, default=os.getcwd(), metavar='', help='working folder, default pwd')
    parser.add_argument('--in_csv', type=str, default='input.csv', metavar='', help='default input.csv')
    parser.add_argument('--smi_title', type=str, default='SMILES', metavar='', help='default SMILES')
    parser.add_argument('--name_title', type=str, default='Name', metavar='', help='default Name')
    parser.add_argument('--out_sdf', type=str, default='data2D.sdf', metavar='', help='default data2D.sdf')

    # unpack args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    in_csv = args.in_csv
    smi_title = args.smi_title
    name_title = args.name_title
    out_sdf = args.out_sdf

    # change working folder
    os.chdir(cur_folder)

    # execute
    main(in_csv, smi_title, name_title, out_sdf)
