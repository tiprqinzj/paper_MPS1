from rdkit import Chem, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    # load reference mol
    ref_smis = [
        'CCc1cccc(c1NC(=O)c2c3c(n(n2)C)-c4c(cnc(n4)Nc5ccc(cc5OC(F)(F)F)C(=O)NC6CCN(CC6)C)CC3)CC',
        'CC1=C(C=CC(=C1)C2=C3N=C(C=C(N3N=C2)NCC4CC(C4)(C)O)OC5=CN=CC=C5)C(=O)NC6CC6',
        'CCOc1cc(ccc1Nc2ncc3cc(nc(c3n2)NCC(C)(C)C)C)c4nncn4C',
        'C[C@H](c1ccc(cc1)F)C(=O)Nc2ccc(cc2)c3ccc4nc(nn4c3)Nc5ccc(cc5OC)S(=O)(=O)C',
        'Cc1cc(ccc1C(=O)NC2CC2)c3cnc4n3nc(cc4NCCC(F)(F)F)Oc5ccc(c(c5F)F)OC',
    ]
    
    ref_mols = [Chem.MolFromSmiles(smi) for smi in ref_smis]
    ref_fps = [GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in ref_mols]

    # load sampled SMILES
    sampled_files = ['epoch{}_sampled.smi'.format(i) for i in range(1, 16)]

    # matplotlib
    fig, ax = plt.subplots()

    for epoch, file in enumerate(sampled_files):
        
        # epoch + 1
        epoch = epoch + 1

        # load smiles to list
        with open(file) as f: smiles_ls = [line.strip() for line in f]

        # smiles to mols
        mols_ls = [Chem.MolFromSmiles(smi) for smi in smiles_ls]

        # cal similarity between ref and genmols
        simi_ls = []

        for mol in mols_ls:
            fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

            # average of five ref mols
            simi = 0
            for ref_fp in ref_fps:
                simi += DataStructs.TanimotoSimilarity(ref_fp, fp)
            simi_ls.append(simi / len(ref_smis))
        
        # transfer to pd.Series for plotting
        simi_sr = pd.Series(simi_ls, name='Epoch {}'.format(epoch))
        
        # plot in given ax
        simi_sr.plot(kind='kde', ax=ax)
    
    # set labels etc
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel('Tanimoto similarity')
    ax.set_ylabel('Probability density')
    ax.legend(loc='best')
    

    plt.savefig('plot_5refs.png', dpi=300, bbox_inches='tight')



