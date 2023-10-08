import time
import pandas as pd
import math
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumHBA, CalcNumHBD, CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.Descriptors import MolLogP
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as image
from rdkit.Chem.MolStandardize import rdMolStandardize
import json
from sklearn.model_selection import StratifiedKFold



def _cal_distance(center, X):
    dist_ar = np.zeros(shape=X.shape[0])
    
    for i in range(X.shape[0]):
        dist = np.sqrt(np.sum((center - X[i]) ** 2))
        dist_ar[i] = dist
    return dist_ar

def trtesplit_by_TSNE(df, smiles_title, trte_title='trte', target_title='target', STEP=5):
    ''' Example: result_df, result_dict = trtesplit_by_TSNE(**kward)
    '''
    smiles_ar = df.loc[:, smiles_title].values
    targets = df.loc[:, target_title].values

    des_ar = np.zeros(shape=(smiles_ar.shape[0], 6))
    
    for i, smi in enumerate(smiles_ar):
        mol = Chem.MolFromSmiles(smi)
        des_ar[i, 0] = CalcExactMolWt(mol)
        des_ar[i, 1] = CalcNumHBA(mol)
        des_ar[i, 2] = CalcNumHBD(mol)
        des_ar[i, 3] = CalcTPSA(mol)
        des_ar[i, 4] = CalcNumRotatableBonds(mol)
        des_ar[i, 5] = MolLogP(mol)
    
    scaler = StandardScaler()
    des_ar = scaler.fit_transform(des_ar)
    
    tsne = TSNE(n_iter=10000, init='pca', random_state=42)
    tsne_ar = tsne.fit_transform(des_ar)
    
    scaler = MinMaxScaler()
    tsne_ar = scaler.fit_transform(tsne_ar)
    
    max_score = 0

    for n_cluster in range(2, 6):
        _kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        _cluster_labels = _kmeans.fit_predict(tsne_ar)
        _score = silhouette_score(tsne_ar, _cluster_labels)
        print('n_cluster = {}, Score = {:.3f}'.format(n_cluster, _score))

        if _score > max_score:
            max_score = _score
            best_cluster = n_cluster
            labels = _cluster_labels
            kmeans = _kmeans
    
    centers = kmeans.cluster_centers_
    # calculate the distance between center and samples
    result_df = pd.DataFrame()
    
    for cluster in range(best_cluster):
        for target in set(targets):
            _id = np.where((labels == cluster) & (targets == target))[0]
            _dist = _cal_distance(centers[cluster], tsne_ar[_id])
            _df = pd.DataFrame({'ID': _id, 'DIST': _dist})
            _df = _df.sort_values(by='DIST', ascending=False).reset_index(drop=True)

            # if _id.shape[0] >= 5:
            if _id.shape[0] >= STEP:
                _trte = np.zeros(shape=_id.shape[0])
                # for i in range(4, _id.shape[0], 5):
                for i in range(STEP - 1, _id.shape[0], STEP):
                    _trte[i] = 1
                # _sr = pd.Series(_trte, name=trte_title).replace(1, 'test').replace(0, 'train')
                _sr = pd.Series(_trte, name=trte_title).replace([1, 0], ['test', 'train'])
            else:
                _sr = pd.Series(['train'] * _id.shape[0], name=trte_title)
            
            _df = pd.concat([_df, _sr], axis=1)
            result_df = pd.concat([result_df, _df]).reset_index(drop=True)
    
    result_df = result_df.sort_values(by='ID', ascending=True).reset_index(drop=True)
    result_df = pd.concat([result_df,
                           pd.Series(labels, name='cluster_label'),
                           pd.Series(targets, name=target_title)], axis=1)

    return_df = pd.concat([df, result_df.loc[:, trte_title]], axis=1)

    return_dict = {
        'df': result_df,
        'tsne_ar': tsne_ar,
        'centers': centers
    }

    return return_df, return_dict


def main(in_csv, out_csv, smi_title, target_title, out_trte_title='trte', out_trtecv_title='trtecv', radio=6, valid=5):

    '''radio=6 means 5:1'''

    print('Start trte split...\n')

    # split train and test
    df = pd.read_csv(in_csv)
    df_trte, _ = trtesplit_by_TSNE(df, smi_title, out_trte_title, target_title, radio)
    
    # add original seriesNO to df
    df_trte = pd.concat([pd.Series(range(1, df_trte.shape[0] + 1), name='temp_SeriesNO'), df_trte], axis=1)

    tr_df = df_trte.loc[df_trte.loc[:, out_trte_title] == 'train'].reset_index(drop=True)
    te_df = df_trte.loc[df_trte.loc[:, out_trte_title] == 'test'].reset_index(drop=True)

    N = tr_df.shape[0]
    X = np.random.random(N)
    Y = tr_df.loc[:, target_title].values

    result_ls = [''] * N
    skf = StratifiedKFold(n_splits=valid, shuffle=True, random_state=42)

    for i, (_, index) in enumerate(skf.split(X, Y)):
        name = 'train' + str(i+1)
        for j in index:
            result_ls[j] = name
    
    result_ls += ['test'] * te_df.shape[0]
    sr = pd.Series(result_ls, name=out_trtecv_title)

    df_out = pd.concat([tr_df, te_df], axis=0).reset_index(drop=True)
    df_out = pd.concat([df_out, sr], axis=1)

    # print(df_out)

    df_out = df_out.sort_values(by='temp_SeriesNO').reset_index(drop=True)
    df_out = df_out.drop(columns=['temp_SeriesNO'])

    df_out.to_csv(out_csv, index=None)

    print('Done')




if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='train, validation, and test splitting')
    parser.add_argument('--cur_folder', type=str, help='working folder, SHOULD BE absolute path, end with /')
    parser.add_argument('--in_csv', type=str, help='in csv')
    parser.add_argument('--out_csv', type=str, help='out csv')
    parser.add_argument('--smi_title', type=str, default='SMILES', help='SMILE title')
    parser.add_argument('--target_title', type=str, default='target_binary', help='target title')
    parser.add_argument('--out_trte_title', type=str, default='trte', help='')
    parser.add_argument('--out_trtecv_title', type=str, default='trtecv', help='')
    parser.add_argument('--radio', type=int, default=6, help='6 means tr:te = 5:1')
    parser.add_argument('--valid', type=int, default=5, help='train -> trainX')


    # unpacked args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    in_csv = args.in_csv
    out_csv = args.out_csv
    smi_title = args.smi_title
    target_title = args.target_title
    out_trte_title = args.out_trte_title
    out_trtecv_title = args.out_trtecv_title
    radio = args.radio
    valid = args.valid

    # change working directory
    os.chdir(cur_folder)

    # execute
    main(in_csv, out_csv, smi_title, target_title, out_trte_title, out_trtecv_title, radio, valid)
