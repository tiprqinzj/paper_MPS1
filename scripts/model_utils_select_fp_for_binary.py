import json, os, sys
import numpy as np
import pandas as pd

def save_json(d, file):
    with open(file, 'w') as f:
        f.write(json.dumps(d, sort_keys=False, indent=4, separators=(',', ': ')))

def cal_entropy(X):
    '''Calculate information entropy of 1D array X, i.e., H(X)
    '''
    unique_ls = set([X[i] for i in range(X.shape[0])])
    
    ent = 0
    for value in unique_ls:
        p = X[X == value].shape[0] / X.shape[0]
        logp = np.log2(p)
        ent -= p  * logp
    return ent

def cal_conditional_entropy(X, Y):
    '''Calculate conditional entropy of given 1D arrays X and Y, i.e., H(Y|X)
    '''
    uniqueX_ls = set([X[i] for i in range(X.shape[0])])
    
    cond_ent = 0
    for value in uniqueX_ls:
        subset_Y = Y[X == value]
        p = subset_Y.shape[0] / Y.shape[0]
        cond_ent += p * cal_entropy(subset_Y)
    return cond_ent

def cal_infogain(X, Y):
    '''Calculate information gain of X and Y
    usually used for cal_infogain(fp, label)
    '''
    result = cal_entropy(Y) - cal_conditional_entropy(X, Y)
    return result

def IGfilter(model_df, trte_title, label_title, des_ls, keep_bits):
    '''
    Example:
    >>> select_des = IGfilter(model_df, 'trte', 'target', des_dict['ECFP4-1024_all'], keep_bits=128)
    '''
    trte_ar = model_df.loc[:, trte_title].values
    tr_X = model_df.loc[trte_ar == 'train', des_ls].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values
    
    ig_ar = np.zeros(shape=len(des_ls))
    for i in range(len(des_ls)):
        ig_ar[i] = cal_infogain(tr_X[:, i], tr_Y)
    
    keep_id = np.argsort(-ig_ar)[:keep_bits]
    select_des = list(np.array(des_ls)[keep_id])

    return select_des


def main(in_info_csv, in_feats_csv, in_failed_id,
         out_json, trte_title, target_title, feat_start, feat_end, keepnum_str):
    '''
    keepnum_str: string for comma split, e.g., keepnumstr = "256,128,64"
    '''

    # unpacked keepnum
    keepnum_ls = [int(n) for n in keepnum_str.split(',')]

    # load input features file
    df_info = pd.read_csv(in_info_csv)
    df_des = pd.read_csv(in_feats_csv)

    # load failed id
    if in_failed_id != 'N':

        # failed id
        with open(in_failed_id) as f:
            failed_ls = [int(line.strip()) for line in f]
        
        # drop failed rows
        df_info = df_info.drop(index=failed_ls).reset_index(drop=True)

    # combine
    df_init = pd.concat([df_info, df_des], axis=1)

    # empty result dict
    dict_feats = {}

    # extract features
    df_feats = df_init.loc[:, feat_start:feat_end]
    all_feats_ls = df_feats.columns.tolist()
    dict_feats['all'] = all_feats_ls

    # feature selection by IG
    for n in keepnum_ls:
        dict_feats['top{}'.format(n)] = IGfilter(df_init, trte_title, target_title, all_feats_ls, n)
    
    # save
    save_json(dict_feats, out_json)

    print('Success.')



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Molecular features selection: molecular fingerprints with binary classifier')
    parser.add_argument('--cur_folder', type=str, help='working folder, SHOULD BE absolute path, end with /')
    parser.add_argument('--in_info_csv', type=str, help='in_info_csv')
    parser.add_argument('--in_feats_csv', type=str, help='in_feats_csv')
    parser.add_argument('--in_failed_id', type=str, default='N', help='in_failed_id')
    parser.add_argument('--out_json', type=str, help='')
    parser.add_argument('--trte_title', type=str, default='trte', help='')
    parser.add_argument('--target_title', type=str, default='target_binary', help='')
    parser.add_argument('--feat_start', type=str, help='')
    parser.add_argument('--feat_end', type=str, help='')
    parser.add_argument('--keepnum_str', type=str, default='256,128', help='e.g., 256,128')
    
    # unpacked args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    in_info_csv = args.in_info_csv
    in_feats_csv = args.in_feats_csv
    in_failed_id = args.in_failed_id
    out_json = args.out_json
    trte_title = args.trte_title
    target_title = args.target_title
    feat_start = args.feat_start
    feat_end = args.feat_end
    keepnum_str = args.keepnum_str

    # change working directory
    os.chdir(cur_folder)

    # execute
    main(in_info_csv, in_feats_csv, in_failed_id,
         out_json, trte_title, target_title, feat_start, feat_end, keepnum_str)

