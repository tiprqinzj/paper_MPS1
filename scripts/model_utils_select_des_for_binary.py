import time, json, os, math, sys, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

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

def save_json(d, file):
    with open(file, 'w') as f:
        f.write(json.dumps(d, sort_keys=False, indent=4, separators=(',', ': ')))
    

def PCCfilter(model_df, trte_title, label_title, des_ls, inner_threshold=0.85):

    print('Initial descriptor number: {}'.format(len(des_ls)))
    
    trte_ar = model_df.loc[:, trte_title].values
    tr_X = model_df.loc[trte_ar == 'train', des_ls].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values
    print('training samples: {}'.format(tr_X.shape[0]))

    # scaler
    scaler = StandardScaler()
    tr_scaleX = scaler.fit_transform(tr_X)
    
    del_by_std = []

    for i, des in enumerate(des_ls):
        std = tr_scaleX[:, i].std()
        if std == 0:
            del_by_std.append(des)
    print('Total of {} descritors delete because of STD == 0'.format(len(del_by_std)))
    
    # Corr with label
    corr_ls = []

    for i, des in enumerate(des_ls):
        if des not in del_by_std:
            corr = np.abs(np.corrcoef(tr_scaleX[:, i], tr_Y)[0, 1])
            corr_ls.append(corr)

    corr_sr = pd.Series(corr_ls, index=[des for des in des_ls if des not in del_by_std])
    corr_sr = corr_sr.sort_values(ascending=False)
    
    # filter with inner-correlation
    del_by_inner = []

    for i, desi in enumerate(corr_sr.index):
        if desi in del_by_inner:
            continue
        else:
            for j in range(i+1, len(corr_sr)):
                desj = corr_sr.index[j]
                if desj in del_by_inner:
                    continue
                else:
                    indexi = des_ls.index(desi)
                    indexj = des_ls.index(desj)
                    corr = np.abs(np.corrcoef(tr_scaleX[:, indexi], tr_scaleX[:, indexj])[0, 1])
                    if corr >= inner_threshold:
                        del_by_inner.append(desj)
    print('Total of {} descritors delete because of inner-corr >= {}'.format(len(del_by_inner), inner_threshold))
    
    remain_des = []
    for des in des_ls:
        if des not in del_by_std and des not in del_by_inner:
            remain_des.append(des)

    # rank by corrlation-with-label
    remain_des = [des for des in corr_sr.index.tolist() if des in remain_des]
    print('After STD and CORR filtering, remain descritors: {}'.format(len(remain_des)))
    
    return remain_des

def RFECVfilter_RFC(model_df, trte_title, label_title, des_dict, des_title, FOLD=5):

    # extract select des
    select_des = des_dict[des_title]

    trte_ar = model_df.loc[:, trte_title].values
    tr_X = model_df.loc[trte_ar == 'train', select_des].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values
    tr_N = tr_Y.shape[0]

    scaler = StandardScaler()
    tr_X = scaler.fit_transform(tr_X)

    if tr_N >= 2000:
        n_trees = 100
    elif tr_N <= 200:
        n_trees = 10
    else:
        n_trees = int(tr_N / 20)
    
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    selector = RFECV(clf, cv=StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=42), n_jobs=8)
    
    selector.fit(tr_X, tr_Y)
    ranking_ = selector.ranking_

    des_ar = np.array(select_des)
    result_des = list(des_ar[np.where(ranking_ == 1)[0]])
    
    return result_des

def SFSrank_RFC(model_df, trte_title, label_title, des_dict, des_title, logfile, min_des_num, max_des_num, step_des_num):

    trte_ar = model_df.loc[:, trte_title].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values
    tr_N = tr_Y.shape[0]

    # base estimator
    if tr_N >= 2000:
        n_trees = 100
    elif tr_N <= 200:
        n_trees = 10
    else:
        n_trees = int(tr_N / 20)
    
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)

    des_ls = des_dict[des_title]
    print('Start SFS calculation: start {}'.format(len(des_ls)))

    with open(logfile, 'w') as f:
        f.write('Start SFS calculation\n')
        f.write('descriptor number: init {}\n\n'.format(len(des_ls)))

    t = time.time()
    scaler = StandardScaler()
    tr_X = scaler.fit_transform(model_df.loc[trte_ar == 'train', des_ls].values)

    for des_num in range(min_des_num, max_des_num + 1, step_des_num):

        if des_num >= len(des_ls):
            break

        selector = SequentialFeatureSelector(clf,
                                             n_features_to_select = des_num,
                                             direction='forward',
                                             scoring='balanced_accuracy',
                                             n_jobs=8)

        selector.fit(tr_X, tr_Y)
        select_des = np.array(des_ls)[selector.get_support()].tolist()
        des_dict['{}_SFSforward{}'.format(des_title, des_num)] = select_des

        with open(logfile, 'a') as f:
            f.write('forward {}, time {}, des = {}\n'.format(len(select_des), cal_time(t), ', '.join(select_des)))
    
    return des_dict


def SFSrank_MLP(model_df, trte_title, label_title, des_dict, des_title, logfile, min_des_num, max_des_num, step_des_num):

    trte_ar = model_df.loc[:, trte_title].values
    tr_Y = model_df.loc[trte_ar == 'train', label_title].values

    # base estimator
    clf = MLPClassifier(random_state=42)

    des_ls = des_dict[des_title]
    print('Start SFS calculation: start {}'.format(len(des_ls)))

    with open(logfile, 'w') as f:
        f.write('Start SFS calculation\n')
        f.write('descriptor number: init {}\n\n'.format(len(des_ls)))

    t = time.time()
    scaler = StandardScaler()
    tr_X = scaler.fit_transform(model_df.loc[trte_ar == 'train', des_ls].values)

    for des_num in range(min_des_num, max_des_num + 1, step_des_num):

        if des_num >= len(des_ls):
            break

        warnings.filterwarnings('ignore')

        selector = SequentialFeatureSelector(clf,
                                             n_features_to_select = des_num,
                                             direction='forward',
                                             scoring='balanced_accuracy',
                                             n_jobs=8)

        selector.fit(tr_X, tr_Y)
        select_des = np.array(des_ls)[selector.get_support()].tolist()
        des_dict['{}_SFSforward{}'.format(des_title, des_num)] = select_des

        with open(logfile, 'a') as f:
            f.write('forward {}, time {}, des = {}\n'.format(len(select_des), cal_time(t), ', '.join(select_des)))
    
    return des_dict



def main(in_info_csv, in_feats_csv, in_failed_id,
         out_json, trte_title, target_title, feat_start, feat_end, ml_method='dnn',
         pcc_inner=0.85, min_des_num=4, max_des_num=64, step_des_num=4):
    ''' ml_method: rfc, dnn
    '''
    # output filenames
    out_log = out_json[:-5] + '_SFSrank.log'

    # load input features file
    df_info = pd.read_csv(in_info_csv)

    if in_feats_csv != 'N':
        df_des = pd.read_csv(in_feats_csv)

    # load failed id
    if in_failed_id != 'N':

        # failed id
        with open(in_failed_id) as f:
            failed_ls = [int(line.strip()) for line in f]
        
        # drop failed rows
        df_info = df_info.drop(index=failed_ls).reset_index(drop=True)

    # combine
    if in_feats_csv != 'N':
        df_init = pd.concat([df_info, df_des], axis=1)
    else:
        df_init = df_info

    # empty result dict
    dict_feats = {}

    # extract features
    df_feats = df_init.loc[:, feat_start:feat_end]
    all_feats_ls = df_feats.columns.tolist()
    dict_feats['all'] = all_feats_ls

    # filter by PCC
    pcc_remain_feats_ls = PCCfilter(df_init, trte_title, target_title, all_feats_ls, pcc_inner)
    dict_feats['PCC'] = pcc_remain_feats_ls

    if ml_method == 'dnn':
        # ranking by SFS
        dict_feats = SFSrank_MLP(df_init, trte_title, target_title, dict_feats, 'PCC', out_log,
                                 min_des_num, max_des_num, step_des_num)
    elif ml_method == 'rfc':
        # RFECV
        rfecv_remain_feats_ls = RFECVfilter_RFC(df_init, trte_title, target_title, dict_feats, 'PCC')
        dict_feats['RFECV'] = rfecv_remain_feats_ls

        # rank by SFS
        dict_feats = SFSrank_RFC(df_init, trte_title, target_title, dict_feats, 'RFECV', out_log,
                                 min_des_num, max_des_num, step_des_num)
    else:
        sys.exit('Error: wrong ml_method')
    
    # save
    save_json(dict_feats, out_json)

    print('Done')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Molecular features selection: numeric features with binary classifier')
    parser.add_argument('--cur_folder', type=str, help='working folder, SHOULD BE absolute path, end with /')
    parser.add_argument('--in_info_csv', type=str, help='in_info_csv')
    parser.add_argument('--in_feats_csv', type=str, help='in_feats_csv')
    parser.add_argument('--in_failed_id', type=str, default='N', help='in_failed_id')
    parser.add_argument('--out_json', type=str, help='')
    parser.add_argument('--trte_title', type=str, default='trte', help='')
    parser.add_argument('--target_title', type=str, default='target_binary', help='')
    parser.add_argument('--feat_start', type=str, help='')
    parser.add_argument('--feat_end', type=str, help='')
    parser.add_argument('--ml_method', type=str, default='dnn', help='rfc or dnn')
    parser.add_argument('--pcc_inner', type=float, default=0.85, help='')
    parser.add_argument('--min_des_num', type=int, default=2, help='')
    parser.add_argument('--max_des_num', type=int, default=24, help='')
    parser.add_argument('--step_des_num', type=int, default=2, help='')
    
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
    ml_method = args.ml_method
    pcc_inner = args.pcc_inner
    min_des_num = args.min_des_num
    max_des_num = args.max_des_num
    step_des_num = args.step_des_num

    # change working directory
    os.chdir(cur_folder)

    # execute
    main(in_info_csv, in_feats_csv, in_failed_id,
         out_json, trte_title, target_title, feat_start, feat_end, ml_method,
         pcc_inner, min_des_num, max_des_num, step_des_num)

