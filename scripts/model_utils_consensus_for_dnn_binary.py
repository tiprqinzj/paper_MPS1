import joblib, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import scipy.stats

def load_json(file):
    with open(file) as f:
        d = json.load(f)
    return d

class DatasetForPred(Dataset):
    
    def __init__(self, ar_X):
        self.X = ar_X
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).type(torch.float)
        return X

class DNN_FC0001(nn.Module):
    
    def __init__(self, input_size):
        super(DNN_FC0001, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.25)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out) # logits
        return out

class DNN_FC0002(nn.Module):
    
    def __init__(self, input_size):
        super(DNN_FC0002, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.25)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out) # logits
        return out

def Metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    try: se = float(tp) / float(tp + fn)
    except ZeroDivisionError: se = -1.0
    try: sp = float(tn) / float(tn + fp)
    except ZeroDivisionError: sp = -1.0

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return tp, tn, fp, fn, se, sp, acc, mcc


def predict(dataloader, model, device):

    model.eval()
    
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)

            pred = model(X)
            pred_proba = nn.Softmax(dim=1)(pred)
            score = pred_proba.cpu().numpy()[:, 1]

            try:
                result_ar = np.append(result_ar, score)
            except:
                result_ar = score
    
    return result_ar


def load_model(pth, model):
    model.load_state_dict(torch.load(pth, map_location='cpu'))


def cal_score_conf(ar, avgY):
    mean = ar[~np.isnan(ar)].mean()
    std  = ar[~np.isnan(ar)].std()

    if np.isnan(mean) or np.isnan(std):
        return np.nan, np.nan
        
    else:
        if std == 0:
            conf = 0.0
        else:
            conf = min(
                scipy.stats.norm(mean, std).cdf(avgY),
                1 - scipy.stats.norm(mean, std).cdf(avgY)
            )
        
        return round(mean, 3), round(conf, 3)

def extract_id(file):
    with open(file) as f:
        failed_ls = [int(line.strip()) for line in f]
    return failed_ls


def summary_score(tr_score, tr_Y, te_score, te_Y, out_txt):

    N_posi = tr_Y[tr_Y == 1].shape[0] + te_Y[te_Y == 1].shape[0]
    N_nega = tr_Y[tr_Y == 0].shape[0] + te_Y[te_Y == 0].shape[0]
    print('Num of posi {}, num of nega {}'.format(N_posi, N_nega))

    range_left  = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                   0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    range_right = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    

    with open(out_txt, 'w') as f:
        f.write('N_posi = {}, N_nega = {}\n\n'.format(N_posi, N_nega))

        f.write('Range\tn_posi\tpercent_posi\tn_nega\tpercent_nega\tratio\n')
    

    score = np.concatenate((tr_score, te_score))
    Y = np.concatenate((tr_Y, te_Y))
    
    for l, r in zip(range_left, range_right):

        sel = np.where( (score >= l) & (score < r), True, False)

        n_posi = (Y[sel] == 1).sum()
        n_nega = (Y[sel] == 0).sum()

        n_posi = Y[sel][Y[sel] == 1].shape[0]
        n_nega = Y[sel][Y[sel] == 0].shape[0]

        percent_posi = 100 * n_posi / N_posi
        percent_nega = 100 * n_nega / N_nega

        try: ratio = percent_posi / (percent_posi + percent_nega)
        except ZeroDivisionError: ratio = -1.0

        with open(out_txt, 'a') as f:

            if r != 1.01:
                f.write('[{}, {})\t{}\t{:.2f}%\t{}\t{:.2f}%\t{:.3f}\n'\
                    .format(l, r, n_posi, percent_posi, n_nega, percent_nega, ratio))
            else:
                f.write('[{}, 1.0]\t{}\t{:.2f}%\t{}\t{:.2f}%\t{:.3f}\n'\
                    .format(l, n_posi, percent_posi, n_nega, percent_nega, ratio))

def pipeline(folder, des_prefix, des_mode, model_class='DNN_FC0001', device='cpu', validnum=5):
    '''
    des_mode: MACCS, ECFP4, PubChemFP, SubFP, PaDEL2D, RDKit2D
    model_class: DNN_FC0001, DNN_FC0002
    '''
    with open(folder + '/model_des.txt') as f:
        model_des = [line.strip() for line in f]
    
    # input_size
    input_size = len(model_des)

    # load des
    des_df = pd.read_csv('{}_{}.csv.gz'.format(des_prefix, des_mode))

    # load failed ID
    if des_mode == 'MACCS' or des_mode == 'ECFP4':
        failed_ls = []
    else:
        with open('{}_{}_failedID.txt'.format(des_prefix, des_mode)) as f:
            failed_ls = [int(line.strip()) for line in f]

    # te_X
    te_X = des_df.loc[:, model_des].values

    # scaler
    if des_mode in ['PaDEL2D', 'RDKit2D']:
        scaler = joblib.load(folder + '/build.scaler')
        te_X = scaler.transform(te_X)

    # load models
    clfs = []
    valid_ls = ['train{}'.format(i) for i in range(1, validnum+1)]
    for valid in valid_ls:
        if model_class == 'DNN_FC0001':
            model = DNN_FC0001(input_size).to(device)
        elif model_class == 'DNN_FC0002':
            model = DNN_FC0002(input_size).to(device)
        else:
            return 'Error model_class'
        
        load_model(folder + '/{}.model'.format(valid), model)
        clfs.append(model)

    # batch size
    if te_X.shape[0] >= 1024:
        batch_size = 1024
    else:
        batch_size = te_X.shape[0]

    # torch dataset & dataloader
    te_dataset = DatasetForPred(te_X)
    te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    te_dataloader = list(te_dataloader)

    # pred result, consider failed num
    te_pred = np.zeros(shape=(te_X.shape[0] + len(failed_ls), validnum))

    for i, model in enumerate(clfs):
        score = predict(te_dataloader, model, device)
        score = score.flatten()

        # insert nan
        for idx in failed_ls:
            score = np.insert(score, idx, np.nan)
        
        # save
        te_pred[:, i] = score

    return te_pred


def main(info_file, trtecv_title, target_title, consensus_json,
         logfile, figfile, summary_score_file, device, validnum, des_prefix):

    # load consensus json and unpack
    d_params = load_json(consensus_json)
    model_folders = d_params['model_folders']
    des_mode_ls = d_params['des_mode']
    model_class_ls = d_params['model_class']
    
    # load info file
    df_curated = pd.read_csv(info_file)

    # 
    trtecv_ar = df_curated.loc[:, trtecv_title].values

    # calculate train average Y
    tr_avgY = round(df_curated.loc[df_curated.loc[:, trtecv_title] != 'test', target_title].values.mean(), 3)
    print('Training average Y = {}'.format(tr_avgY))
    
    for folder, des_mode, model_class in zip(model_folders, des_mode_ls, model_class_ls):

        pred_ar = pipeline(folder, des_prefix, des_mode, model_class, device, validnum)

        try: pred = np.concatenate((pred, pred_ar), axis=1)
        except: pred = pred_ar
    
    # calculate average score and conf
    score, conf = np.zeros(shape=pred.shape[0]), np.zeros(shape=pred.shape[0])

    for i in range(pred.shape[0]):
        score[i], conf[i] = cal_score_conf(pred[i], tr_avgY)

    # split to train score and test score
    tr_score = score[trtecv_ar != 'test']
    te_score = score[trtecv_ar == 'test']
    tr_conf = conf[trtecv_ar != 'test']
    te_conf = conf[trtecv_ar == 'test']
    

    tr_Y = df_curated.loc[df_curated.loc[:, trtecv_title] != 'test', target_title].values
    te_Y = df_curated.loc[df_curated.loc[:, trtecv_title] == 'test', target_title].values


    tr_preY, te_preY = np.zeros(shape=tr_Y.shape[0]), np.zeros(shape=te_Y.shape[0])
    tr_preY[tr_score >= tr_avgY] = 1
    te_preY[te_score >= tr_avgY] = 1


    tr_auc = roc_auc_score(tr_Y, tr_score)
    te_auc = roc_auc_score(te_Y, te_score)

    tr_tp, tr_tn, tr_fp, tr_fn, tr_se, tr_sp, tr_acc, tr_mcc = Metrics(tr_Y, tr_preY)
    te_tp, te_tn, te_fp, te_fn, te_se, te_sp, te_acc, te_mcc = Metrics(te_Y, te_preY)

    tr_fpr, tr_tpr, tr_thresholds = roc_curve(tr_Y, tr_score)
    te_fpr, te_tpr, te_thresholds = roc_curve(te_Y, te_score)

    print("training set:")
    print("  tp={}, tn={}, fp={}, fn={}".format(tr_tp, tr_tn, tr_fp, tr_fn))
    print("  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n"\
        .format(tr_se, tr_sp, tr_acc, tr_mcc, tr_auc))

    print("test set:")
    print("  tp={}, tn={}, fp={}, fn={}".format(te_tp, te_tn, te_fp, te_fn))
    print("  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n"\
        .format(te_se, te_sp, te_acc, te_mcc, te_auc))
    

    # write logfile

    log_string = '{}\n\n'.format('\n'.join(model_folders))
    log_string += 'train\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\n'\
        .format(tr_tp, tr_tn, tr_fp, tr_fn, tr_se*100, tr_sp*100, tr_acc*100, tr_mcc, tr_auc)
    log_string += 'test\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\n'\
        .format(te_tp, te_tn, te_fp, te_fn, te_se*100, te_sp*100, te_acc*100, te_mcc, te_auc)
    
    with open(logfile, 'w') as f:
        f.write(log_string)
    
    # plotting
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(6.4 * 5, 4.8 * 1))

    tr_posi_weight = [1.0 / tr_Y[tr_Y == 1].shape[0]] * tr_Y[tr_Y == 1].shape[0]
    tr_nega_weight = [1.0 / tr_Y[tr_Y == 0].shape[0]] * tr_Y[tr_Y == 0].shape[0]

    ax[0].hist([tr_score[tr_Y == 0], tr_score[tr_Y == 1]],
                   weights=[tr_nega_weight, tr_posi_weight],
                   bins=10, range=[0, 1],
                   color=['blue', 'red'], label=['negative', 'positive'])
    ax[0].legend(loc='upper right')
    ax[0].set_xlabel('modeling score')
    ax[0].set_ylabel('number of compounds (%)')
    ax[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax[0].set_title('Training score')

    te_posi_weight = [1.0 / te_Y[te_Y == 1].shape[0]] * te_Y[te_Y == 1].shape[0]
    te_nega_weight = [1.0 / te_Y[te_Y == 0].shape[0]] * te_Y[te_Y == 0].shape[0]

    ax[1].hist([te_score[te_Y == 0], te_score[te_Y == 1]],
                   weights=[te_nega_weight, te_posi_weight],
                   bins=10, range=[0, 1],
                   color=['blue', 'red'], label=['negative', 'positive'])
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel('modeling score')
    ax[1].set_ylabel('number of compounds')
    ax[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax[1].set_title('Test score')

    ## ROC Curve
    ax[2].plot([0, 1], [0, 1], 'k')
    ax[2].plot(tr_fpr, tr_tpr, 'k', label='training, AUC={:.3f}'.format(tr_auc))
    ax[2].plot(te_fpr, te_tpr, 'r--', label='test, AUC={:.3f}'.format(te_auc))
    ax[2].set_xlabel('FPR')
    ax[2].set_ylabel('TPR')
    ax[2].set_title('ROC curve')
    ax[2].legend(loc='lower right')

    ## Confidence
    ax[3].scatter(tr_conf[np.where((tr_Y == 0)&(tr_score < tr_avgY))], tr_score[np.where((tr_Y == 0)&(tr_score < tr_avgY))],
                     marker='x', color='b', alpha=0.7, label='negative')
    ax[3].scatter(tr_conf[np.where((tr_Y == 1)&(tr_score >= tr_avgY))], tr_score[np.where((tr_Y == 1)&(tr_score >= tr_avgY))],
                     marker='x', color='r', alpha=0.8, label='positive')
    ax[3].scatter(tr_conf[np.where((tr_Y == 1)&(tr_score < tr_avgY))], tr_score[np.where((tr_Y == 1)&(tr_score < tr_avgY))],
                     marker='x', color='r', alpha=0.8)
    ax[3].scatter(tr_conf[np.where((tr_Y == 0)&(tr_score >= tr_avgY))], tr_score[np.where((tr_Y == 0)&(tr_score >= tr_avgY))],
                     marker='x', color='b', alpha=0.7)

    ax[3].set_ylim(-0.03, 1.03)
    ax[3].set_xlabel('d$_{STD-PTO}$')
    ax[3].set_ylabel('modeling score')
    ax[3].set_title('Training confidence: train avgY = {:.3f}'.format(tr_avgY))
    ax[3].legend(loc='best')

    ax[4].scatter(te_conf[np.where((te_Y == 0)&(te_score < tr_avgY))], te_score[np.where((te_Y == 0)&(te_score < tr_avgY))],
                  marker='x', color='b', alpha=0.7, label='negative')
    ax[4].scatter(te_conf[np.where((te_Y == 1)&(te_score >= tr_avgY))], te_score[np.where((te_Y == 1)&(te_score >= tr_avgY))],
                  marker='x', color='r', alpha=0.8, label='positive')
    ax[4].scatter(te_conf[np.where((te_Y == 1)&(te_score < tr_avgY))], te_score[np.where((te_Y == 1)&(te_score < tr_avgY))],
                  marker='x', color='r', alpha=0.8)
    ax[4].scatter(te_conf[np.where((te_Y == 0)&(te_score >= tr_avgY))], te_score[np.where((te_Y == 0)&(te_score >= tr_avgY))],
                     marker='x', color='b', alpha=0.7)
    ax[4].set_ylim(-0.03, 1.03)
    ax[4].set_xlabel('d$_{STD-PTO}$')
    ax[4].set_ylabel('modeling score')
    ax[4].set_title('Test confidence')
    ax[4].legend(loc='best')
    
    plt.savefig(figfile, dpi=600, bbox_inches='tight')


    # summary_score
    summary_score(tr_score, tr_Y, te_score, te_Y, summary_score_file)


if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cur_folder', type=str, help='working folder, SHOULD BE absolute path, end with /')
    parser.add_argument('--info_file', type=str, help='in_info_csv')
    parser.add_argument('--trtecv_title', type=str, default='trtecv', help='')
    parser.add_argument('--target_title', type=str, default='target_binary', help='')
    parser.add_argument('--consensus_json', type=str, default='consensus.json', help='')
    parser.add_argument('--logfile', type=str, default='consensus.log', help='')
    parser.add_argument('--figfile', type=str, default='consensus.png', help='')
    parser.add_argument('--summary_score_file', type=str, default='consensus_score.txt', help='')
    parser.add_argument('--validnum', type=int, default=5, help='')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:')
    parser.add_argument('--des_prefix', type=str, default='data2D', help='')


    # unpacked args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    info_file = args.info_file
    trtecv_title = args.trtecv_title
    target_title = args.target_title

    consensus_json = args.consensus_json
    logfile = args.logfile
    figfile = args.figfile

    summary_score_file = args.summary_score_file
    validnum = args.validnum
    device = args.device

    des_prefix = args.des_prefix



    main(info_file, trtecv_title, target_title, consensus_json,
         logfile, figfile, summary_score_file, device, validnum, des_prefix)
    