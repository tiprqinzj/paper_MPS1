import joblib, json, math, sys
import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import scipy.stats
from scipy.stats import spearmanr

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

def load_json(file):
    with open(file) as f:
        d = json.load(f)
    return d

class DatasetClf(Dataset):
    
    def __init__(self, ar_X, ar_Y):
        self.X = ar_X
        self.Y = ar_Y
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).type(torch.float)
        Y = self.Y[idx]
        # Y = torch.tensor(self.Y[idx]).type(torch.float)
        return X, Y

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


def train(dataloader, model, loss_fn, optimizer, device):

    model.train()
    num_steps = len(dataloader)
    train_loss = 0
    
    for step, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y) # (logits, y_true)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= num_steps
    
    return train_loss


def test(dataloader, model, loss_fn, device):

    num_steps = len(dataloader)
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            pred_proba = nn.Softmax(dim=1)(pred)
            score = pred_proba.cpu().numpy()[:, 1]

            # y_pred = np.zeros(shape=score.shape[0])
            # y_pred[score >= avgY] = 1

            try:
                result_ar = np.append(result_ar, score)
            except:
                result_ar = score

    test_loss /= num_steps

    return test_loss, result_ar


def predict(dataloader, model, device):

    model.eval()
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            pred_proba = nn.Softmax(dim=1)(pred)
            score = pred_proba.cpu().numpy()[:, 1]

            try:
                result_ar = np.append(result_ar, score)
            except:
                result_ar = score
    
    return result_ar

def decrease_learning_rate(optimizer, decrease_by):

    for param_group in optimizer.param_groups:
        
        if param_group['lr'] > 0.0005:
            param_group['lr'] *= (1 - decrease_by)
        else:
            pass


def extract_XY(df, trtecv_title, valid_title, des_title, target, scale, scaler):
    
    if scale:
        tr_X = scaler.transform(df.loc[df.loc[:, trtecv_title] != valid_title, des_title].values)
        va_X = scaler.transform(df.loc[df.loc[:, trtecv_title] == valid_title, des_title].values)
    else:
        tr_X = df.loc[df.loc[:, trtecv_title] != valid_title, des_title].values
        va_X = df.loc[df.loc[:, trtecv_title] == valid_title, des_title].values
    
    tr_Y = df.loc[df.loc[:, trtecv_title] != valid_title, target].values
    va_Y = df.loc[df.loc[:, trtecv_title] == valid_title, target].values

    return tr_X, tr_Y, va_X, va_Y


def Metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    try: se = float(tp) / float(tp + fn)
    except ZeroDivisionError: se = -1.0
    try: sp = float(tn) / float(tn + fp)
    except ZeroDivisionError: sp = -1.0

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return tp, tn, fp, fn, se, sp, acc, mcc


def cal_conf(ar, avgY):
    mean = np.mean(ar)
    std  = np.std(ar)

    if std == 0:
        conf = 0.0
    else:
        conf = min(
            scipy.stats.norm(mean, std).cdf(avgY),
            1 - scipy.stats.norm(mean, std).cdf(avgY)
        )
    
    return conf


def save_model(model, pth):
    torch.save(model.state_dict(), pth)

def load_model(pth, model):
    model.load_state_dict(torch.load(pth))


def main(in_info_csv, in_feats_csv, in_failed_id, json_file,
         des_mode, des_title, trtecv_title, target_title, prefix, validnum, device,
         modelframe, batch_size, optim, lr):
    lr_decrease_epoch = 20
    lr_decrease_by = 0.01
    epochs = 99999

    if device != 'cpu': torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    warnings.filterwarnings('ignore')

    print('Start calculation ...\n')
    time_start = time.time()

    # initialize model
    des_dict = load_json(json_file)
    model_des = des_dict[des_title]

    # create folder for model
    save_folder = '{}_{}-{}_batchsize-{}_optim-{}_lr-{}/'.format(modelframe, prefix, des_title, batch_size, optim, lr)
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    
    # load data
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
        model_df = pd.concat([df_info, df_des], axis=1)
    else:
        model_df = df_info
    # model_df = pd.read_csv(modeldf_file)
    tr_df = model_df.loc[model_df.loc[:, trtecv_title] != 'test'].reset_index(drop=True)
    te_df = model_df.loc[model_df.loc[:, trtecv_title] == 'test'].reset_index(drop=True)
    tr_avgY = tr_df.loc[:, target_title].values.mean()

    # scaler
    if des_mode == 'des':
        scaler = StandardScaler()
        scaler.fit(tr_df.loc[:, model_des].values)
        joblib.dump(scaler, save_folder + 'build.scaler')

    training_logfile = save_folder + 'training.log'

    with open(training_logfile, 'w') as f:
        f.write('Hyper-parameters:\n\n')
        f.write('  batch_size   {}\n'.format(batch_size))
        f.write('  optimizer  {}\n'.format(optim))
        f.write('  lr           {}\n'.format(lr))
        f.write('  lr_decrease_epoch {}\n'.format(lr_decrease_epoch))
        f.write('  lr_decrease_by    {}\n\n\n'.format(lr_decrease_by))
    

    clfs = []
    valid_Y = []
    valid_preY = []
    valid_score = []
    
    for valid in ['train' + str(i) for i in range(1, validnum+1)]:

        with open(training_logfile, 'a') as f:
            f.write('processing {}:\n\n'.format(valid))
        

        log_string = ''

        if des_mode == 'des':
            tr_X, tr_Y, va_X, va_Y = extract_XY(tr_df, trtecv_title, valid, model_des, target_title, True, scaler)
        else:
            tr_X, tr_Y, va_X, va_Y = extract_XY(tr_df, trtecv_title, valid, model_des, target_title, False, None)
        
        input_size = tr_X.shape[1]

        tr_dataset = DatasetClf(tr_X, tr_Y)
        va_dataset = DatasetClf(va_X, va_Y)

        
        if device == 'cpu':
            tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            va_dataloader = DataLoader(va_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        else:
            tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
            va_dataloader = DataLoader(va_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
        
        tr_dataloader = list(tr_dataloader)
        va_dataloader = list(va_dataloader)

        # initialize model, loss_fn, and optimizer
        if modelframe == 'DNN_FC0001':
            model = DNN_FC0001(input_size).to(device)
        elif modelframe == 'DNN_FC0002':
            model = DNN_FC0002(input_size).to(device)
        else:
            sys.exit('Error modelframe')

        loss_fn = nn.CrossEntropyLoss()

        if optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            return None

        tr_loss_ls, va_loss_ls = [], []

        for t in range(epochs):
            
            tr_loss = train(tr_dataloader, model, loss_fn, optimizer, device)
            va_loss, va_score = test(va_dataloader, model, loss_fn, device)

            # print(va_Y)
            # print(va_score)

            tr_loss_ls.append(tr_loss)
            va_loss_ls.append(va_loss)

            va_auc = roc_auc_score(va_Y, va_score)

            with open(training_logfile, 'a') as f:

                f.write('Epoch {}, train loss = {:.3f}, valid loss = {:.3f}, time {}\n'\
                    .format(t+1, tr_loss, va_loss, cal_time(time_start)))
            
            if (t + 1) % lr_decrease_epoch == 0:
                decrease_learning_rate(optimizer, lr_decrease_by)
            
            if (t + 1) % 20 == 0:

                sr = spearmanr(list(range(20)), va_loss_ls[-20:])[0]

                log_string += 'Epoch {}, train loss = {:.3f}, valid loss = {:.3f}, Spearmanr = {:.2f}\n'\
                    .format(t+1, tr_loss, va_loss, sr)
                
                if sr <= 0 or t == 19:
                    save_model(model, save_folder + '{}.model'.format(valid))
                else:
                    if modelframe == 'DNN_FC0001':
                        model_final = DNN_FC0001(input_size).to(device)
                    elif modelframe == 'DNN_FC0002':
                        model_final = DNN_FC0002(input_size).to(device)
                    else:
                        sys.exit('Error modelframe')

                    # model_final = DNN_FC0001(input_size).to(device)
                    load_model(save_folder + '{}.model'.format(valid), model_final)
                    epoch_final = t + 1 - 20
                    break
        

        # save result
        va_loss, va_score = test(va_dataloader, model_final, loss_fn, device)
        va_preY = np.zeros(shape=va_score.shape[0])
        va_preY[va_score >= tr_avgY] = 1
        clfs.append(model_final)
        valid_Y = np.append(valid_Y, va_Y)
        valid_score = np.append(valid_score, va_score)
        valid_preY = np.append(valid_preY, va_preY)
        va_tp, va_tn, va_fp, va_fn, va_se, va_sp, va_acc, va_mcc = Metrics(va_Y, va_preY)

        with open(training_logfile, 'a') as f:

            f.write('\nSelect best Epoch:\n')
            f.write(log_string)
            f.write('Final Epoch: {}\n\n'.format(epoch_final))

            f.write('valid for {}:\n'.format(valid))
            f.write('  tp={}, tn={}, fp={}, fn={}\n'.format(va_tp, va_tn, va_fp, va_fn))
            f.write('  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n\n\n'\
                    .format(va_se, va_sp, va_acc, va_mcc, va_auc))
    
    # validation set
    valid_auc = roc_auc_score(valid_Y, valid_score)
    valid_tp, valid_tn, valid_fp, valid_fn, valid_se, valid_sp, valid_acc, valid_mcc = Metrics(valid_Y, valid_preY)

     
    # training & test set
    if des_mode == 'des':
        tr_X, tr_Y = scaler.transform(tr_df.loc[:, model_des].values), tr_df.loc[:, target_title].values
        te_X, te_Y = scaler.transform(te_df.loc[:, model_des].values), te_df.loc[:, target_title].values
    else:
        tr_X, tr_Y = tr_df.loc[:, model_des].values, tr_df.loc[:, target_title].values
        te_X, te_Y = te_df.loc[:, model_des].values, te_df.loc[:, target_title].values

    tr_dataset = DatasetClf(tr_X, tr_Y)
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    tr_dataloader = list(tr_dataloader)

    te_dataset = DatasetClf(te_X, te_Y)
    te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    te_dataloader = list(te_dataloader)

    tr_pred, te_pred = np.zeros(shape=(tr_Y.shape[0], validnum)), np.zeros(shape=(te_Y.shape[0], validnum))
    tr_preY, te_preY = np.zeros(shape=tr_Y.shape[0]), np.zeros(shape=te_Y.shape[0])

    for i, model in enumerate(clfs):
        score = predict(tr_dataloader, model, device)
        tr_pred[:, i] = score

        score = predict(te_dataloader, model, device)
        te_pred[:, i] = score
    
    tr_score, te_score = tr_pred.mean(axis=1), te_pred.mean(axis=1)
    tr_preY[tr_score >= tr_avgY] = 1
    te_preY[te_score >= tr_avgY] = 1
    
    tr_tp, tr_tn, tr_fp, tr_fn, tr_se, tr_sp, tr_acc, tr_mcc = Metrics(tr_Y, tr_preY)
    te_tp, te_tn, te_fp, te_fn, te_se, te_sp, te_acc, te_mcc = Metrics(te_Y, te_preY)

    tr_fpr, tr_tpr, tr_thresholds = roc_curve(tr_Y, tr_score)
    te_fpr, te_tpr, te_thresholds = roc_curve(te_Y, te_score)
    tr_auc = roc_auc_score(tr_Y, tr_score)
    te_auc = roc_auc_score(te_Y, te_score)

    tr_conf = np.array([cal_conf(tr_pred[i], tr_avgY) for i in range(tr_pred.shape[0])])
    te_conf = np.array([cal_conf(te_pred[i], tr_avgY) for i in range(te_pred.shape[0])])

    pd.concat([
        pd.Series(tr_Y, name='tr_Y'),
        pd.DataFrame(np.around(tr_pred, 3), columns=['valid{}'.format(i) for i in range(1, validnum+1)]),
        pd.Series(np.around(tr_score, 3), name='tr_score'),
        pd.Series(np.around(tr_conf, 3), name='tr_conf')
    ], axis=1).to_csv(save_folder + 'tr_result.csv')

    pd.concat([
        pd.Series(te_Y, name='te_Y'),
        pd.DataFrame(np.around(te_pred, 3), columns=['valid{}'.format(i) for i in range(1, validnum+1)]),
        pd.Series(np.around(te_score, 3), name='te_score'),
        pd.Series(np.around(te_conf, 3), name='te_conf')
    ], axis=1).to_csv(save_folder + 'te_result.csv')

    with open(save_folder + 'model_des.txt', 'w') as f:
        for s in des_dict[des_title]:
            f.write(s + '\n')
    
    with open(training_logfile, 'a') as f:
        f.write('Validation set:\n')
        f.write('  tp={}, tn={}, fp={}, fn={}\n'.format(valid_tp, valid_tn, valid_fp, valid_fn))
        f.write('  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n\n\n'\
                .format(valid_se, valid_sp, valid_acc, valid_mcc, valid_auc))
        
        f.write('Training set:\n')
        f.write('  tp={}, tn={}, fp={}, fn={}\n'.format(tr_tp, tr_tn, tr_fp, tr_fn))
        f.write('  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n\n\n'\
                .format(tr_se, tr_sp, tr_acc, tr_mcc, tr_auc))
        
        f.write('Test set:\n')
        f.write('  tp={}, tn={}, fp={}, fn={}\n'.format(te_tp, te_tn, te_fp, te_fn))
        f.write('  se={:>.4f}, sp={:>.4f}, acc={:>.4f}, mcc={:>.3f}, auc={:.3f}\n\n\n'\
                .format(te_se, te_sp, te_acc, te_mcc, te_auc))
        
        f.write('Valid\t{:.2f}\t{:.3f}\t{:.3f}\n'.format(valid_acc*100, valid_mcc, valid_auc))
        f.write('Train\t{:.2f}\t{:.3f}\t{:.3f}\n'.format(tr_acc*100, tr_mcc, tr_auc))
        f.write('Test\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}\n'\
            .format(te_tp, te_tn, te_fp, te_fn, te_se*100, te_sp*100, te_acc*100, te_mcc, te_auc))
    


    # plotting
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6.4 * 3, 4.8 * 2))

    tr_posi_weight = [1.0 / tr_Y[tr_Y == 1].shape[0]] * tr_Y[tr_Y == 1].shape[0]
    tr_nega_weight = [1.0 / tr_Y[tr_Y == 0].shape[0]] * tr_Y[tr_Y == 0].shape[0]

    ax[0][0].hist([tr_score[tr_Y == 0], tr_score[tr_Y == 1]],
                   weights=[tr_nega_weight, tr_posi_weight],
                   bins=10, range=[0, 1],
                   color=['blue', 'red'], label=['negative', 'positive'])
    ax[0][0].legend(loc='upper right')
    ax[0][0].set_xlabel('modeling score')
    ax[0][0].set_ylabel('number of compounds (%)')
    ax[0][0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax[0][0].set_title('training set score')

    te_posi_weight = [1.0 / te_Y[te_Y == 1].shape[0]] * te_Y[te_Y == 1].shape[0]
    te_nega_weight = [1.0 / te_Y[te_Y == 0].shape[0]] * te_Y[te_Y == 0].shape[0]

    ax[0][1].hist([te_score[te_Y == 0], te_score[te_Y == 1]],
                   weights=[te_nega_weight, te_posi_weight],
                   bins=10, range=[0, 1],
                   color=['blue', 'red'], label=['negative', 'positive'])
    ax[0][1].legend(loc='upper right')
    ax[0][1].set_xlabel('modeling score')
    ax[0][1].set_ylabel('number of compounds')
    ax[0][1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax[0][1].set_title('test set score')

    ## ROC Curve
    ax[0][2].plot([0, 1], [0, 1], 'k')
    ax[0][2].plot(tr_fpr, tr_tpr, 'k', label='training, AUC={:.3f}'.format(tr_auc))
    ax[0][2].plot(te_fpr, te_tpr, 'r--', label='test, AUC={:.3f}'.format(te_auc))
    ax[0][2].set_xlabel('FPR')
    ax[0][2].set_ylabel('TPR')
    ax[0][2].set_title('ROC curve')
    ax[0][2].legend(loc='lower right')

    ## Confidence
    ax[1][0].scatter(tr_conf[tr_Y == 0], tr_score[tr_Y == 0], marker='x', color='b', alpha=0.8, label='negative')
    ax[1][0].scatter(tr_conf[tr_Y == 1], tr_score[tr_Y == 1], marker='x', color='r', alpha=0.7, label='positive')
    ax[1][0].set_ylim(-0.03, 1.03)
    ax[1][0].set_xlabel('Confidence (i.e., distance to model)')
    ax[1][0].set_ylabel('Score')
    ax[1][0].set_title('Training confidence: train avgY = {:.3f}'.format(tr_avgY))
    ax[1][0].legend(loc='best')

    ax[1][1].scatter(te_conf[te_Y == 0], te_score[te_Y == 0], marker='x', color='b', alpha=0.8, label='negative')
    ax[1][1].scatter(te_conf[te_Y == 1], te_score[te_Y == 1], marker='x', color='r', alpha=0.7, label='positive')
    ax[1][1].set_ylim(-0.03, 1.03)
    ax[1][1].set_xlabel('Confidence (i.e., distance to model)')
    ax[1][1].set_ylabel('Score')
    ax[1][1].set_title('Test confidence')
    ax[1][1].legend(loc='best')

    ax[1][2].set_axis_off()

    plt.savefig(save_folder + 'build.png', dpi=600, bbox_inches='tight')
    

    print('All Done! Time {}\n'.format(cal_time(time_start)))





if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='DNN modeling for binary classifier')
    parser.add_argument('--cur_folder', type=str, help='working folder, SHOULD BE absolute path, end with /')
    parser.add_argument('--in_info_csv', type=str, help='in_info_csv')
    parser.add_argument('--in_feats_csv', type=str, help='in_feats_csv')
    parser.add_argument('--in_failed_id', type=str, default='N', help='in_failed_id')
    parser.add_argument('--in_des_json', type=str, help='in_des_json')
    parser.add_argument('--des_mode', type=str, help='des or fp')
    parser.add_argument('--des_title', type=str, help='des_title')
    parser.add_argument('--trtecv_title', type=str, default='trtecv', help='')
    parser.add_argument('--target_title', type=str, default='target_binary', help='')
    parser.add_argument('--valid', type=int, default=5, help='')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0-3')
    parser.add_argument('--out_prefix', type=str, default='Glide_XP_items', help='Glide_XP_items')
    parser.add_argument('--modelframe', type=str, default='DNN_FC0001', help='DNN_FC0001')
    parser.add_argument('--param_batchsize_str', type=str, default='4,8,16,32', help='4,8,16,32')
    parser.add_argument('--param_lr_str', type=str, default='0.001,0.0025,0.005', help='0.001,0.0025,0.005')
    parser.add_argument('--param_optim_str', type=str, default='sgd,adam', help='sgd,adam')

    # unpacked args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    in_info_csv = args.in_info_csv
    in_feats_csv = args.in_feats_csv
    in_failed_id = args.in_failed_id
    in_des_json = args.in_des_json
    des_mode = args.des_mode
    des_title = args.des_title
    trtecv_title = args.trtecv_title
    target_title = args.target_title
    valid = args.valid
    device = args.device
    out_prefix = args.out_prefix
    modelframe = args.modelframe
    param_batchsize_str = args.param_batchsize_str
    param_lr_str = args.param_lr_str
    param_optim_str = args.param_optim_str

    # change working folder
    os.chdir(cur_folder)

    # unpacked hyper parameters string
    params = {}
    params['batch_size'] = [int(n) for n in param_batchsize_str.split(',')]
    params['lr'] = [float(n) for n in param_lr_str.split(',')]
    params['optim'] = [n for n in param_optim_str.split(',')]

    # params = {
    #     'batch_size': [4, 8, 16, 32],
    #     'lr': [0.001, 0.0025, 0.005],
    #     'optim': ['adam', 'sgd']
    # }

    # '{}_{}-{}_batchsize-{}_optim-{}_lr-{}/'.format(model_frame, prefix, des_title, batch_size, optim, lr)
    # build model
    for batch_size in params['batch_size']:
        for lr in params['lr']:
            for optim in params['optim']:
                main(in_info_csv, in_feats_csv, in_failed_id, in_des_json,
                     des_mode, des_title, trtecv_title, target_title, out_prefix, valid, device,
                     modelframe, batch_size, optim, lr)

