import joblib, json, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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


def pipeline(root_folder, folder, des_prefix, des_mode, model_class='DNN_FC0001', device='cpu', validnum=5):
    '''
    des_mode: MACCS, ECFP4, PubChemFP, SubFP, PaDEL2D, RDKit2D, custom
    model_class: DNN_FC0001, DNN_FC0002
    '''
    with open('{}/{}/model_des.txt'.format(root_folder, folder)) as f:
        model_des = [line.strip() for line in f]
    
    # input_size
    input_size = len(model_des)

    # load des
    if des_mode != 'custom':
        des_df = pd.read_csv('{}_{}.csv.gz'.format(des_prefix, des_mode))
    else:
        des_df = pd.read_csv('{}.csv'.format(des_prefix))

    # load failed ID
    if des_mode == 'MACCS' or des_mode == 'ECFP4' or des_mode == 'custom':
        failed_ls = []
    else:
        with open('{}_{}_failedID.txt'.format(des_prefix, des_mode)) as f:
            failed_ls = [int(line.strip()) for line in f]

    # te_X
    te_X = des_df.loc[:, model_des].values

    # scaler
    if des_mode in ['PaDEL2D', 'RDKit2D', 'custom']:
        scaler = joblib.load('{}/{}/build.scaler'.format(root_folder, folder))
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
        
        load_model('{}/{}/{}.model'.format(root_folder, folder, valid), model)
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


def main(consensus_json, out_csv, device, des_prefix):

    # load consensus json and unpack
    d_params = load_json(consensus_json)
    root_folder = d_params['root_folder']
    model_folders = d_params['model_folders']
    des_mode_ls = d_params['des_mode']
    model_class_ls = d_params['model_class']
    tr_avgY = d_params['tr_avgY']
    validnum = d_params['validnum']
    
    
    for folder, des_mode, model_class in zip(model_folders, des_mode_ls, model_class_ls):

        pred_ar = pipeline(root_folder, folder, des_prefix, des_mode, model_class, device, validnum)

        try: pred = np.concatenate((pred, pred_ar), axis=1)
        except: pred = pred_ar
    
    # calculate average score and conf
    score, conf = np.zeros(shape=pred.shape[0]), np.zeros(shape=pred.shape[0])

    for i in range(pred.shape[0]):
        score[i], conf[i] = cal_score_conf(pred[i], tr_avgY)

    # save
    d_out = {'Score': score, 'Conf': conf}
    pd.DataFrame(d_out).to_csv(out_csv, index=None)


if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cur_folder', type=str, help='working folder, SHOULD BE absolute path, end with /')
    parser.add_argument('--consensus_json', type=str, default='consensus.json', help='')
    parser.add_argument('--out_csv', type=str, default='out.csv', help='')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:')
    parser.add_argument('--des_prefix', type=str, default='data2D', help='')

    # unpacked args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    consensus_json = args.consensus_json
    out_csv = args.out_csv
    device = args.device
    des_prefix = args.des_prefix

    # change working folder
    os.chdir(cur_folder)

    # execute
    main(consensus_json, out_csv, device, des_prefix)
    