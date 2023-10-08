import time, json
import math
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

def load_json(file):
    with open(file) as f:
        d = json.load(f)
    return d


def load_model(pth, model):
    model.load_state_dict(torch.load(pth))


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


def replace_double(smi):
    smi_out = smi.replace('Cl', 'L').replace('Br', 'X').replace('Si', 'G')
    return smi_out


def check_smi(smi, voc):

    # check smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            pass
        else:
            return False
    except:
        return False

    # remove frag
    try:
        remover = rdMolStandardize.FragmentRemover()
        mol = remover.remove(mol)
        mol = rdMolStandardize.FragmentParent(mol)
    except:
        return False

    # flatten
    smi = Chem.MolToSmiles(mol, isomericSmiles=False)
    smi_out = smi

    all_chars = set()
    
    regex = '(\[[^\[\]]{1,6}\])'
    smi = replace_double(smi)
    char_ls = re.split(regex, smi)
    
    for char in char_ls:
        if char.startswith('['):
            all_chars.add(char)
        else:
            for unit in char:
                all_chars.add(unit)
    
    all_chars_ls = sorted(all_chars)

    for char in all_chars_ls:
        if char not in voc.vocab.keys():
            return False
        else:
            continue
    
    return smi_out


def tokenize(smi):
    
    regex = '(\[[^\[\]]{1,6}\])'
    smi = replace_double(smi)
    char_ls = re.split(regex, smi)
    tokenized = []

    
    for char in char_ls:

        if char.startswith('['):
            tokenized.append(char)
        else:
            tokenized += list(char)
    
    tokenized.append('EOS')
    
    return tokenized


def encode(char_ls, voc):
    result_ar = np.zeros(shape=len(char_ls), dtype=np.float32)
    
    for i, char in enumerate(char_ls):
        result_ar[i] = voc.vocab[char]
        
    return result_ar


class Read_Vocabulary(object):

    def __init__(self, json_file):

        self.vocab = load_json(json_file)
        self.reversed_vocab = {v:k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)


class MolData(Dataset):
    
    def __init__(self, smiles_ar, voc):
        self.voc = voc
        self.smiles_ar = smiles_ar
        self.max_length = self.get_max_length()
        print('Trained, Max length: {}'.format(self.max_length))
    
    def get_max_length(self):
        max_length = 0
        
        for smi in self.smiles_ar:
            tokenized = tokenize(smi)
            if len(tokenized) > max_length:
                max_length = len(tokenized)
        return max_length
    
    def __len__(self):
        return len(self.smiles_ar)
    
    def __getitem__(self, idx):
        smi = self.smiles_ar[idx]
        tokenized = tokenize(smi)
        encoded = encode(tokenized, self.voc)
        tensor = torch.from_numpy(encoded)
        
        out_tensor = torch.zeros(self.max_length)
        out_tensor[: tensor.size(0)] = tensor
        
        # return Variable(out_tensor)
        return out_tensor


class MultiGRU(nn.Module):
    
    def __init__(self, voc_size):
        super(MultiGRU, self).__init__()
        self.embedding = nn.Embedding(voc_size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.linear = nn.Linear(512, voc_size)
    
    def forward(self, x, h):
        x = self.embedding(x)
        h_out = torch.zeros(h.size())
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out
    
    def init_h(self, batch_size):
        return torch.zeros(3, batch_size, 512)


def decode(ar, voc):
    chars = []
    for i in ar:
        if i == voc.vocab['EOS']: break
        chars.append(voc.reversed_vocab[i.item()])
    smi = ''.join(chars)
    smi = smi.replace('L', 'Cl').replace('X', 'Br').replace('G', 'Si')
    return smi


def samples(model, voc, max_length, device):

    smiles_ls = []
    batch_size = 10000

    # start_token = Variable(torch.zeros(batch_size).long())
    start_token = torch.zeros(batch_size).long()
    start_token[:] = voc.vocab['GO']
    h = model.init_h(batch_size)
    x = start_token

    sequences = []
    finished = torch.zeros(batch_size).byte()

    for step in range(max_length): # max_length

        logits, h = model(x, h)
        prob = F.softmax(logits, dim=1)
        x = torch.multinomial(prob, 1).view(-1)
        sequences.append(x.view(-1, 1))
        
        x = x.data
        EOS_sampled = (x == voc.vocab['EOS']).data
        finished = torch.ge(finished + EOS_sampled, 1)
        
        if torch.prod(finished) == 1: break
        
    sequences = torch.cat(sequences, 1)

    for seq in sequences:
        smi = decode(seq, voc)

        if Chem.MolFromSmiles(smi):
            smiles_ls.append(smi)

    return smiles_ls


def check_unique(input_inchikey, unique_smiles, unique_inchikey, gen_smi_ls):

    inchikey_ls = []
    for smi in gen_smi_ls:
        # try: inchikey_ls.append(Chem.MolToInchiKey(Chem.MolFromSmiles(smi))[:14])
        try: inchikey_ls.append(Chem.MolToInchiKey(Chem.MolFromSmiles(smi)))
        except: inchikey_ls.append('')

    for smi, inchikey in zip(gen_smi_ls, inchikey_ls):

        if (inchikey not in unique_inchikey) and (inchikey not in input_inchikey) and (inchikey != ''):

            unique_smiles.append(smi)
            unique_inchikey.append(inchikey)
    
    return unique_smiles, unique_inchikey



def main(pre_model, voc_json, out_csv, log_file, pre_files, dup_title, exp_gen_num, device):

    print('Start generating ...')
    time_start = time.time()

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    voc = Read_Vocabulary(voc_json)
    model = MultiGRU(voc.vocab_size).to(device)
    load_model(pre_model, model)

    # load pre-generation files
    input_inchikey = []

    for file in pre_files:
        ls = pd.read_csv(file).loc[:, dup_title].tolist()
        input_inchikey += ls

    # generate results
    unique_smiles = []
    unique_inchikey = []

    with open(log_file, 'w') as f:
        f.write('Pre-generation files: \n\n')
        for file in pre_files:
            f.write('  {}\n'.format(file))
        f.write('\nPre-generation compds: {}\n\n'.format(len(input_inchikey)))

    print('Pre-generation compds: {}'.format(len(input_inchikey)))

    for epoch in range(1000):

        smiles_ls = samples(model, voc, 126, device)

        num_before = len(unique_smiles)
        unique_smiles, unique_inchikey = check_unique(input_inchikey, unique_smiles, unique_inchikey, smiles_ls)
        num_after = len(unique_smiles)

        with open(log_file, 'a') as f:
            f.write('processing: epoch {}, unique {}, added {}, time {}\n'\
                    .format(epoch+1, num_after, num_after - num_before, cal_time(time_start)))
        
        print('processing: epoch {}, unique {}, added {}, time {}'\
              .format(epoch+1, num_after, num_after - num_before, cal_time(time_start)))

        d = {
            'SMILES': unique_smiles,
            dup_title: unique_inchikey,
        }
        pd.DataFrame(d).to_csv(out_csv, index=None)

        if num_after >= exp_gen_num:
            break
    
        
    print('Generation Done.')



if __name__ == '__main__':

    pre_files_ls = [
        'S3a_checked_flatten.csv',
    ]
            
    main(
        pre_model = '../rnn_pretrain/epoch_4.pth',
        voc_json  = '../rnn_pretrain/voc.json',
        out_csv   = 'genmols.csv',
        log_file  = 'genmols.log',
        pre_files = pre_files_ls,
        dup_title = 'InChiKey',
        exp_gen_num = 500000,
        device    = 'cpu'
    )
