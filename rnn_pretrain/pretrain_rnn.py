import sys

if sys.platform == 'linux':
    sys.path.append('/home/cadd/Project/tipr_coding/')
if sys.platform == 'darwin':
    sys.path.append('/Users/qinzijian/Experiment/Project/tipr_coding_macbook/')

import time
import math
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger

from utils import save_json


def Variable(tensor):
    '''Warpper for torch.autograd.Variable'''
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)

    
def save_model(model, pth):
    torch.save(model.state_dict(), pth)


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


# def load_model(pth, model):
#     model.load_state_dict(torch.load(pth))

def replace_double(smi):
    smi_out = smi.replace('Cl', 'L').replace('Br', 'X').replace('Si', 'G')
    return smi_out


def get_all_chars(smiles_ar):
    
    all_chars = set()
    
    for smi in smiles_ar:
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
    
    return all_chars_ls


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


class Vocabulary(object):
    '''A class for handling encoding/decoding from SMILES to an array of indices'''
    
    def __init__(self, smiles_ar):
        self.all_chars = get_all_chars(smiles_ar) + ['EOS', 'GO']
        
        self.vocab_size = len(self.all_chars)
        print('Vocabulary size: {}'.format(self.vocab_size))
        
        self.vocab = dict(zip(self.all_chars, range(self.vocab_size)))
        self.reversed_vocab = {v:k for k, v in self.vocab.items()}


class MolData(Dataset):
    
    def __init__(self, smiles_ar, voc):
        self.voc = voc
        self.smiles_ar = smiles_ar
        self.max_length = self.get_max_length()
        print('Max length: {}'.format(self.max_length))
    
    def get_max_length(self):
        max_length = 0
        
        for smi in self.smiles_ar:
            tokenized = tokenize(smi)
            if len(tokenized) > max_length:
                max_length = len(tokenized)
        return max_length
    
    def __len__(self):
        return self.smiles_ar.shape[0]
    
    def __getitem__(self, idx):
        smi = self.smiles_ar[idx]
        tokenized = tokenize(smi)
        encoded = encode(tokenized, self.voc)
        tensor = torch.from_numpy(encoded)
        
        out_tensor = torch.zeros(self.max_length)
        out_tensor[: tensor.size(0)] = tensor
        
        return Variable(out_tensor)


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


def NLLLoss(inputs, targets):
    target_expanded = torch.zeros(inputs.size())
    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss


def trainloop(model, voc, seqs):
    
    batch_size, seq_length = seqs.size()
    
    start_token = Variable(torch.zeros(batch_size, 1).long())
    start_token[:] = voc.vocab['GO']
    
    x = torch.cat((start_token, seqs[:, :-1]), 1)
    h = model.init_h(batch_size)
    log_probs, entropy = Variable(torch.zeros(batch_size)), Variable(torch.zeros(batch_size))
    
    for i in range(seq_length):
        logits, h = model(x[:, i], h)
        log_prob = F.log_softmax(logits, dim=1)
        prob = F.softmax(logits, dim=1)
        log_probs += NLLLoss(log_prob, seqs[:, i])
        entropy += -torch.sum((log_prob * prob), 1)
    
    return log_probs, entropy


def plotting(all_loss_ls, valid_ls, out_png):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6.4 * 2, 4.8))

    ax[0].plot([i for i in range(1, len(all_loss_ls)+1)], all_loss_ls)
    ax[0].set_xlabel('Gradient updates')
    ax[0].set_ylabel('Loss')
    ax[0].set_ylim(-5, 105)

    ax[1].plot([i * 10 for i in range(1, len(valid_ls)+1)], valid_ls)
    ax[1].set_xlabel('Gradient updates')
    ax[1].set_ylabel('Valid SMILES (%)')
    ax[1].set_ylim(-5, 105)

    fig.savefig(out_png, dpi=600, bbox_inches='tight')

    plt.close(fig)


def decode(ar, voc):
    chars = []
    for i in ar:
        if i == voc.vocab['EOS']: break
        chars.append(voc.reversed_vocab[i.item()])
    smi = ''.join(chars)
    smi = smi.replace('L', 'Cl').replace('X', 'Br').replace('G', 'Si')
    return smi

def samples(model, voc, max_length):

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    total = 1280
    smiles_ls = []

    for cycle in range(10):

        batch_size = 128

        start_token = Variable(torch.zeros(batch_size).long())
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
    
    # inchikey_ls = [Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in smiles_ls]

    inchikey_ls = []
    for smi in smiles_ls:
        try: inchikey_ls.append(Chem.MolToInchiKey(Chem.MolFromSmiles(smi)))
        except: pass

    unique = len(set(inchikey_ls))

    valid_score = len(smiles_ls) / total
    unique_score = unique / total

    return valid_score, unique_score


def decrease_learning_rate(optimizer, decrease_by=0.01):

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def main(in_csv, smi_title, save_folder):
    smiles_ar = pd.read_csv(in_csv).loc[:, smi_title].values

    voc = Vocabulary(smiles_ar)
    save_json(voc.vocab, save_folder + 'voc.json')
    dataset = MolData(smiles_ar, voc)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    model = MultiGRU(voc.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    time_start = time.time()

    all_loss_ls = []
    avg_loss = 0

    valid_ls = []
    unique_ls = []

    logfile = save_folder + 'pretrain.log'
    with open(logfile, 'w') as f:
        pass

    process = 0

    for epoch in range(1, 6):

        for step, batch in enumerate(dataloader):

            process += 1

            seqs = batch.long()
            
            log_p, _ = trainloop(model, voc, seqs)
            loss = -log_p.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            all_loss_ls.append(loss.item())
            avg_loss += loss.item()
            
            if process % 500 == 0:
                save_model(model, save_folder + 'epoch{}_step{}.pth'.format(epoch, step+1))
                decrease_learning_rate(optimizer, decrease_by=0.03)

            if process % 10 == 0:

                avg_loss /= 10
                valid_score, unique_score = samples(model, voc, 126)
                valid_ls.append(valid_score * 100)
                unique_ls.append(unique_score * 100)
                
                print('  processing {}: epoch {}, {} / {}, avg loss = {:.3f}, valid = {:.2f}%, unique = {:.2f}%, time {}'\
                    .format(process, epoch, step+1, len(dataloader), avg_loss, valid_score * 100, unique_score * 100, cal_time(time_start)))
                
                with open(logfile, 'a') as f:
                    f.write('processing {}: epoch {}, {} / {}, avg loss = {:.3f}, valid = {:.2f}%, unique = {:.2f}%, time {}\n'\
                        .format(process, epoch, step+1, len(dataloader), avg_loss, valid_score * 100, unique_score * 100, cal_time(time_start))
                    )

                avg_loss = 0

                plotting(all_loss_ls, valid_ls, save_folder + 'lossfig.png'.format(epoch, step+1))
    
        save_model(model, save_folder + 'epoch{}_allstep{}.pth'.format(epoch, step+1))

                
                


if __name__ == '__main__':

    main(
        in_csv = '/home/cadd/Project/ChEMBL/2203/ChEMBL_allcompds_unique.csv',
        smi_title = 'Flatten SMILES',
        save_folder = '/home/cadd/Project/ChEMBL/2203/RNN_0427/'
    )