import time, json
import math
import re
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


def load_json(file):
    with open(file) as f:
        d = json.load(f)
    return d

def save_model(model, pth):
    torch.save(model.state_dict(), pth)

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

class My_Loss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets):
        target_expanded = torch.zeros(inputs.size())
        target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
        loss = target_expanded * inputs
        loss = torch.sum(loss, 1)

        return loss

def trainloop(model, voc, seqs):

    criterion = My_Loss()
    
    batch_size, seq_length = seqs.size()
    
    start_token = torch.zeros(batch_size, 1).long()
    start_token[:] = voc.vocab['GO']
    
    x = torch.cat((start_token, seqs[:, :-1]), 1)
    h = model.init_h(batch_size)

    log_probs = torch.zeros(batch_size)
    
    for i in range(seq_length):
        logits, h = model(x[:, i], h)
        log_prob = F.log_softmax(logits, dim=1)
        log_probs += criterion(log_prob, seqs[:, i])

    return log_probs


def decrease_learning_rate(optimizer, decrease_by=0.01):

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


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

    batch_size = 20000
    smiles_ls = []

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
    
    out_smiles = []

    for smi in smiles_ls:
        try:
            inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))

            if len(inchikey) < 27 or inchikey == None:
                pass
            else:
                out_smiles.append(smi)

        except:
            pass

    return out_smiles



def check_unique(input_inchikey, unique_smiles, unique_inchikey, smiles_ls):

    inchikey_ls = []
    for smi in smiles_ls:
        try: inchikey_ls.append(Chem.MolToInchiKey(Chem.MolFromSmiles(smi)))
        except: inchikey_ls.append('')

    for smi, inchikey in zip(smiles_ls, inchikey_ls):

        if (inchikey not in unique_inchikey) and (inchikey not in input_inchikey) and (inchikey != ''):

            unique_smiles.append(smi)
            unique_inchikey.append(inchikey)
    
    return unique_smiles, unique_inchikey


def main(pre_model, voc_json, in_csv, smi_title, out_folder, logfile, device):

    time_start = time.time()

    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    smiles_ar = pd.read_csv(in_csv).loc[:, smi_title].values
    voc = Read_Vocabulary(voc_json)
    model = MultiGRU(voc.vocab_size).to(device)
    load_model(pre_model, model)

    input_smiles = []
    for smi in smiles_ar:
        smi_out = check_smi(smi, voc)
        if smi_out != False:
            input_smiles.append(smi_out)

    input_smiles = list(set(input_smiles))

    # dataset, dataloader, optim
    dataset = MolData(input_smiles, voc)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    all_loss_ls, record_epoch_ls, record_step_ls = [], [], []
    avg_loss = 0

    input_inchikey = []
    for smi in input_smiles:
        try: input_inchikey.append(Chem.MolToInchiKey(Chem.MolFromSmiles(smi)))
        except: pass
    
    with open(logfile, 'w') as f:
        pass

    # generate results
    for epoch in range(1, 16):

        for step, batch in enumerate(dataloader):

            seqs = batch.long()
            log_p = trainloop(model, voc, seqs)
            loss = -log_p.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # all_loss_ls.append(loss.item())
            avg_loss += loss.item()

            # record epoch and step
            all_loss_ls.append(loss.item())
            record_epoch_ls.append(epoch)
            record_step_ls.append(step + 1)
            

        avg_loss /= (step + 1)

        with open(logfile, 'a') as f:
            f.write('Epoch {}: avg_loss = {:.3f}\n'.format(epoch, avg_loss))
        
        # save loss
        with open(out_folder + 'loss.log', 'w') as f:
            f.write('Epoch\tStep\tLoss\n')

            for item_epoch, item_step, item_loss in zip(record_epoch_ls, record_step_ls, all_loss_ls):
                f.write('{}\t{}\t{:.2f}\n'.format(item_epoch, item_step, item_loss))
        
        avg_loss = 0

        decrease_learning_rate(optimizer, decrease_by=0.03)

        # sampling
        smiles_ls = samples(model, voc, 126)
        inchikey_ls = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in smiles_ls]
        
        
        # duplicate for sampling
        unique_smiles, unique_inchikey = [], []
        for smi, inchikey in zip(smiles_ls, inchikey_ls):
            if inchikey not in unique_inchikey:
                unique_smiles.append(smi)
                unique_inchikey.append(inchikey)
        

        # save sampling
        with open(out_folder + 'epoch{}_sampled.smi'.format(epoch), 'w') as f:
            for smi in unique_smiles:
                f.write(smi + '\n')



        with open(logfile, 'a') as f:
            f.write('  valid {:.2%}, unique {:.2%}, time {}\n\n'\
                .format(len(smiles_ls) / 20000, len(unique_inchikey) / 20000, cal_time(time_start)))

        save_model(model, out_folder + 'epoch_{}.pth'.format(epoch))
    
    
    print('Done.')

                

if __name__ == '__main__':


    main(
        pre_model  = '/home/cadd/paper_MPS1/rnn_pretrain/epoch5_allstep13496.pth',
        voc_json   = '/home/cadd/paper_MPS1/rnn_pretrain/voc.json',
        in_csv     = 'highly_actives.csv',
        smi_title  = 'Flatten_SMILES',
        out_folder = '/home/cadd/paper_MPS1/rnn_fine_tune/',
        logfile    = '/home/cadd/paper_MPS1/rnn_fine_tune/tune.log',
        device     = 'cpu'
    )
