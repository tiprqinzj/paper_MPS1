import pandas as pd
import math, time, os

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


def main(cur_folder, in_csv, out_csv, logfile, name_title, title_string):

    time_start = time.time()
    print_every = 100

    # working folder
    os.chdir(cur_folder)

    # write log
    with open(logfile, 'a') as f:
        f.write('Start duplicate:\n\n')
    
    # load input file and extract names and smiles
    df_input = pd.read_csv(in_csv)
    inchikey_ls = df_input.loc[:, 'InChiKey'].tolist()
    
    # create result list
    unique_inchikey, unique_id, repeat_id = [], [], []
    
    # execute
    for i, inchikey in enumerate(inchikey_ls):
        if inchikey not in unique_inchikey:
            unique_inchikey.append(inchikey)
            unique_id.append(i)
        else:
            repeat_id.append(i)
            
        if (i + 1) % print_every == 0:
            with open(logfile, 'a') as f:
                f.write('  process {} / {}, unique {}, repeat {}, time {}, ...\n'\
                    .format(i+1, len(inchikey_ls), len(unique_id), len(repeat_id), cal_time(time_start)))
    
    with open(logfile, 'a') as f:
        f.write('  process {} / {}, unique {}, repeat {}, time {}, done.\n\n'\
            .format(i+1, len(inchikey_ls), len(unique_id), len(repeat_id), cal_time(time_start)))
    
    # extract unique entrys
    df_unique = df_input.loc[unique_id].reset_index(drop=True)
    df_unique = pd.concat([df_unique, pd.Series(['Unique'] * df_unique.shape[0], name='Duplicate')], axis=1)

    # unpack title_string
    if title_string == None:
        titles_ls = [name_title]
    elif title_string.strip() == '':
        titles_ls = [name_title]
    else:
        titles_ls = [name_title] + title_string.strip().split(',')

    # remark repeat entry
    for idx in repeat_id:
        
        # get repeat inchikey
        repeat_sr = df_input.iloc[idx]
        repeat_inchikey = repeat_sr.loc['InChiKey']

        # make remark string
        remark_str = '['
        for title in titles_ls:
            remark_str += '{} = {}, '.format(title, repeat_sr.loc[title])
        remark_str = remark_str[:-2] + ']'
        
        # get unique index (i.e., the index of out file)
        unique_idx = unique_inchikey.index(repeat_inchikey)
        
        if df_unique.loc[unique_idx, 'Duplicate'] == 'Unique':
            df_unique.loc[unique_idx, 'Duplicate'] = remark_str
        else:
            df_unique.loc[unique_idx, 'Duplicate'] += ', ' + remark_str
    
    # save
    df_unique.to_csv(out_csv, index=None)

    with open(logfile, 'a') as f:
        f.write('Unique file was saved at: {}\n'.format(out_csv))
        f.write('Done. Time {}\n'.format(cal_time(time_start)))

    
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Duplicate, NEED check before by check_smi_for_csv.py')

    parser.add_argument('--cur_folder', type=str, default='', help='SHOULD end with /')
    parser.add_argument('--in_csv', type=str, default='checked.csv', help='checked.csv')
    parser.add_argument('--out_csv', type=str, default='unique.csv', help='unique.csv')
    parser.add_argument('--logfile', type=str, default='unique.log', help='unique.log')
    parser.add_argument('--name_title', type=str, default='Name', help='')
    parser.add_argument('--title_string', type=str, help='')

    # unpack args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    in_csv = args.in_csv
    out_csv = args.out_csv
    logfile = args.logfile
    name_title = args.name_title
    title_string = args.title_string

    # execute
    main(cur_folder, in_csv, out_csv, logfile, name_title, title_string)

