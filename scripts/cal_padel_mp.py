import os
import subprocess
import pandas as pd
import time, math
from multiprocessing import Pool

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


def split_sdf(sdf_file, split_n):

    # count total mols in sdf
    N = 0
    with open(sdf_file) as f:
        for line in f:
            if line == '$$$$\n':
                N += 1
    
    # split_n should be less than N
    if split_n >= N: split_n = 1

    # assign number of mols in each split
    n_each = math.ceil(N / split_n)
    prefix_ls = [sdf_file[:-4] + '_{:0>3d}'.format(i) for i in range(1, split_n + 1)]

    # split original sdf to splitted sdf
    with open(sdf_file) as fr:
        mol_count, file_tag, temp_string = 0, 0, ''

        for line in fr:
            temp_string += line

            if line == '$$$$\n':
                mol_count += 1
            
            if mol_count == n_each:
                with open(prefix_ls[file_tag] + '.sdf', 'w') as fw:
                    fw.write(temp_string)
                
                mol_count = 0
                file_tag += 1
                temp_string = ''
    
    if temp_string != '':
        with open(prefix_ls[file_tag] + '.sdf', 'w') as fw:
            fw.write(temp_string)
    
    return prefix_ls


def cal_padel_with_errors(prefix, mode):
    ''' mode: PubChemFP, SubFP, PaDEL2D
    '''

    # calculate, obtain no-ordered, with errors -> csv
    if mode == 'PaDEL2D':
        subprocess.call(['/home/cadd/workstation/jre1.8.0_381/bin/java', '-jar',
                         '/home/cadd/workstation/PaDEL-Descriptor/PaDEL-Descriptor.jar',
                         '-2d', '-threads', '1',
                         '-descriptortypes', '/home/cadd/workstation/PaDEL-Descriptor/descriptors_PaDEL2D.xml',
                         '-detectaromaticity', '-standardizenitro', '-maxruntime', '120000',
                         '-dir', prefix + '.sdf', '-file', '{}_{}_cal.csv'.format(prefix, mode)])
    
    else:
        subprocess.call(['/home/cadd/workstation/jre1.8.0_381/bin/java', '-jar',
                         '/home/cadd/workstation/PaDEL-Descriptor/PaDEL-Descriptor.jar',
                        '-fingerprints', '-threads', '1',
                        '-descriptortypes', '/home/cadd/workstation/PaDEL-Descriptor/descriptors_{}.xml'.format(mode),
                        '-detectaromaticity', '-standardizenitro', '-maxruntime', '120000',
                        '-dir', prefix + '.sdf', '-file', '{}_{}_cal.csv'.format(prefix, mode)])
    
    os.remove(prefix + '.sdf')


def pack_cal_padel_with_errors(params_ls):
    return cal_padel_with_errors(params_ls[0], params_ls[1])


def combine_and_remove(prefix_ls, mode, out_prefix):

    # combine all _cal csv
    fobj_cal = []
    for i, prefix in enumerate(prefix_ls):
        with open('{}_{}_cal.csv'.format(prefix, mode)) as fr:
            if i == 0:
                fobj_cal += fr.readlines()
            else:
                fobj_cal += fr.readlines()[1:]
        
        os.remove('{}_{}_cal.csv'.format(prefix, mode))
    
    # judge errors
    fobj_success, failed_ls = [], []
    for i, line in enumerate(fobj_cal):
        if i == 0:
            fobj_success.append(line)
        else:
            if (',,' in line) or (',\n' in line) or ('Inf' in line) or ('inf' in line):
                failed_ls.append(i-1)
                continue
            else:
                fobj_success.append(line)

    # save _noerrors
    with open('{}_{}_noerrors.csv'.format(out_prefix, mode), 'w') as fw:
        for line in fobj_success: fw.write(line)

    # save failed idx
    with open('{}_{}_failedID.txt'.format(out_prefix, mode), 'w') as fw:
        for idx in failed_ls: fw.write('{}\n'.format(idx))

    # load noerrors
    df_noerrors = pd.read_csv('{}_{}_noerrors.csv'.format(out_prefix, mode))
    os.remove('{}_{}_noerrors.csv'.format(out_prefix, mode))

    # rename columns
    if mode == 'PubChemFP':
        df_noerrors.columns = ['Name'] + ['PubChemFP_{}'.format(i) for i in range(1, 882)]
    elif mode == 'SubFP':
        df_noerrors.columns = ['Name'] + ['SubFP_{}'.format(i) for i in range(1, 308)]
    elif mode == 'PaDEL2D':
        pass
    else:
        pass

    # save
    df_noerrors.to_csv('{}_{}.csv.gz'.format(out_prefix, mode), index=None, compression='gzip')


def main(sdf_file, mode, split_n):
    ''' mode: PubChemFP, SubFP, PaDEL2D
    '''

    time_start = time.time()

    # split by sdf file to get prefix ls
    prefix_ls = split_sdf(sdf_file, split_n)

    # make params list
    params_ls = []
    for prefix in prefix_ls:
        params_ls.append((prefix, mode))

    # start calculate
    pool = Pool(processes=split_n)
    pool.map(pack_cal_padel_with_errors, params_ls)
    pool.close()
    pool.join()

    # combine and save
    out_prefix = sdf_file[:-4]
    combine_and_remove(prefix_ls, mode, out_prefix)

    print('Done. Time ({})'.format(cal_time(time_start)))



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Calculate PaDEL fingerprints/descriptors, Multi thread')

    parser.add_argument('--cur_folder', type=str, default=os.getcwd(), metavar='', help='working folder, default pwd')
    parser.add_argument('--sdf_file', type=str, default='data2D.sdf', metavar='', help='default data2D.sdf')
    parser.add_argument('--mode', type=str, metavar='', help='PubChemFP, SubFP, PaDEL2D')
    parser.add_argument('--split_n', type=int, default=1, metavar='', help='multiple subprocess, default 1')

    # unpack args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    sdf_file = args.sdf_file
    mode = args.mode
    split_n = args.split_n
    
    # working folder
    os.chdir(cur_folder)
    
    # execute
    main(sdf_file, mode, split_n)
