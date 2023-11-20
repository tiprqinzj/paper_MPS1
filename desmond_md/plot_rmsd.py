'''Plotting RMSD for Desmond output file 'PL_RMSD.dat'
'''

import matplotlib.pyplot as plt

def extract_data(dat_file):

    # load all data to a list
    with open(dat_file) as f:
        lines = f.readlines()
    
    # The first title line
    # title_ls = lines[0].replace('#', '').strip().split()

    # extract lists
    frames = []
    prot_ca, prot_backbone, prot_sidechain, prot_allheavy = [], [], [], []
    lig_fitby_prot, lig_fitby_lig = [], []

    # execute
    for i, line in enumerate(lines):
        if i == 0:
            continue
        else:
            # extract data and transfer from str to float
            data_ls = line.strip().split()
            data_ls = [float(n) for n in data_ls]

            # append to result lists
            frames.append(data_ls[0])
            prot_ca.append(data_ls[1])
            prot_backbone.append(data_ls[2])
            prot_sidechain.append(data_ls[3])
            prot_allheavy.append(data_ls[4])
            lig_fitby_prot.append(data_ls[5])
            lig_fitby_lig.append(data_ls[6])
    
    return frames, prot_ca, prot_backbone, prot_sidechain, prot_allheavy, lig_fitby_prot, lig_fitby_lig


def main(dat_file, out_file, dt, minY, maxY, title_str,
         sel_prot_ca, sel_prot_backbone, sel_prot_sidechain,
         sel_prot_allheavy, sel_lig_fitby_prot, sel_lig_fitby_lig):
    
    # extract dat file to lists
    frames, prot_ca, prot_backbone, prot_sidechain, prot_allheavy, lig_fitby_prot, lig_fitby_lig = extract_data(dat_file)

    # transfer frames to time (unit ns)
    times = [t * dt for t in frames]

    # plotting
    fig, ax = plt.subplots()
    
    if sel_prot_allheavy == 'Y':
        ax.plot(times, prot_allheavy, 'k-', alpha=0.7, label='protein all heavy')
    if sel_prot_backbone == 'Y':
        ax.plot(times, prot_backbone, 'k--', alpha=0.7, label='protein backbone')
    if sel_prot_sidechain == 'Y':
        ax.plot(times, prot_sidechain, 'k-.', alpha=0.7, label='protein sidechain')
    if sel_prot_ca == 'Y':
        ax.plot(times, prot_ca, 'b-', alpha=0.7, label='protein alphaC')
    if sel_lig_fitby_prot == 'Y':
        ax.plot(times, lig_fitby_prot, 'r-', alpha=0.7, label='ligand (fit by protein)')
    if sel_lig_fitby_lig == 'Y':
        ax.plot(times, lig_fitby_lig, 'r-', alpha=0.7, label='ligand (fit by ligand)')
    
    if (minY == 'auto') and (maxY == 'auto'):
        pass
    else:
        if minY == 'auto': minY = 0
        else: minY = float(minY)
        if maxY == 'auto': maxY = 10
        else: maxY = float(maxY)
        step = 0.03 * (maxY - minY)
        ax.set_ylim([minY - step, maxY + step])
    
    
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('RMSD (angstrom)')
    ax.set_title(title_str)
    ax.legend(loc='best')

    fig.savefig(out_file, dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser(description='Plotting RMSD for Desmond output file PL_RMSD.dat')

    parser.add_argument('--cur_folder', type=str, default=os.getcwd(), help='working folder, default pwd')
    parser.add_argument('--dat_file', type=str, default='raw-data/PL_RMSD.dat', metavar='', help='default raw-data/PL_RMSD.dat')
    parser.add_argument('--out_file', type=str, default='protlig_rmsd.png', metavar='', help='default protlig_rmsd.png')
    parser.add_argument('--dt', type=float, default=0.01, metavar='', help='time step (unit ns), default 0.01')
    parser.add_argument('--minY', type=str, default='auto', metavar='', help='minY, default auto')
    parser.add_argument('--maxY', type=str, default='auto', metavar='', help='maxY, default auto')
    parser.add_argument('--title_str', type=str, default='', metavar='', help='title string, default None')
    parser.add_argument('--sel_prot_ca', type=str, default='Y', metavar='', help='default Y')
    parser.add_argument('--sel_prot_backbone', type=str, default='N', metavar='', help='default N')
    parser.add_argument('--sel_prot_sidechain', type=str, default='N', metavar='', help='default N')
    parser.add_argument('--sel_prot_allheavy', type=str, default='N', metavar='', help='default N')
    parser.add_argument('--sel_lig_fitby_prot', type=str, default='Y', metavar='', help='default Y')
    parser.add_argument('--sel_lig_fitby_lig', type=str, default='N', metavar='', help='default N')

    # unpack args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    dat_file = args.dat_file
    out_file = args.out_file
    dt = args.dt
    minY = args.minY
    maxY = args.maxY
    title_str = args.title_str
    sel_prot_ca = args.sel_prot_ca
    sel_prot_backbone = args.sel_prot_backbone
    sel_prot_sidechain = args.sel_prot_sidechain
    sel_prot_allheavy = args.sel_prot_allheavy
    sel_lig_fitby_prot = args.sel_lig_fitby_prot
    sel_lig_fitby_lig = args.sel_lig_fitby_lig

    # working dir
    os.chdir(cur_folder)

    # execute
    main(dat_file, out_file, dt, minY, maxY, title_str,
         sel_prot_ca, sel_prot_backbone, sel_prot_sidechain,
         sel_prot_allheavy, sel_lig_fitby_prot, sel_lig_fitby_lig)
