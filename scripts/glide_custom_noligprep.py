import os, subprocess


def glide_docking_tier(gridfile, ligprep_file, keepnum_htvs, keepnum_sp, keepnum_xp, n_jobs):
    
    # HTVS: in file
    with open('Glide_HTVS.in', 'w') as f:
        f.write('AMIDE_MODE   penal\n')
        f.write('GRIDFILE   {}\n'.format(gridfile))
        f.write('LIGANDFILE   {}\n'.format(ligprep_file))
        f.write('NREPORT   {}\n'.format(keepnum_htvs))
        f.write('PRECISION   HTVS\n')
        f.write('WRITE_CSV   TRUE\n')
        f.write('WRITE_RES_INTERACTION   TRUE\n')
    
    # SP: in file
    with open('Glide_SP.in', 'w') as f:
        f.write('AMIDE_MODE   penal\n')
        f.write('GRIDFILE   {}\n'.format(gridfile))
        f.write('LIGANDFILE   Glide_HTVS_pv.maegz\n'.format())
        f.write('NREPORT   {}\n'.format(keepnum_sp))
        f.write('PRECISION   SP\n')
        f.write('WRITE_RES_INTERACTION   TRUE\n')
        f.write('WRITE_CSV   TRUE\n')
    
    # XP: in file
    with open('Glide_XP.in', 'w') as f:
        f.write('DOCKING_METHOD   mininplace\n')
        f.write('GRIDFILE   {}\n'.format(gridfile))
        f.write('LIGANDFILE   Glide_SP_pv.maegz\n'.format())
        f.write('POSTDOCK_XP_DELE   0.5\n')
        f.write('NREPORT   {}\n'.format(keepnum_xp))
        f.write('PRECISION   XP\n')
        f.write('WRITE_RES_INTERACTION   TRUE\n')
        f.write('WRITE_XP_DESC   TRUE\n')
        f.write('WRITE_CSV   TRUE\n')
    
    # submit HTVS
    with open('submit_glide_3tier.sh', 'w') as f:
        f.write('/opt/schrodinger2020-3/glide Glide_HTVS.in -HOST localhost:{} -NJOBS {} -WAIT\n'.format(n_jobs, n_jobs))
        f.write('/opt/schrodinger2020-3/glide Glide_SP.in -HOST localhost:{} -NJOBS {} -WAIT\n'.format(n_jobs, n_jobs))
        f.write('/opt/schrodinger2020-3/glide Glide_XP.in -HOST localhost:{} -NJOBS {} -WAIT\n'.format(n_jobs, n_jobs))

    # execute
    subprocess.call(['bash', 'submit_glide_3tier.sh'])

    # convert XP result to sdf and csv
    subprocess.call(['/opt/schrodinger2020-3/utilities/structconvert', '-n', '2:',
                      'Glide_XP_pv.maegz', 'Glide_XP_convert.sdf'])
    subprocess.call(['/opt/schrodinger2020-3/utilities/structconvert', 'Glide_XP_convert.sdf', 'Glide_XP_convert.csv'])



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Maestro three-tier Glide docking for custom databases')

    parser.add_argument('--cur_folder', type=str, help='working folder, end with /')
    parser.add_argument('--gridfile', type=str, default='glide-grid.zip', help='grid file name')
    parser.add_argument('--ligprep_file', type=str, default='ligprep.maegz', help='ligprep file')
    parser.add_argument('--keepnum_htvs', type=int, default=100000, help='Glide HTVS, keep percent')
    parser.add_argument('--keepnum_sp', type=int, default=50000, help='Glide SP, keep percent')
    parser.add_argument('--keepnum_xp', type=int, default=20000, help='Glide XP, keep percent')
    parser.add_argument('--n_jobs', type=int, default=48, help='n_jobs')


    # unpack args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    gridfile = args.gridfile
    ligprep_file = args.ligprep_file
    keepnum_htvs = args.keepnum_htvs
    keepnum_sp = args.keepnum_sp
    keepnum_xp = args.keepnum_xp
    n_jobs = args.n_jobs

    # change working directory
    os.chdir(cur_folder)
    
    # execute Glide docking
    glide_docking_tier(gridfile, ligprep_file, keepnum_htvs, keepnum_sp, keepnum_xp, n_jobs)

