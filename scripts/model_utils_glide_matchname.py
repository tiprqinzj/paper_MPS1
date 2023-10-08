import os
import pandas as pd

def main(dataset_csv, dataset_name_title, dataset_target_title_str, glide_csv, glide_name_title, out_match_csv,
         match_unique='Y', unique_title='docking_score', unique_keepmax='N'):
    '''
    match_unique: True (each name keep only one entry) or False (keep all entries)
    '''

    # load data
    df_dataset = pd.read_csv(dataset_csv)
    df_glide = pd.read_csv(glide_csv)
    print('Input dataset mols {}, glide mols {}'.format(df_dataset.shape[0], df_glide.shape[0]))

    # extract dataset names
    dataset_names_ls = df_dataset.loc[:, dataset_name_title].tolist()
    dataset_names_ls = [str(s) for s in dataset_names_ls]

    if match_unique == 'Y':

        # obtain glide set names (this is the final number of entries)
        glide_names_set = sorted(set(df_glide.loc[:, glide_name_title].tolist()))
        
        # obtain index in original glide dataframe
        idx_in_glide = []
        for n in glide_names_set:
            # extract all rows with the given name
            df = df_glide.loc[df_glide.loc[:, glide_name_title] == n]

            if unique_keepmax == 'Y':
                idx_in_glide.append(df.index[df.loc[:, unique_title].values.argmax()])
            else:
                idx_in_glide.append(df.index[df.loc[:, unique_title].values.argmin()])
        
        # extract new df_glide
        df_glide = df_glide.loc[idx_in_glide].reset_index(drop=True)
        print('Unique for glide, {} mols remains'.format(df_glide.shape[0]))


    # extract glide name in new df_glide
    glide_names_ls = df_glide.loc[:, glide_name_title].tolist()
    glide_names_ls = [str(s) for s in glide_names_ls]

    # match names
    idx_in_dataset = []
    for n in glide_names_ls:
        idx_in_dataset.append(dataset_names_ls.index(n))
    
    # transfer target_str to list
    dataset_target_ls = dataset_target_title_str.split(',')

    # extract index from curated file
    df_target = df_dataset.loc[idx_in_dataset].reset_index(drop=True).loc[:, dataset_target_ls]

    # concat
    df_out = pd.concat([df_glide, df_target], axis=1)

    # resort
    if match_unique == 'Y':
        if unique_keepmax == 'Y':
            df_out = df_out.sort_values(by=unique_title, ascending=False).reset_index(drop=True)
        else:
            df_out = df_out.sort_values(by=unique_title, ascending=True).reset_index(drop=True)

    # save
    df_out.to_csv(out_match_csv, index=None)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Match dataset.csv and glide.csv by Name')
    parser.add_argument('--cur_folder', type=str, help='working folder, SHOULD BE absolute path, end with /')
    parser.add_argument('--dataset_csv', type=str, help='dataset csv')
    parser.add_argument('--dataset_name_title', type=str, default='Name', help='dataset Name title')
    parser.add_argument('--dataset_target_title_str', type=str, default='target_binary', help='sep by comma')
    parser.add_argument('--glide_csv', type=str, help='glide csv')
    parser.add_argument('--glide_name_title', type=str, default='Name', help='glide Name title')
    parser.add_argument('--out_match_csv', type=str, help='output matched csv')
    parser.add_argument('--match_unique', type=str, default='Y', help='match unique or not, Y or N')
    parser.add_argument('--unique_title', type=str, default='r_i_docking_score', help='unique_title')
    parser.add_argument('--unique_keepmax', type=str, default='N', help='keepmax, Y or N')

    # unpacked args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    dataset_csv = args.dataset_csv
    dataset_name_title = args.dataset_name_title
    dataset_target_title_str = args.dataset_target_title_str
    glide_csv = args.glide_csv
    glide_name_title = args.glide_name_title
    out_match_csv = args.out_match_csv
    match_unique = args.match_unique
    unique_title = args.unique_title
    unique_keepmax = args.unique_keepmax

    # change working directory
    os.chdir(cur_folder)

    # execute
    main(dataset_csv, dataset_name_title, dataset_target_title_str, glide_csv, glide_name_title, out_match_csv,
         match_unique, unique_title, unique_keepmax)
