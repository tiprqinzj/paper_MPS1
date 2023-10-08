import os, subprocess
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def main(cur_folder, in_csv, in_sdf, out_csv, n_clusters, benchmark, keepmax):

    # judge
    df_input = pd.read_csv(in_csv)
    n_input = df_input.shape[0]

    if n_clusters > n_input:
        df_out = pd.concat([df_input, pd.Series(range(n_input), name='Cluster')], axis=1)
        df_out.to_csv(out_csv, index=None)
        return 'n_clusters > n_input, no needed to clustring'

    # calculate ECFP4, result filename sdf_ECFP4.csv.gz
    subprocess.call(['/home/cadd/workstation/miniconda3/envs/production/bin/python',
                     '/home/cadd/workstation/scripts/toolbox/cal_rdkit_mp.py',
                     '--cur_folder', cur_folder,
                     '--sdf_file', in_sdf,
                     '--mode', 'ECFP4',
                     '--split_n', '1'])

    # load ecfp4
    ecfp4_ar = pd.read_csv(in_sdf[:-4] + '_ECFP4.csv.gz')

    # dimentional by t-SNE
    tsne = TSNE(n_iter=10000, init='pca', random_state=42)
    tsne_embed_ar = tsne.fit_transform(ecfp4_ar)

    # clustering by K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tsne_embed_ar)
    
    # concat result to in_csv
    df_out = pd.concat([pd.read_csv(in_csv), pd.Series(cluster_labels, name='Cluster')], axis=1)

    # pick
    picked_idx = []
    unique_cluster_labels = sorted(set(cluster_labels))

    for cluster in unique_cluster_labels:
        
        # donot reset index
        df_cur_cluster = df_out.loc[df_out.loc[:, 'Cluster'] == cluster]

        if keepmax == 'Y':
            picked_idx.append(df_cur_cluster.index[df_cur_cluster.loc[:, benchmark].values.argmax()])
        else:
            picked_idx.append(df_cur_cluster.index[df_cur_cluster.loc[:, benchmark].values.argmin()])
    
    picked_idx.sort()

    # extract picked
    df_extract = df_out.loc[picked_idx].reset_index(drop=True)

    # save
    df_extract.to_csv(out_csv, index=None)
    

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Clustering')

    parser.add_argument('--cur_folder', type=str, help='working folder, end with /')
    parser.add_argument('--logfile', type=str, default='status.txt', help='logfile, default status.txt')
    parser.add_argument('--in_csv', type=str, default='Glide_XP_convert.csv', help='')
    parser.add_argument('--in_sdf', type=str, default='Glide_XP_convert.sdf', help='')
    parser.add_argument('--out_csv', type=str, default='Glide_XP_cluster.csv', help='')
    parser.add_argument('--n_clusters', type=int, default=1000, help='')
    parser.add_argument('--benchmark', type=str, default='r_i_docking_score', help='')
    parser.add_argument('--keepmax', type=str, default='N', help='')

    # unpack args
    args = parser.parse_args()
    cur_folder = args.cur_folder
    logfile = args.logfile
    in_csv = args.in_csv
    in_sdf = args.in_sdf
    out_csv = args.out_csv
    n_clusters = args.n_clusters
    benchmark = args.benchmark
    keepmax = args.keepmax

    # change working folder
    os.chdir(cur_folder)
    
    # execute
    main(cur_folder, in_csv, in_sdf, out_csv, n_clusters, benchmark, keepmax)
