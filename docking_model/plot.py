import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def roc_data(df, Y_title, score_title, greater_is_better=True):

    # drop na
    sr = df.loc[:, score_title].isna()
    df = df.loc[~sr].reset_index(drop=True)

    # extract
    score = df.loc[:, score_title].values
    Y = df.loc[:, Y_title].values

    if greater_is_better:
        pass
    else:
        score = -1 * score

    auc = roc_auc_score(Y, score)
    fpr, tpr, thresholds = roc_curve(Y, score)

    return fpr, tpr, auc


def data_for_bar(ar):

    data_ls = []
    
    for i in range(0, 6):
        data_ls.append(ar[np.where(ar == i)].shape[0])
    
    return np.array(data_ls)


if __name__ == '__main__':

    # load data
    df_init = pd.read_csv('combine_manual.csv')

    titles = ['2x9e', '4zeg', '6h3k', '6tnb', '6tnd']

    

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4 * 1, 4.8 * 1))

    colors = ['pink', 'r', 'b', 'g', 'cyan', 'k']

    for title, color in zip(titles, colors):
        fpr, tpr, auc = roc_data(df_init, 'target_binary', title, False)
        ax.plot(fpr, tpr, color, label='{}, AUC={:.3f}'.format(title, auc))

    # plot consensus
    fpr, tpr, auc = roc_data(df_init, 'target_binary', 'sum')
    ax.plot(fpr, tpr, 'k', label='consensus, AUC={:.3f}'.format(auc))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC curve')
    ax.legend(loc='lower right')

    plt.savefig('docking_score_roc.png', dpi=300, bbox_inches='tight')


    # extract data
    score = df_init.loc[:, 'sum'].values
    Y = df_init.loc[:, 'target_binary'].values

    # posi_weight = [1.0 / Y[Y == 1].shape[0]] * Y[Y == 1].shape[0]
    # nega_weight = [1.0 / Y[Y == 0].shape[0]] * Y[Y == 0].shape[0]

    # ax[1].hist([score[Y == 0], score[Y == 1]],
    #             weights=[nega_weight, posi_weight],
    #             bins=6, range=[0, 5],
    #             color=['blue', 'red'], label=['decoys', 'actives'])
    # ax[1].legend(loc='upper right')
    # ax[1].set_xlabel('consensus score')
    # ax[1].set_ylabel('number of compounds (%)')
    # ax[1].set_xticks([0, 1, 2, 3, 4, 5])
    # ax[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    # ax[1].set_title('Consensus score')

    posi_weight = 1.0 / Y[Y == 1].shape[0]
    nega_weight = 1.0 / Y[Y == 0].shape[0]
    
    data_nega, data_posi = data_for_bar(score[Y == 0]), data_for_bar(score[Y == 1])
    data_nega, data_posi = data_nega * nega_weight, data_posi * posi_weight

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4 * 1, 4.8 * 1))

    ax.bar(np.arange(0, 6) - 0.15, data_nega, width=0.3, align='center', color='blue', label='decoys')
    ax.bar(np.arange(0, 6) + 0.15, data_posi, width=0.3, align='center', color='red', label='actives')
    ax.set_xticks(np.arange(0, 6), labels=range(0, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.legend(loc='upper right')
    ax.set_xlabel('consensus score')
    ax.set_ylabel('number of compounds (%)')

    plt.savefig('docking_score_hist.png', dpi=300, bbox_inches='tight')