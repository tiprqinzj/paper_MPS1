import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':

    # load loss data
    loss_file = 'loss.log'
    df_loss = pd.read_table(loss_file)
    loss_ls = df_loss.loc[:, 'Loss'].tolist()

    # load valid and unique data
    valid_unique_file = 'unique.log'
    df_unique = pd.read_csv(valid_unique_file)
    valid_ls = df_unique.loc[:, 'Valid'].tolist()
    unique_ls = df_unique.loc[:, 'Unique'].tolist()
    
    # matplotlib
    fig, ax = plt.subplots()

    x_step = list(range(1, 331))
    x_epoch = list(range(22, 331, 22))

    ax1 = ax
    ax2 = ax.twinx()

    line_loss = ax1.plot(x_step, loss_ls, 'k-', label='loss')
    line_unique = ax2.plot(x_epoch, unique_ls, 'r-', label='unique')
    line_valid = ax2.plot(x_epoch, valid_ls, 'b-', label='valid')

    ax1.set_ylim([0, 50])
    ax2.set_ylim([0, 100])

    # first param: which x need a keduxian
    # second params: each keduxian, show text
    ax.set_xticks(list(range(0, 331, 22)), list(range(0, 16, 1)))

    ax.set_xlabel('Epoch')

    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Valid / Unique SMILES (%)')

    # legend
    lines_all = line_loss + line_unique + line_valid
    labels_all = [l.get_label() for l in lines_all]
    ax.legend(lines_all, labels_all, loc='right') 

    plt.savefig('plot_loss.png', dpi=300, bbox_inches='tight')



