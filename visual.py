import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.cbook as cbook
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt

sector_colors = {'Consumer Discretionary': 'blue',
 'Consumer Staples': 'green',
 'Energy': 'red',
 'Financials': 'cyan',
 'Health Care': 'magenta',
 'Industrials': 'yellow',
 'Information Technology': 'black',
 'Materials': 'purple',
 'Real Estate': 'pink',
 'Telecommunication Services': 'brown',
 'Utilities': 'orange'}

def get_prices(s_name, norm=True):

    with open(f'./data/raw/Stocks/{s_name}.us.txt', 'r') as f:
        rl = f.readlines()[1:] #Skip header

    rl = [(pd.to_datetime([l.split(',')[0]]), float(l.split(',')[1])) for l in rl]

    if (norm):
        _, vals =  zip(*rl)
        max_val = np.max(vals)
        rl = [(el[0], el[1]/max_val) for el in rl]

    return (rl)

def get_stock_sectors(symbol_idx_mapping):
    sectors_df = pd.read_csv('./data/sectors/constituents_csv.csv', index_col='Symbol')
    sectors_dict = {s:sectors_df['Sector'][s.upper()] for s in symbol_idx_mapping.keys()}
    return (sectors_dict)

def compare_n_stocks(names, prices):
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    fig, ax = plt.subplots()

    for i,n in enumerate(names):
        x,y = zip(*prices[i])
        ax.plot(x, y, label=n)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    plt.title('Compare Normalized Stocks')
    plt.legend(loc='best')

    plt.show()

def plot_loss_progress(loss_tracking, loss='MSE'):
    plt.plot(loss_tracking)
    plt.xlabel('# Epochs')
    plt.ylabel(f'Loss ({loss})')
    plt.title('Stock LSTM Embedding Loss Progress')
    plt.grid()
    plt.show()

def two_dim_pca_map(state_dict, symbol_idx_mapping, init='pca', perplexity=4, random_state=23, sectors = True):
    from sklearn.manifold import TSNE
    labels = list(symbol_idx_mapping.keys())
    if (sectors):
        sectors_dict = get_stock_sectors(symbol_idx_mapping)
    tokens = state_dict['embeds.weight']
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init=init, n_iter=2500, random_state=random_state)

    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        if (sectors):
            plt.scatter(x[i], y[i], c=sector_colors[sectors_dict[labels[i]]])
        else:
            plt.scatter(x[i], y[i])

        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    if (sectors):
        patches = [mpatches.Patch(color=sector_colors[s], label=s) for s in sector_colors.keys()]
        plt.legend(handles=patches)

    plt.show()

if __name__ == "__main__":
    import pickle

    # with open(
    #         '/Users/guygozlan/Downloads/runs/Mar03_15-11-10_ip-172-31-47-15NET_EmbeddingLstm_EPOCHS15_LR_0.001_BATCH_1024/data.pickle',
    #         'rb') as f:
    #     data = pickle.load(f)
    # from main import *
    #
    # train_data_df, test_data_df = load_data()
    # _, symbol_idx_mapping = convert_unique_idx(train_data_df, "symbol")
    # state_dict = torch.load(
    #     '/Users/guygozlan/Downloads/runs/Mar03_15-11-10_ip-172-31-47-15NET_EmbeddingLstm_EPOCHS15_LR_0.001_BATCH_1024/model.pth')

    #two_dim_pca_map(state_dict, symbol_idx_mapping, perplexity=4)


    names = ['SNPS', 'HP']
    prices = [get_prices(n) for n in names]
    compare_n_stocks(names, prices)