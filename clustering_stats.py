import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

# from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans

import utils

UMAP_MIN_DIST = 0
UMAP_NEIGHBORS = 10

def fashion_scatter(x, colors, text):
    # choose a color palette with seaborn.
    colors +=1
    unique = np.unique(colors)
    num_classes = len(unique)
    col_min = np.min(colors)
    col_max = np.max(colors)
    norm = matplotlib.colors.Normalize(vmin=col_min, vmax=col_max)
    # format color if out of bound
    if np.average(colors) < (col_max-col_min)/3:
        print("use log scale")
        norm = matplotlib.colors.LogNorm(vmin=col_min, vmax=col_max)
    palette = 'viridis'

    while not np.isin(np.array([0]), unique):
        colors = colors -1
        unique = np.unique(colors)

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    # no clustering -------------------------
    # sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=200, c=colors,
    #                 cmap=palette, norm=norm)
    # for clustering <<<<<<<<<<<<<<<<<<<<<
    palette = np.array(sns.color_palette("hls", num_classes))
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=200, c=palette[colors.astype(np.int)])
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # txts = []
    # for xtext, ytext, text in text:
    #     txt = ax.text(xtext, ytext, text, fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    # add the labels for each digit corresponding to the label
    txts = []

    if num_classes < 20:
        for i in range(num_classes):

            # Position of each label at median of data points.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=34)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

    return f, ax, sc, txts


if __name__ == '__main__':
    root = ''
    base_path = root+'stats_extract/20000/'
    stat_path = base_path+'0to4test_per_dataset_nmi.csv'
    info_path = root+'archives/UCRArchive_2018/dataset_info.csv'
    dataset_key = 'dataset_name'
    utils.create_directory(base_path+'clustering/')

    df_perf = pd.read_csv(stat_path, index_col=False)
    df_perf = df_perf.fillna(0)
    df_info = pd.read_csv(info_path)
    dataset_names = df_perf[dataset_key].values
    join = df_perf.join(df_info.set_index(dataset_key), on=dataset_key)

    cols = ['dilated_cnn_reconstruction_None', 'dilated_rnn_reconstruction_None',
                   'bi_gru_reconstruction_None', 'bi_lstm_reconstruction_None', 'bi_rnn_reconstruction_None',
                   'res_cnn_reconstruction_None', 'fcnn_reconstruction_None', 'mlp_reconstruction_None']
    # cols = ['dilated_cnn_reconstruction_None', 'bi_rnn_reconstruction_None', ''
    #         'res_cnn_reconstruction_None']
    df_perf = df_perf[cols]
    values = df_perf.values
    color = 'clustering'
    kmeans = KMeans(n_clusters=10, n_init=20)
    pred = kmeans.fit_predict(values)

    centroids = pd.DataFrame(kmeans.cluster_centers_)
    centroids.columns = df_perf.columns
    centroids.to_csv(base_path+'clustering/'+color+'centroids_datasets.csv')

    cluster_map = np.concatenate([[dataset_names], [pred]], axis=0)
    cluster_map = pd.DataFrame(cluster_map)
    cluster_map.to_csv(base_path+'clustering/'+color+'cluster_map.csv', index=False)
    # np.savetxt(root+'clustering/cluster_map.csv', cluster_map, delimiter=",")

    # color = 'Type'
    # # color = 'Length'
    # pred = join[color].values
    # pred = np.unique(pred, return_inverse=True)[1].tolist()
    # pred = np.array(pred)

    md = float(UMAP_MIN_DIST)
    hle = umap.UMAP(
        random_state=0,
        metric='euclidean',
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=md).fit_transform(values)

    txt = []
    # i = 101
    # txt.append((hle[i][0], hle[i][1], 'CNN >> others'))
    # i = 111
    # txt.append((hle[i][0], hle[i][1], 'All high'))
    # i = 112
    # txt.append((hle[i][0], hle[i][1], 'All low'))
    # i = 112
    # txt.append((hle[i][0], hle[i][1], 'All medium'))
    fashion_scatter(hle, pred, txt)

    plt.savefig(base_path+'clustering/umap_on_perf_'+color+'.pdf')
