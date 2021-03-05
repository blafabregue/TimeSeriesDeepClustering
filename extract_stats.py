"""
Script to extract stats into csv files

Author:
Baptiste Lafabregue 2019.25.04
"""

import os
import pathlib
import argparse

import pandas as pd
import numpy as np

import utils

ELEMENT_COUNT = 6


def get_elements(file_name):
    split = file_name.split('_')
    arch = split[0]
    limit = ELEMENT_COUNT
    shift = 0
    if arch == "dilated" or arch == "bi" or arch == 'res':
        arch += "_" + split[1]
        enc_loss = split[2]
        limit += 1
    else:
        enc_loss = split[1]
    if len(split) < limit:
        clust_loss = "None"
        shift = 1
    else:
        clust_loss = split[-2]
    dropout = split[(shift-4)]
    noise = split[(shift-3)]
    dataset_name = split[-1]

    return arch, enc_loss, clust_loss, dataset_name, dropout, noise


def main(itr, seed_itrs, type, stat_path, root_dir):
    dfs = []
    for seed in seed_itrs:
        stats = []

        stat_path = root_dir + '/stats/' + str(itr) + '/' + str(seed)

        stats_files = [name for name in os.listdir(stat_path)
                       if os.path.isfile(os.path.join(stat_path, name)) and
                       pathlib.Path(name).suffix != '.error']
        stats_files.sort()

        error_files = [name for name in os.listdir(stat_path)
                       if os.path.isfile(os.path.join(stat_path, name)) and
                       pathlib.Path(name).suffix == '.error']
        error_files.sort()

        # each element contains in the following order :
        # encoder architecture, encoder loss, clustering loss, dataset name, nmi, max nmi
        for file_name in stats_files:
            arch, enc_loss, clust_loss, dataset_name, dropout, noise = get_elements(file_name)
            try:
                with open(os.path.join(stat_path, file_name)) as f:
                    row = f.readline()
                    # the test's stats are in the second row
                    if type == 'test':
                        row = f.readline()
                    split_stats = row.split(',')
                    acc = float(split_stats[0])
                    max_acc = float(split_stats[1])
                    nmi = float(split_stats[2])
                    max_nmi = float(split_stats[3])
                    ari = float(split_stats[4])
                    max_ari = float(split_stats[5])
                    gap = max_nmi - nmi
            except:
                print('error with '+os.path.join(stat_path, file_name))
                nmi, max_nmi, gap, ari, acc = -1, -1, 0, -1, -1
            stats.append([arch, enc_loss, clust_loss, dataset_name, nmi, max_nmi, gap, ari, acc])

        for file_name in error_files:
            arch, enc_loss, clust_loss, dataset_name, dropout, noise = get_elements(file_name.split('.')[0])
            stats.append([arch, enc_loss, clust_loss, dataset_name, np.nan, np.nan, 0.0, np.nan, np.nan])

        stats = np.array(stats)
        dfs.append(pd.DataFrame({"encoder_architecture": stats[:, 0],
                                 "encoder_loss": stats[:, 1],
                                 "clustering_loss": stats[:, 2],
                                 "dataset_name": stats[:, 3],
                                 "nmi": stats[:, 4],
                                 "max_nmi": stats[:, 5],
                                 "gap": stats[:, 6],
                                 "ari": stats[:, 7],
                                 "acc": stats[:, 8]}))

    concat = pd.concat(dfs)
    arch_list = concat.encoder_architecture.unique()
    encoder_loss = concat.encoder_loss.unique()
    clustering_loss = concat.clustering_loss.unique()
    dataset_name = concat.dataset_name.unique()
    out_path = root_dir + '/stats_extract/' + str(itr) + '/' + str(seed)
    if len(seed_itrs) > 1:
        out_path = root_dir + '/stats_extract/' + str(itr) + '/' + str(seed_itrs[0]) + 'to' + str(seed_itrs[-1]) + type
    utils.create_directory(root_dir + '/stats_extract/' + str(itr) + '/')

    concat["nmi"] = concat["nmi"].astype(float)
    concat["max_nmi"] = concat["max_nmi"].astype(float)
    concat["gap"] = concat["gap"].astype(float)
    concat["acc"] = concat["acc"].astype(float)
    concat["ari"] = concat["ari"].astype(float)
    concat.to_csv(out_path + '_raw_data.csv', index=False)
    concat["model"] = concat["encoder_architecture"] + "_" + concat["encoder_loss"] + "_" + concat["clustering_loss"]
    # concat_cleaned = concat.drop(["encoder_architecture", "encoder_loss", "clustering_loss"], axis=1)
    # concat_cleaned = concat_cleaned.set_index(["model"])
    # df_nmi = df2.drop(["max_nmi"])
    df_nmi = pd.pivot_table(concat, values="nmi", index="dataset_name", columns="model", aggfunc=np.nanmean)
    df_nmi.to_csv(out_path + '_per_dataset_nmi.csv', index=True)
    df_max_nmi = pd.pivot_table(concat, values="max_nmi", index="dataset_name", columns="model", aggfunc=np.nanmean)
    df_max_nmi.to_csv(out_path + '_per_dataset_max_nmi.csv', index=True)
    df_gap = pd.pivot_table(concat, values="gap", index="dataset_name", columns="model", aggfunc=np.nanmean)
    df_gap.to_csv(out_path + '_per_dataset_gap.csv', index=True)
    df_gap = pd.pivot_table(concat, values="ari", index="dataset_name", columns="model", aggfunc=np.nanmean)
    df_gap.to_csv(out_path + '_per_dataset_ari.csv', index=True)
    df_gap = pd.pivot_table(concat, values="acc", index="dataset_name", columns="model", aggfunc=np.nanmean)
    df_gap.to_csv(out_path + '_per_dataset_acc.csv', index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract stats from logs'
    )
    parser.add_argument('--itr', type=str, metavar='X', default='50320',
                        help='iteration index')
    parser.add_argument('--seeds_itr', type=str, metavar='X', default='0,1,2,3,4',
                        help='seeds index, can be either an integer or a comma separated list, '
                             'example : --seeds_itr 0,1,2,3,4')
    parser.add_argument('--root_dir', type=str, metavar='PATH', default='.',
                        help='path of the root dir where archives and results are stored')
    parser.add_argument('--type', type=str, metavar='xxx', default='test',
                        choices=['train', 'test'],
                        help='run on either train or test results clustering')

    args = parser.parse_args()
    itr = args.itr
    seed_itrs = args.seeds_itr.split(',')
    type = args.type
    dfs = []
    stat_path = ''
    root_dir = args.root_dir
    main(itr, seed_itrs, type, stat_path, root_dir)
    for manifold_learner in ['UMAP', 'TSNE', 'Isomap', 'LLE', '']:
        for method in ['Gmm', 'Kmeans', 'Spectral']:
            sub_itr = method+manifold_learner+'/'+itr
            print('start '+sub_itr)
            main(sub_itr, seed_itrs, type, stat_path, root_dir)
