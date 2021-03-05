"""
Script to launch experiments with non deep methods
See parse_arguments() function for details on arguments

Author:
Baptiste Lafabregue 2019.25.04
"""
import os
import json
import numpy as np
import argparse
from tslearn.clustering import TimeSeriesKMeans
from kshape.core import _kshape
from sklearn import metrics

import tensorflow as tf

import utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='d', required=True,
                        help='dataset name')
    parser.add_argument('--archives', type=str, metavar='DIR', required=True,
                        help='archive name')
    parser.add_argument('--itr', type=str, metavar='X', default='0',
                        help='iteration index')
    parser.add_argument('--seeds_itr', type=int, metavar='X', default=None,
                        help='seeds index, do not specify or -1 if none')
    parser.add_argument('--root_dir', type=str, metavar='PATH', default='.',
                        help='path of the root dir where archives and results are stored')
    parser.add_argument('--not_use_previous', default=False, action="store_true",
                        help='Flag to not use previous results and only computes one with errors, '
                             'but computes everything again, only for All option')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    tf.keras.backend.set_floatx('float64')

    root_dir = args.root_dir
    dataset_name = args.dataset
    archive_name = args.archives
    use_previous = not args.not_use_previous
    itr = args.itr
    seeds_itr = args.seeds_itr
    seeds = None
    if seeds_itr is not None and seeds_itr >= 0:
        seeds = utils.read_seeds(root_dir, archive_name, dataset_name, seeds_itr)

    train_dict = utils.read_dataset(root_dir, archive_name, dataset_name, True)
    x_train = train_dict[dataset_name][0]
    y_train = train_dict[dataset_name][1]
    input_shape = x_train.shape[1:]
    nb_classes = np.shape(np.unique(y_train, return_counts=True)[1])[0]

    train_dict = utils.read_dataset(root_dir, archive_name, dataset_name, False)
    x_test = train_dict[dataset_name][0]
    y_test = train_dict[dataset_name][1]

    stats_dir = root_dir + '/stats/' + str(itr) + '/' + str(seeds_itr) + '/'

    print('Launch euclidien kmeans : ' + dataset_name)
    km_euc = TimeSeriesKMeans(n_clusters=nb_classes, metric="euclidean", max_iter=200).fit(x_train)
    y_pred = km_euc.predict(x_train)
    km_euc = TimeSeriesKMeans(n_clusters=nb_classes, metric="euclidean", max_iter=200).fit(x_test)
    y_pred_test = km_euc.predict(x_test)

    acc = np.round(utils.cluster_acc(y_train, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_train, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_train, y_pred), 5)
    summarized = str(acc)+","+str(0)+","+str(nmi)+","+str(0)+","+str(ari)+","+str(0)
    print('acc : ' + str(acc))
    print('nmi : ' + str(nmi))
    print('ari : ' + str(ari))

    acc_test = np.round(utils.cluster_acc(y_test, y_pred_test), 5)
    nmi_test = np.round(metrics.normalized_mutual_info_score(y_test, y_pred_test), 5)
    ari_test = np.round(metrics.adjusted_rand_score(y_test, y_pred_test), 5)
    summarized_test = str(acc_test)+","+str(0)+","+str(nmi_test)+","+str(0)+","+str(ari_test)+","+str(0)

    utils.create_directory(stats_dir)
    with open(stats_dir + "kmeans_Eucl_None_0.0_" + dataset_name, "w") as f:
        f.write(summarized + '\n')
        f.write(summarized_test + '\n')
        f.write('0\n')

    print('Launch DTW/DBA kmeans : ' + dataset_name)
    km_dtw = TimeSeriesKMeans(n_clusters=nb_classes, metric="dtw", max_iter=200).fit(x_train)
    y_pred = km_dtw.predict(x_train)
    km_dtw = TimeSeriesKMeans(n_clusters=nb_classes, metric="dtw", max_iter=200).fit(x_test)
    y_pred_test = km_dtw.predict(x_test)

    acc = np.round(utils.cluster_acc(y_train, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_train, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_train, y_pred), 5)
    summarized = str(acc)+","+str(0)+","+str(nmi)+","+str(0)+","+str(ari)+","+str(0)
    print('acc : ' + str(acc))
    print('nmi : ' + str(nmi))
    print('ari : ' + str(ari))

    acc_test = np.round(utils.cluster_acc(y_test, y_pred_test), 5)
    nmi_test = np.round(metrics.normalized_mutual_info_score(y_test, y_pred_test), 5)
    ari_test = np.round(metrics.adjusted_rand_score(y_test, y_pred_test), 5)
    summarized_test = str(acc_test)+","+str(0)+","+str(nmi_test)+","+str(0)+","+str(ari_test)+","+str(0)

    with open(stats_dir + "kmeans_DTW_None_0.0_" + dataset_name, "w") as f:
        f.write(summarized + '\n')
        f.write(summarized_test + '\n')
        f.write('0\n')

    print('Launch Kshape : ' + dataset_name)
    x_train_kshape = np.reshape(x_train, (x_train.shape[0], -1))
    x_test_kshape = np.reshape(x_test, (x_test.shape[0], -1))
    y_pred, _ = _kshape(x_train_kshape, nb_classes)
    y_pred_test, _ = _kshape(x_test_kshape, nb_classes)

    acc = np.round(utils.cluster_acc(y_train, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_train, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_train, y_pred), 5)
    summarized = str(acc) + "," + str(0) + "," + str(nmi) + "," + str(0) + "," + str(ari) + "," + str(0)
    print('acc : ' + str(acc))
    print('nmi : ' + str(nmi))
    print('ari : ' + str(ari))

    acc_test = np.round(utils.cluster_acc(y_test, y_pred_test), 5)
    nmi_test = np.round(metrics.normalized_mutual_info_score(y_test, y_pred_test), 5)
    ari_test = np.round(metrics.adjusted_rand_score(y_test, y_pred_test), 5)
    summarized_test = str(acc_test) + "," + str(0) + "," + str(nmi_test) + "," + str(0) + "," + str(
        ari_test) + "," + str(0)

    utils.create_directory(stats_dir)
    with open(stats_dir + "kshape_None_None_0.0_" + dataset_name, "w") as f:
        f.write(summarized + '\n')
        f.write(summarized_test + '\n')
        f.write('0\n')

