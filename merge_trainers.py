"""
Script to merge representation (used for TripletKcombine loss)

Author:
Baptiste Lafabregue 2019.25.04
"""
import os
import numpy as np
import argparse

from sklearn import metrics
from sklearn.cluster import KMeans

import utils
from networks.trainer import get_logs

TRANSLATE_DICT = {'GAN': 'ClusterGAN', 'DEC': 'IDEC'}


def gen_triplet_list(prefix='dilated_cnn_tripletK', suffix='_None_0.0', itrs=['1', '2', '5', '10']):
    list = []
    for k in itrs:
        list.append(prefix + k + suffix)
    return list


def merge(methods_to_merge, itr, seed_itrs, output_method, stats_dir, dataset_names=None,
          archive_name='UCRArchive_2018'):
    # extract representation of each method
    if dataset_names is None:
        dataset_names = [name for name in os.listdir('./ae_weights/' + itr + '/' + methods_to_merge[0])]
        dataset_names.sort()

    types = ['_test', 'train']
    if type == 'test':
        types = ['_test']
    elif type == 'train':
        types = ''

    for seed_itr in seed_itrs:
        for dataset in dataset_names:
            print(dataset)
            train_dict = utils.read_dataset('.', archive_name, dataset, True)
            y_train = train_dict[dataset][1]
            nb_classes = np.shape(np.unique(y_train, return_counts=True)[1])[0]

            test_dict = utils.read_dataset('.', archive_name, dataset, False)
            y_test = test_dict[dataset][1]

            representations_train = []
            representations_test = []
            last_logs_train = {'ari': 0.0, 'nmi': 0.0, 'acc': 0.0,
                               'max_ari': -1.0, 'max_nmi': -1.0, 'max_acc': -1.0}
            last_logs_test = {'ari': 0.0, 'nmi': 0.0, 'acc': 0.0,
                              'max_ari': -1.0, 'max_nmi': -1.0, 'max_acc': -1.0}
            try:
                for method in methods_to_merge:
                    enc_type = method.split('_')[-1]
                    # special case for DEC that use the IDEC object
                    if enc_type in TRANSLATE_DICT:
                        enc_type = TRANSLATE_DICT[enc_type]
                    if not os.path.exists('./ae_weights/' + itr + '/' + method + '/' + dataset + '/' + enc_type):
                        enc_type = 'pre_train_encoder'
                    file = './ae_weights/' + itr + '/' + method + '/' + dataset + '/' + enc_type + '/x_encoded_'\
                           + seed_itr + '.npy'
                    representations_train.append(np.load(file))
                    file = './ae_weights/' + itr + '/' + method + '/' + dataset + '/' + enc_type + '/x_test_encoded_'\
                           + seed_itr + '.npy'
                    representations_test.append(np.load(file))

                x_train = np.concatenate(representations_train, axis=1)
                kmeans_train = KMeans(n_clusters=nb_classes, n_init=20)
                y_pred_train = kmeans_train.fit_predict(x_train)
                last_logs_train['acc'] = np.round(utils.cluster_acc(y_train, y_pred_train), 5)
                last_logs_train['nmi'] = np.round(metrics.normalized_mutual_info_score(y_train, y_pred_train), 5)
                last_logs_train['ari'] = np.round(metrics.adjusted_rand_score(y_train, y_pred_train), 5)

                x_test = np.concatenate(representations_test, axis=1)
                kmeans_test = KMeans(n_clusters=nb_classes, n_init=20)
                y_pred_test = kmeans_test.fit_predict(x_test)
                last_logs_test['acc'] = np.round(utils.cluster_acc(y_test, y_pred_test), 5)
                last_logs_test['nmi'] = np.round(metrics.normalized_mutual_info_score(y_test, y_pred_test), 5)
                last_logs_test['ari'] = np.round(metrics.adjusted_rand_score(y_test, y_pred_test), 5)

                save_dir = './ae_weights/' + itr + '/' + output_method + '/' + dataset
                utils.create_directory(save_dir + '/pre_train_encoder')
                np.save(save_dir + '/pre_train_encoder/kmeans_affect_' + str(seed_itr) + '.npy', y_pred_test)
                np.save(save_dir + '/pre_train_encoder/kmeans_clusters_' + str(seed_itr) + '.npy',
                        kmeans_test.cluster_centers_)
                np.save(save_dir + '/pre_train_encoder/x_encoded_' + str(seed_itr) + '.npy', x_train)
                np.save(save_dir + '/pre_train_encoder/x_test_encoded_' + str(seed_itr) + '.npy', x_test)

                itr_stats_dir = stats_dir + itr + '/' + seed_itr + '/' + output_method
                f = open(itr_stats_dir + "_" + dataset, "w")
                f.write(get_logs(last_logs_train) + '\n')
                f.write(get_logs(last_logs_test) + '\n')
                f.close()
            except:
                print("error on dataset " + dataset + " itr " + seed_itr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute combined kmeans'
    )
    parser.add_argument('--dataset', type=str, metavar='d', default=None,
                        help='dataset name')
    parser.add_argument('--archives', type=str, metavar='DIR', required=True,
                        help='archive name')
    parser.add_argument('--itr', type=str, metavar='d', default='20000',
                        help='iteration number')
    parser.add_argument('--prefix', type=str, metavar='d', default='dilated_cnn_tripletK',
                        help='prefix to use to select methods')
    parser.add_argument('--suffix', type=str, metavar='d', default='_None_0.0',
                        help='suffix to use to select methods')
    args = parser.parse_args()
    itr = args.itr
    prefix = args.prefix
    suffix = args.suffix
    dataset_name = args.dataset
    archive_name = args.archives
    if dataset_name is not None:
        dataset_name = [dataset_name]

    output_method = prefix + 'combined' + suffix
    methods_to_merge = gen_triplet_list(prefix=prefix, suffix=suffix)
    seed_itr = ['0', '1', '2', '3', '4']
    merge(methods_to_merge, itr, seed_itr, output_method, './stats/', dataset_names=dataset_name,
          archive_name=archive_name)
