"""
Script to launch experiments with a dimension reduction
See parse_arguments() function for details on arguments

Based on implementation in https://github.com/rymc/n2d
and article:
McConville, R., Santos-Rodriguez, R., Piechocki, R. J., & Craddock, I. (2019).
N2d:(not too) deep clustering via clustering the local manifold of an autoencoded embedding

Author:
Baptiste Lafabregue 2019.25.04
"""
import numpy as np
import argparse
import traceback

import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import metrics
from sklearn import mixture

import umap

import utils
from networks.SDCN import SDCN
from networks.IDEC import IDEC
from networks.DTCR import DTCR
from networks.DEPICT import DEPICT
from networks.ClusterGAN import ClusterGAN
from networks.VADE import VADE
from networks.trainer import get_logs

UMAP_MIN_DIST = 0
UMAP_NEIGHBORS = 10


def regrister_logs(y, y_pred):
    logs = {'ari': np.round(metrics.adjusted_rand_score(y, y_pred), 5),
            'nmi': np.round(metrics.normalized_mutual_info_score(y, y_pred), 5),
            'acc': np.round(utils.cluster_acc(y, y_pred), 5),
            'max_ari': -1.0, 'max_nmi': -1.0, 'max_acc': -1.0}
    return logs


def eval_all_methods(x, y, n_clusters):
    meta_logs = {}
    for manifold_learner in ['UMAP', 'TSNE', 'Isomap', 'LLE', '']:
        hle = x
        if manifold_learner == 'UMAP':
            md = float(UMAP_MIN_DIST)
            hle = umap.UMAP(
                random_state=0,
                metric='euclidean',
                n_components=n_clusters,
                n_neighbors=UMAP_NEIGHBORS,
                min_dist=md).fit_transform(x)
        elif manifold_learner == 'LLE':
            hle = LocallyLinearEmbedding(
                n_components=n_clusters,
                n_neighbors=UMAP_NEIGHBORS).fit_transform(x)
        elif manifold_learner == 'TSNE':
            hle = TSNE(
                n_components=3,
                n_jobs=2,
                random_state=0,
                verbose=0).fit_transform(x)
        elif manifold_learner == 'Isomap':
            hle = Isomap(
                n_components=n_clusters,
                n_neighbors=5,
            ).fit_transform(x)

        for method in ['Gmm', 'Kmeans', 'Spectral']:
            y_pred = None
            if method == 'Gmm':
                gmm = mixture.GaussianMixture(
                    covariance_type='full',
                    n_components=n_clusters,
                    random_state=0)
                gmm.fit(hle)
                y_pred_prob = gmm.predict_proba(hle)
                y_pred = y_pred_prob.argmax(1)
            elif method == 'Kmeans':
                y_pred = KMeans(
                    n_clusters=n_clusters,
                    random_state=0).fit_predict(hle)
            elif method == 'Spectral':
                sc = SpectralClustering(
                    n_clusters=n_clusters,
                    random_state=0,
                    affinity='nearest_neighbors')
                y_pred = sc.fit_predict(hle)
            meta_logs[method + manifold_learner] = regrister_logs(y, y_pred)
    return meta_logs


def launch_clustering(output_directory, trainer, seeds_itr, root_dir, itr, framework_name,
                      dataset_name, y_test, y_train, x_train, x_test):
    features_test = np.load(output_directory + trainer + '/x_test_encoded_' + str(seeds_itr) + '.npy')
    features_train = np.load(output_directory + trainer + '/x_encoded_' + str(seeds_itr) + '.npy')
    nb_cluster = len(np.unique(y_train))

    train_logs = eval_all_methods(features_train, y_train, nb_cluster)
    test_logs = eval_all_methods(features_test, y_test, nb_cluster)
    concat_logs = eval_all_methods(np.concatenate((features_train, features_test), axis=0),
                                   np.concatenate((y_train, y_test), axis=0), nb_cluster)

    for key in train_logs.keys():
        stats_dir = root_dir + '/stats/'+key+'/' + str(itr) + '/' + str(seeds_itr) + '/'
        f = open(stats_dir + framework_name + "_" + dataset_name, "w")
        f.write(get_logs(train_logs[key]) + '\n')
        f.write(get_logs(test_logs[key]) + '\n')
        f.write(get_logs(concat_logs[key]) + '\n')
        f.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Alternative clustering methods on embeddings'
    )
    parser.add_argument('--dataset', type=str, metavar='d', required=True,
                        help='dataset name')
    parser.add_argument('--archives', type=str, metavar='DIR', required=True,
                        help='archive name')
    parser.add_argument('--architecture', type=str, metavar='xxx', required=True,
                        choices=['dilated_cnn', 'mlp', 'fcnn', 'res_cnn', 'bi_lstm', 'dilated_rnn', 'bi_rnn', 'bi_gru'],
                        help='Type of encoder architecture to use among : '
                             '[dilated_cnn, mlp, fcnn, res_cnn, bi_lstm, dilated_rnn, bi_rnn, bi_gru]')
    parser.add_argument('--encoder_loss', type=str, metavar='xxx', required=True,
                        choices=['joint', 'reconstruction', 'triplet', 'vae', 'combined'],
                        help='Type of loss to pretrain the encoder to use among : '
                             '[joint, reconstruction, triplet, vae, combined]')
    parser.add_argument('--clustering_loss', type=str, metavar='xxx', required=True,
                        choices=['DEPICT', 'DTCR', 'SDCN', 'DEC', 'IDEC', 'GAN', 'VADE', 'None', 'All'],
                        help='Type of clustering framework to use among : [DEPICT, DTCR, SDCN, DEC, IDEC, GAN, VADE, '
                             'None, All]')
    parser.add_argument('--itr', type=str, metavar='X', default='0',
                        help='iteration index')
    parser.add_argument('--seeds_itr', type=int, metavar='X', default=None,
                        help='seeds index, do not specify or -1 if none')
    parser.add_argument('--dropout', type=float, default=0.0, metavar="X.X",
                        help='Rate of dropout to use in the encoder architecture (not supported in RNN architecture)')
    parser.add_argument('--noise', type=str, metavar='xxx', default=None,
                        choices=['uniform', 'gaussian', 'laplace', 'drop', 'none'],
                        help='Type of noise to use, no noise is applied if not specified : '
                             '[uniform, gaussian, laplace, drop, none]')
    parser.add_argument('--nbneg', type=str, metavar='X', default=None,
                        help='number of negative sample, only for triplet loss, 1, 2, 5 or 10 are recommended')
    parser.add_argument('--root_dir', type=str, metavar='PATH', default='.',
                        help='path of the root dir where archives and results are stored')

    return parser.parse_args()


def main(root_dir, dataset_name, archive_name, architecture, encoder_loss,
         clustering_loss, dropout_rate, itr, seeds_itr):
    if encoder_loss == "triplet":
        encoder_loss = encoder_loss + 'K' + str(args.nbneg)

    print('Launch ' + clustering_loss + ' with ' + architecture + 'encoder on : ' + dataset_name)

    train_dict = utils.read_dataset(root_dir, archive_name, dataset_name, True)
    x_train = train_dict[dataset_name][0]
    y_train = train_dict[dataset_name][1]

    test_dict = utils.read_dataset(root_dir, archive_name, dataset_name, False)
    x_test = test_dict[dataset_name][0]
    y_test = test_dict[dataset_name][1]

    # we init the directory here because of different triplet loss versions
    enc_name = architecture + '_' + encoder_loss + '_' + str(noise) + '_' + str(dropout_rate)
    framework_name = enc_name
    # if clustering_loss != 'None':
    #     framework_name += '_' + clustering_loss
    # output_directory = utils.create_output_path(root_dir, itr, framework_name, dataset_name)
    # if clustering_loss is not 'All':
    #     output_directory = utils.create_directory(output_directory)
    # create akk
    for manifold_learner in ['UMAP', 'TSNE', 'Isomap', 'LLE', '']:
        for method in ['Gmm', 'Kmeans', 'Spectral']:
            stats_dir = root_dir + '/stats/' + method + manifold_learner + '/' + str(itr) + '/' + str(seeds_itr) + '/'
            utils.create_directory(stats_dir)

    if clustering_loss == "All":
        for clust_loss in ['DEPICT', 'SDCN', 'DTCR', 'DEC', 'IDEC', 'GAN', 'VADE', 'None']:
            framework_name = enc_name
            if clust_loss != 'None':
                framework_name += '_' + clust_loss
            # save_path = stats_dir+framework_name+"_"+dataset_name
            output_directory = utils.create_output_path(root_dir, itr, framework_name, dataset_name)

            trainer = None
            if clust_loss == 'DTCR':
                trainer = DTCR(dataset_name, framework_name, None).get_trainer_name()
            elif clust_loss == 'IDEC':
                trainer = IDEC(dataset_name, framework_name, None).get_trainer_name()
            elif clust_loss == 'DEC':
                trainer = IDEC(dataset_name, framework_name, None).get_trainer_name()
            elif clust_loss == 'DEPICT':
                trainer = DEPICT(dataset_name, framework_name, None).get_trainer_name()
            elif clust_loss == 'SDCN':
                trainer = SDCN(dataset_name, framework_name, None).get_trainer_name()
            elif clust_loss == "GAN":
                if encoder_loss != "reconstruction":
                    continue
                trainer = ClusterGAN(dataset_name, framework_name, None, None, None).get_trainer_name()
            elif clust_loss == "VADE":
                if encoder_loss != "vae":
                    continue
                trainer = VADE(dataset_name, framework_name, None).get_trainer_name()
            elif clust_loss == "None":
                trainer = 'pre_train_encoder'
            try:
                launch_clustering(output_directory, trainer, seeds_itr, root_dir, itr, framework_name,
                                  dataset_name, y_test, y_train, x_train, x_test)
            except:
                print('***********************************************************')
                print('ERROR ' + clust_loss)
                traceback.print_exc()
                print('***********************************************************')
    else:
        trainer = None
        if clustering_loss == 'DTCR':
            trainer = DTCR(dataset_name, framework_name, None).get_trainer_name()
        elif clustering_loss == 'IDEC':
            trainer = IDEC(dataset_name, framework_name, None).get_trainer_name()
        elif clustering_loss == 'DEC':
            trainer = IDEC(dataset_name, framework_name, None).get_trainer_name()
        elif clustering_loss == 'DEPICT':
            trainer = DEPICT(dataset_name, framework_name, None).get_trainer_name()
        elif clustering_loss == 'SDCN':
            trainer = SDCN(dataset_name, framework_name, None).get_trainer_name()
        elif clustering_loss == "GAN":
            if encoder_loss is not "reconstruction":
                print('GAN only works with reconstruction loss')
                return
            trainer = ClusterGAN(dataset_name, framework_name, None, None, None).get_trainer_name()
        elif clustering_loss == "VADE":
            if encoder_loss != "vae":
                print('VADE only works with vae loss')
                return
            trainer = VADE(dataset_name, framework_name, None).get_trainer_name()
        elif clustering_loss == "None":
            trainer = 'pre_train_encoder'
        output_directory = utils.create_output_path(root_dir, itr, framework_name, dataset_name)

        launch_clustering(output_directory, trainer, seeds_itr, root_dir, itr, framework_name,
                          dataset_name, y_test, y_train, x_train, x_test)


if __name__ == '__main__':
    args = parse_arguments()

    tf.keras.backend.set_floatx('float64')

    root_dir = args.root_dir
    dataset_name = args.dataset
    archive_name = args.archives
    architecture = args.architecture
    encoder_loss = args.encoder_loss
    clustering_loss = args.clustering_loss
    dropout_rate = args.dropout
    noise = args.noise
    if noise == 'none':
        noise = None
    itr = args.itr
    seeds_itr = args.seeds_itr
    main(root_dir, dataset_name, archive_name, architecture, encoder_loss,
         clustering_loss, dropout_rate, itr, seeds_itr)
