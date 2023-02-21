"""
Main script to launch experiments.
See parse_arguments() function for details on arguments

Based on keras Hassan Fawaz implementation https://github.com/hfawaz/dl-4-tsc

Author:
Baptiste Lafabregue 2019.25.04
"""
import os
import json
import numpy as np
import argparse
import traceback

import tensorflow as tf

from networks.encoders import EncoderModel
from networks import mlp_ae, dilated_causal_cnn, bi_dilated_RNN, bilstm_ae, fcnn_ae, resnet, birnn_ae, attention_rnn
from losses.losses import JointLearningLoss, TripletLoss, MSELoss, CombinedLoss, VAELoss
from networks.SDCN import SDCN
from networks.IDEC import IDEC
from networks.DTCR import DTCR
from networks.DEPICT import DEPICT
from networks.ClusterGAN import ClusterGAN
from networks.VADE import VADE

import utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='d', required=True,
                        help='dataset name')
    parser.add_argument('--archives', type=str, metavar='DIR', required=True,
                        help='archive name')
    parser.add_argument('--architecture', type=str, metavar='xxx', required=True,
                        choices=['dilated_cnn', 'mlp', 'fcnn', 'res_cnn', 'bi_lstm', 'dilated_rnn', 'bi_rnn', 'bi_gru',
                                 'attention'],
                        help='Type of encoder architecture to use among : '
                             '[dilated_cnn, mlp, fcnn, res_cnn, bi_lstm, dilated_rnn, bi_rnn, bi_gru, attention]')
    parser.add_argument('--encoder_loss', type=str, metavar='xxx', required=True,
                        choices=['joint', 'reconstruction', 'triplet', 'vae', 'combined'],
                        help='Type of loss to pretrain the encoder to use among : '
                             '[joint, reconstruction, triplet, vae, combined]')
    parser.add_argument('--clustering_loss', type=str, metavar='xxx', required=True,
                        choices=['DEPICT', 'DTCR', 'SDCN', 'DEC', 'IDEC', 'GAN', 'VADE', 'None', 'All'],
                        help='Type of clustering framework to use among : [DEPICT, DTCR, SDCN, DEC, IDEC, GAN, VADE, '
                             'None, All]')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of hyperparameters to use; ' +
                             'for training; must be a JSON file')
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
    parser.add_argument('--nbneg', type=int, metavar='X', default=None,
                        help='number of negative sample, only for triplet loss, 1, 2, 5 or 10 are recommended')
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
    architecture = args.architecture
    encoder_loss = args.encoder_loss
    clustering_loss = args.clustering_loss
    use_previous = not args.not_use_previous
    dropout_rate = args.dropout
    noise = args.noise
    if noise == 'none':
        noise = None
    itr = args.itr
    seeds_itr = args.seeds_itr
    seeds = None
    if seeds_itr is not None and seeds_itr >= 0:
        seeds = utils.read_seeds(root_dir, archive_name, dataset_name, seeds_itr)

    print('Launch ' + clustering_loss + ' with ' + architecture + 'encoder on : ' + dataset_name)

    train_dict = utils.read_dataset(root_dir, archive_name, dataset_name, True)
    x_train = train_dict[dataset_name][0]
    y_train = train_dict[dataset_name][1]
    input_shape = x_train.shape[1:]
    nb_classes = np.shape(np.unique(y_train, return_counts=True)[1])[0]

    test_dict = utils.read_dataset(root_dir, archive_name, dataset_name, False)
    x_test = test_dict[dataset_name][0]
    y_test = test_dict[dataset_name][1]

    hf = open(os.path.join(args.hyper), 'r')
    params = json.load(hf)
    hf.close()

    if params['latent_dim'] <= 0:
        params['latent_dim'] = int(np.ceil(input_shape[0]*0.1))

    if params['nb_steps_pretrain'] is None:
        params['nb_steps_pretrain'] = params['nb_steps']

    optimizer = tf.keras.optimizers.legacy.Adam(params['lr'])
    layers_generator = None
    loss = None
    trainer = None

    ##################################################################################
    # define the encoder among [dilated_cnn, mlp, res_cnn, fcnn, bi_lstm, dilated_rnn]
    # for each we define an encoder and the autoencoder (the encoder + the decoder)
    if architecture == 'dilated_cnn':
        dilations = dilated_causal_cnn.compute_adaptive_dilations(x_train.shape[1])
        # dilations=None
        layers_generator = dilated_causal_cnn.AutoEncoder(x_train, encoder_loss,
                                                          nb_filters=params['nb_filters'],
                                                          depth=params['depth'],
                                                          reduced_size=params['reduced_size'],
                                                          latent_dim=params['latent_dim'],
                                                          kernel_size=params['kernel_size'],
                                                          dilations=dilations,
                                                          dropout_rate=dropout_rate)
    elif architecture == "dilated_rnn":
        layers_generator = bi_dilated_RNN.AutoEncoder(x_train, encoder_loss,
                                                      nb_steps=params['nb_steps'],
                                                      batch_size=params['batch_size'],
                                                      nb_RNN_units=[100, 50, params['latent_dim']])
        optimizer = layers_generator.optimizer
    elif architecture == "bi_lstm":
        layers_generator = bilstm_ae.AutoEncoder(x_train, encoder_loss, nb_classes,
                                                 nb_steps=params['nb_steps'])
    elif architecture == "bi_rnn":
        layers_generator = birnn_ae.AutoEncoder(x_train, encoder_loss, nb_classes,
                                                latent_dim=params['latent_dim'])
    elif architecture == "bi_gru":
        layers_generator = birnn_ae.AutoEncoder(x_train, encoder_loss, nb_classes,
                                                latent_dim=params['latent_dim'], cell_type='GRU')
    elif architecture == "attention":
        layers_generator = attention_rnn.AutoEncoder(x_train, encoder_loss, nb_classes,
                                                     latent_dim=params['latent_dim'])
    elif architecture == "mlp":
        layers_generator = mlp_ae.AutoEncoder(x_train, encoder_loss,
                                              latent_dim=params['latent_dim'], dropout_rate=dropout_rate)
    elif architecture == "fcnn":
        layers_generator = fcnn_ae.AutoEncoder(x_train, encoder_loss,
                                               latent_dim=params['latent_dim'], dropout_rate=dropout_rate)
    elif architecture == "res_cnn":
        layers_generator = resnet.AutoEncoder(x_train, encoder_loss,
                                              latent_dim=params['latent_dim'], dropout_rate=dropout_rate)

    encoder = layers_generator.get_encoder()
    decoder = layers_generator.get_decoder()
    autoencoder = layers_generator.get_auto_encoder()

    #################################################
    # define the loss among [reconstruction, triplet]
    if encoder_loss == "joint":
        loss = JointLearningLoss(layers_generator)
    if encoder_loss == "reconstruction":
        loss = MSELoss(autoencoder)
    elif encoder_loss == "triplet":
        if args.nbneg is not None:
            params['nb_random_samples'] = args.nbneg
        if params['compared_length'] is None:
            params['compared_length'] = np.inf
        fixed_time_dim = not (architecture in ['bi_lstm', 'bi_rnn', 'bi_gru', 'dilated_cnn'])
        loss = TripletLoss(encoder, x_train, params['compared_length'],
                           params['nb_random_samples'], params['negative_penalty'],
                           fixed_time_dim=fixed_time_dim)
        encoder_loss = encoder_loss + 'K' + str(params['nb_random_samples'])
    elif encoder_loss == "combined":
        if args.nbneg is not None:
            params['nb_random_samples'] = args.nbneg
        if params['compared_length'] is None:
            params['compared_length'] = np.inf
        loss_triplet = TripletLoss(encoder, x_train, params['compared_length'],
                                   params['nb_random_samples'], params['negative_penalty'])
        loss_mse = MSELoss(autoencoder)
        loss = CombinedLoss([loss_mse, loss_triplet])
    elif encoder_loss == 'vae':
        if decoder is None:
            raise utils.CompatibilityException('architecture incompatible with VAE loss')
        loss = VAELoss(encoder, decoder)
        # specified for reconstruction function
        autoencoder.set_use_vae(True)

    # we init the directory here because of different triplet loss versions
    enc_name = architecture + '_' + encoder_loss + '_' + str(noise) + '_' + str(dropout_rate)
    if clustering_loss == 'None':
        framework_name = enc_name
    else:
        framework_name = enc_name + '_' + clustering_loss
    output_directory = utils.create_output_path(root_dir, itr, framework_name, dataset_name)
    if clustering_loss != 'All':
        output_directory = utils.create_directory(output_directory)
    utils.create_directory(output_directory + 'encoder')
    stats_dir = root_dir + '/stats/' + str(itr) + '/' + str(seeds_itr) + '/'
    utils.create_directory(stats_dir)

    ##################################################################
    # define the clustering loss used ['DTCR', 'SDCN', 'IDEC', 'None']
    only_pretrain = False
    # for SDCN we need all layers' outputs
    if clustering_loss == 'SDCN':
        encoder = layers_generator.get_all_layers_encoder()
    encoder_model = EncoderModel(encoder, autoencoder, optimizer, loss, enc_name,
                                 batch_size=params['batch_size'], nb_steps=params['nb_steps'])

    if clustering_loss == 'DTCR':
        trainer = DTCR(dataset_name, framework_name, encoder_model, n_clusters=nb_classes,
                       batch_size=params['batch_size'])
    elif clustering_loss == 'IDEC':
        trainer = IDEC(dataset_name, framework_name, encoder_model, n_clusters=nb_classes,
                       batch_size=params['batch_size'])
    elif clustering_loss == 'DEC':
        trainer = IDEC(dataset_name, framework_name, encoder_model, n_clusters=nb_classes,
                       batch_size=params['batch_size'], gamma=1.0)
    elif clustering_loss == 'DEPICT':
        trainer = DEPICT(dataset_name, framework_name, encoder_model, n_clusters=nb_classes,
                         batch_size=params['batch_size'])
    elif clustering_loss == 'SDCN':
        trainer = SDCN(dataset_name, framework_name, encoder_model, n_clusters=nb_classes,
                       graph_path=output_directory, batch_size=params['batch_size'])
    elif clustering_loss == "None":
        trainer = IDEC(dataset_name, framework_name, encoder_model, n_clusters=nb_classes,
                       batch_size=params['batch_size'])
        only_pretrain = True
        params['nb_steps'] = 0
    elif clustering_loss == "GAN":
        if encoder_loss != "reconstruction":
            raise utils.CompatibilityException('For GAN use default reconstruction encoder loss, '
                                               'it will be not use anyway')
        trainer = ClusterGAN(dataset_name, framework_name, encoder_model.encoder, decoder,
                             layers_generator.get_discriminator(), n_clusters=nb_classes,
                             batch_size=params['batch_size'])
    elif clustering_loss == "VADE":
        if encoder_loss != "vae":
            raise utils.CompatibilityException('VADE is only compatible with vae encoder loss')
        trainer = VADE(dataset_name, framework_name, encoder_model, decoder, n_clusters=nb_classes,
                       batch_size=params['batch_size'])

    if clustering_loss == "All":
        trainer = IDEC(dataset_name, enc_name, encoder_model, n_clusters=nb_classes,
                       batch_size=params['batch_size'])
        trainer.initialize_model(x_train, y_train)
        encoder_directory = utils.create_output_path(root_dir, itr, enc_name, dataset_name)
        encoder_directory = utils.create_directory(encoder_directory)
        saved_ae_weights = encoder_directory + 'pre_train_encoder/encoder_' + str(seeds_itr) + '_'
        if encoder_model.exists(saved_ae_weights) and use_previous:
            print("Encoder already computed")
        else:
            trainer.clustering(x_train, y_train, nb_steps=0, save_dir=encoder_directory,
                               nb_steps_pretrain=params['nb_steps_pretrain'], only_pretrain=True, stats_dir=stats_dir,
                               save_pretrain=True, seeds_itr=seeds_itr, seeds=seeds, x_test=x_test, y_test=y_test)

        for clust_loss in ['DEPICT', 'SDCN', 'DTCR', 'DEC', 'IDEC', 'GAN', 'VADE']:
            framework_name = enc_name + '_' + clust_loss
            save_path = stats_dir + framework_name + "_" + dataset_name
            error_path = save_path + '.error'
            output_directory = utils.create_output_path(root_dir, itr, framework_name, dataset_name)
            output_directory = utils.create_directory(output_directory)
            ae_weights = saved_ae_weights

            if os.path.exists(save_path) and use_previous:
                print(save_path + " alreday exists, clustering loss skipped")
                continue
            else:
                if os.path.exists(error_path):
                    os.remove(error_path)

            if clust_loss == 'DTCR':
                trainer = DTCR(dataset_name, enc_name + '_' + clust_loss, encoder_model, n_clusters=nb_classes,
                               batch_size=params['batch_size'])
            elif clust_loss == 'IDEC':
                trainer = IDEC(dataset_name, enc_name + '_' + clust_loss, encoder_model, n_clusters=nb_classes,
                               batch_size=params['batch_size'])
            elif clust_loss == 'DEC':
                trainer = IDEC(dataset_name, framework_name, encoder_model, n_clusters=nb_classes,
                               batch_size=params['batch_size'], gamma=1.0)
            elif clust_loss == 'DEPICT':
                trainer = DEPICT(dataset_name, framework_name, encoder_model, n_clusters=nb_classes,
                                 batch_size=params['batch_size'])
            elif clust_loss == 'SDCN':
                if architecture in ['bi_lstm', 'bi_rnn', 'bi_gru', 'dilated_rnn']:
                    continue
                encoder_sdcn = layers_generator.get_all_layers_encoder()
                encoder_model_sdcn = EncoderModel(encoder_sdcn, autoencoder, optimizer, loss, enc_name,
                                                  batch_size=params['batch_size'], nb_steps=params['nb_steps'])
                trainer = SDCN(dataset_name, enc_name + '_' + clust_loss, encoder_model_sdcn, n_clusters=nb_classes,
                               graph_path='./graphs/' + dataset_name + '.npy', batch_size=params['batch_size'])
            elif clust_loss == "GAN":
                if encoder_loss != "reconstruction":
                    continue
                trainer = ClusterGAN(dataset_name, enc_name + '_' + clust_loss, encoder_model.encoder, decoder,
                                     layers_generator.get_discriminator(), n_clusters=nb_classes,
                                     batch_size=params['batch_size'])
                ae_weights = None
            elif clust_loss == "VADE":
                if encoder_loss != "vae":
                    continue
                trainer = VADE(dataset_name, enc_name + '_' + clust_loss, encoder_model, decoder, n_clusters=nb_classes,
                               batch_size=params['batch_size'])

            trainer.initialize_model(x_train, y_train, ae_weights=ae_weights)
            try:
                trainer.clustering(x_train, y_train, nb_steps=params['nb_steps'], save_dir=output_directory,
                                   nb_steps_pretrain=params['nb_steps_pretrain'], x_test=x_test, y_test=y_test,
                                   only_pretrain=False, seeds_itr=seeds_itr, seeds=seeds, stats_dir=stats_dir)
            except:
                error_path = stats_dir + framework_name + "_" + dataset_name + '.error'
                print('***********************************************************')
                print('ERROR printed in file ' + error_path)
                print('***********************************************************')
                with open(error_path, "w") as file:
                    traceback.print_exc(file=file)

    else:
        trainer.initialize_model(x_train, y_train)
        trainer.clustering(x_train, y_train, nb_steps=params['nb_steps'], save_dir=output_directory,
                           nb_steps_pretrain=params['nb_steps_pretrain'], x_test=x_test, y_test=y_test,
                           only_pretrain=only_pretrain, seeds_itr=seeds_itr, seeds=seeds, noise=noise)

    # features = trainer.extract_features(x_train)
    #
    # nb_classes = np.shape(np.unique(y_train, return_counts=True)[1])[0]
    # train_size = np.shape(features)[0]
    # penalty = None
    # # classifier = sklearn.svm.SVC(
    # #     C=1 / penalty
    # #         if penalty is not None and penalty > 0
    # #         else np.inf,
    # #     gamma='scale'
    # # )
    # #
    # # grid_search = sklearn.model_selection.GridSearchCV(
    # #     classifier, {
    # #         'C': [
    # #             0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
    # #             np.inf
    # #         ],
    # #         'kernel': ['rbf'],
    # #         'degree': [3],
    # #         'gamma': ['scale'],
    # #         'coef0': [0],
    # #         'shrinking': [True],
    # #         'probability': [False],
    # #         'tol': [0.001],
    # #         'cache_size': [200],
    # #         'class_weight': [None],
    # #         'verbose': [False],
    # #         'max_iter': [10000000],
    # #         'decision_function_shape': ['ovr'],
    # #         'random_state': [None]
    # #     },
    # #     cv=5, iid=False, n_jobs=5
    # # )
    # # if train_size <= 10000:
    # #     grid_search.fit(features, y_train)
    # # else:
    # #     # If the training set is too large, subsample 10000 train
    # #     # examples
    # #     split = sklearn.model_selection.train_test_split(
    # #         features, y_train,
    # #         train_size=10000, random_state=0, stratify=y_train
    # #     )
    # #     grid_search.fit(split[0], split[2])
    # # classifier = grid_search.best_estimator_
    # classifier = RidgeClassifierCV(alphas=10 ** np.linspace(-3, 3, 10), normalize=True)
    # classifier.fit(features, y_train)
    #
    # features = trainer.extract_features(x_test)
    # score = classifier.score(features, y_test)
    # print("Test accuracy: " + str(score))
