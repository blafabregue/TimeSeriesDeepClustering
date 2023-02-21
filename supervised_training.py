"""
Base on Keras Improved Deep Embedded Clustering (IDEC) algorithm implementation:

        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin. Improved Deep Embedded Clustering with Local Structure Preservation. IJCAI 2017.

Original Author:
    Xifeng Guo. 2017.1.30
Autor:
Baptiste Lafabregue 2019.25.04
"""

import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn import metrics
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Model

import utils


class SupervisedResearch(object):
    def __init__(self,
                 dataset_name,
                 classifier_name,
                 encoder_model,
                 gamma=0.1,
                 n_clusters=10,
                 batch_size=10,
                 optimizer=None):

        super(SupervisedResearch, self).__init__()

        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.encoder_model = encoder_model
        self.encoder = encoder_model.encoder
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.optimizer = optimizer

        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.legacy.Adam()

    def load_weights(self, weights_path):
        """
        Load weights of IDEC model
        :param weights_path: path to load weights from
        """
        self.dec_model.load_weights(weights_path)

    def save_weights(self, weights_path):
        """
        Save weights of IDEC model
        :param weights_path: path to save weights to
        """
        self.dec_model.save_weights(weights_path)

    def extract_feature(self, x):
        """
        Extract features from the encoder (before the clustering layer)
        :param x: the data to extract features from
        :return: the encoded features
        """
        return self.encoder.predict(x)

    def log_stats_encoder(self, x, y, x_test, y_test, loss, epoch, log_writer, comment):
        """
        Log the intermediate result to a file
        :param x: train data
        :param y: train labels
        :param x_test: test data
        :param y_test: test labels
        :param loss: array of losses values
        :param epoch: current epoch
        :param logwriter: log file writer
        :param comment: comment to add to the log
        :return:
        """
        acc = 0
        nmi = 0
        ari = 0
        acc_test = 0
        nmi_test = 0
        ari_test = 0

        loss = np.round(loss, 5)

        if y is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            x_pred = self.encoder.predict(x)
            y_pred = kmeans.fit_predict(x_pred)

            acc = np.round(utils.cluster_acc(y, y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            # print('ari : '+str(ari))
            # sq_error = utils.computes_dtw_regularized_square_error(x, y_pred)

        if x_test is not None and y_test is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            x_pred = self.encoder.predict(x_test)
            y_pred_test = kmeans.fit_predict(x_pred)

            acc_test = np.round(utils.cluster_acc(y_test, y_pred_test), 5)
            nmi_test = np.round(metrics.normalized_mutual_info_score(y_test, y_pred_test), 5)
            ari_test = np.round(metrics.adjusted_rand_score(y_test, y_pred_test), 5)

        log_dict = dict(iter=epoch, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2],
                       acc_test=acc_test, nmi_test=nmi_test, ari_test=ari_test, comment='encoder : '+comment)
        log_writer.writerow(log_dict)

        return nmi, y_pred

    def clustering(self, x,
                   y=None,
                   update_interval=1,
                   nb_steps=50,
                   save_dir='./results/idec',
                   save_suffix='',
                   x_test=None,
                   y_test=None,
                   verbose=True,
                   ):

        if verbose:
            print('Update interval', update_interval)

        # logging file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/idec_log.csv', 'w')
        log_writer = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr',
                                                        'acc_test', 'nmi_test', 'ari_test', 'comment'])
        log_writer.writeheader()
        max_ari = -np.inf
        min_sq_error = np.inf
        max_sc_error = -np.inf
        ari_sq_error = 0.0
        max_model = None
        loss = [0, 0, 0]

        if verbose:
            self.encoder_model.summary()

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        train_loss = tf.keras.metrics.Mean(name='train_loss')

        dist_matrix = utils.cdist_dtw(x, n_jobs=3)

        # define the train function
        @tf.function
        def train_step(x_batch):
            with tf.GradientTape() as tape:
                loss = self.encoder_model.loss.compute_loss(x_batch)
            gradients = tape.gradient(loss, self.encoder_model.get_trainable_variables())
            self.optimizer.apply_gradients(zip(gradients, self.encoder_model.get_trainable_variables()))

            train_loss(loss)
        if verbose:
            print('start pre-train')
        # Encoder training
        while i < nb_steps:
            train_loss.reset_states()
            train_ds = tf.data.Dataset.from_tensor_slices(x)
            train_ds = train_ds.shuffle(x.shape[0], reshuffle_each_iteration=True)
            train_ds = train_ds.batch(self.batch_size).as_numpy_iterator()
            for batch in train_ds:
                train_step(batch)

                i += 1
                if i >= nb_steps:
                    break

            if verbose:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epochs + 1, train_loss.result()))
            epochs += 1
            ari, y_pred = self.log_stats_encoder(x, y, x_test, y_test, [train_loss.result(), 0, train_loss.result()],
                                   epochs, log_writer, 'pretrain')
            sq_error = utils.computes_dtw_silhouette_score(dist_matrix, y_pred)
            if ari > max_ari:
                max_ari = ari
                self.encoder.save_weights(save_dir + '/DCC_model_max_sat_.h5')
            if sq_error > max_sc_error:
                ari_sq_error = ari
                max_sc_error = sq_error
            print("ari : "+str(ari)+", max ari"+str(max_ari)+"   ##   sq error : "+str(sq_error)+", min sq error"+str(max_sc_error)+" with ari of : "+str(ari_sq_error))

        if verbose:
            print('end of pre-train')

        self.encoder.load_weights(save_dir + '/DCC_model_max_sat_.h5')
        # save idec model
        if verbose:
            print("max ari: "+str(max_ari))
            print('model saved at: ', save_dir+'/DCC_model_max_sat_'+save_suffix+'.h5')
        self.log_stats_encoder(x, y, x_test, y_test, loss, epochs, log_writer, 'final')

        logfile.close()



