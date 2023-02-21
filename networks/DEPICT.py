"""
Based on Theano implementation https://github.com/herandy/DEPICT
and article :
        Dizaji, K. G., Herandi, A., & Huang, H. (2017).
        Deep clustering via joint convolutional autoencoder embedding and relative entropy minimization

Author:
Baptiste Lafabregue 2019.25.04
"""
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras.models import Model

from networks.trainer import Trainer


class DEPICT(Trainer):
    def __init__(self,
                 dataset_name,
                 classifier_name,
                 encoder_model,
                 keep_both_losses=True,
                 gamma=0.1,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=10,
                 tol=1e-3,
                 update_interval=5,
                 pred_normalizition_flag=True,
                 optimizer=None):

        super(DEPICT, self).__init__(dataset_name, classifier_name, encoder_model, batch_size, n_clusters, optimizer)

        self.keep_both_losses = keep_both_losses
        self.gamma = gamma
        self.alpha = alpha
        self.tol = tol
        self.update_interval = update_interval
        self.pred_normalizition_flag = pred_normalizition_flag
        self.clust_model = None
        self.clust_loss = None

    def initialize_model(self, x, y, ae_weights=None):
        """
        Initialize the model for training
        :param ae_weights: arguments to let the encoder load its weights, None to pre-train the encoder
        """
        if ae_weights is not None:
            self.encoder_model.load_weights(ae_weights)
            print('Pretrained AE weights are loaded successfully.')
            self.pretrain_model = False
        else:
            self.pretrain_model = True

        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.legacy.Adam()

        clustering_layer = tf.keras.layers.Dense(self.n_clusters, name='clustering')(self.encoder.output)
        clustering_layer = tf.keras.layers.Softmax()(clustering_layer)
        self.clust_model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        self.clust_loss = tf.keras.losses.CategoricalCrossentropy()

    def load_weights(self, weights_path):
        """
        Load weights of IDEC model
        :param weights_path: path to load weights from
        """
        self.clust_model.load_weights(weights_path + '.tf')

    def save_weights(self, weights_path):
        """
        Save weights of IDEC model
        :param weights_path: path to save weights to
        """
        self.clust_model.save_weights(weights_path + '.tf')

    def predict_clusters(self, x, seeds=None):
        """
        Predict cluster labels using the output of clustering layer
        :param x: the data to evaluate
        :param seeds: seeds to initialize the K-Means if needed
        :return: the predicted cluster labels
        """
        q = self.clust_model.predict(x, verbose=0)
        y_pred = q.argmax(1)

        return y_pred, np.transpose(self.clust_model.get_layer(name='clustering').get_weights()[0])

    def _run_training(self, x, y, x_test, y_test, nb_steps,
                      seeds, verbose, log_writer, dist_matrix=None):
        if seeds is not None:
            seeds_enc = self.extract_features(seeds)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, init=seeds_enc)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        x_pred = self.extract_features(x)
        y_pred = kmeans.fit_predict(x_pred)

        centroids = kmeans.cluster_centers_.T
        centroids = centroids / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))
        self.clust_model.get_layer(name='clustering').set_weights([centroids, np.zeros((self.n_clusters,))])

        if y is not None:
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            if verbose:
                print('ari kmeans: ', str(ari))
            self.log_stats(x, y, x_test, y_test, [0, 0, 0], 0, log_writer, 'init')

        i = 0  # Number of performed optimization steps
        epoch = 0  # Number of performed epochs

        # evaluate the clustering performance
        target_pred = self.clust_model.predict(x, verbose=0)
        if self.pred_normalizition_flag:
            cluster_frequency = np.sum(target_pred, axis=0)  # avoid unbalanced assignment
            target_pred = target_pred ** 2 / cluster_frequency
            # y_prob = y_prob / np.sqrt(cluster_frequency)
            target_pred = np.transpose(target_pred.T / np.sum(target_pred, axis=1))
        target_pred_last = target_pred

        # define the train function
        train_enc_loss = tf.keras.metrics.Mean(name='encoder train_loss')
        clust_enc_loss = tf.keras.metrics.Mean(name='clustering train_loss')
        depict_enc_loss = tf.keras.metrics.Mean(name='DEPICT train_loss')

        @tf.function
        def train_step(x_batch, target_batch):
            with tf.GradientTape() as tape:
                encoder_loss = self.encoder_model.loss.compute_loss(x_batch, training=True)
                encoding_x = self.clust_model(x_batch, training=True)
                depict_loss = self.clust_loss(target_batch, encoding_x)
                loss = (1 - self.gamma) * encoder_loss + self.gamma * depict_loss
            gradients = tape.gradient(loss,
                                      self.encoder_model.get_trainable_variables() + self.clust_model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.encoder_model.get_trainable_variables() + self.clust_model.trainable_variables))

            train_enc_loss(encoder_loss)
            clust_enc_loss(loss)
            depict_enc_loss(loss)

        if verbose:
            print('start training')
        # idec training
        while i < nb_steps:
            train_enc_loss.reset_states()
            clust_enc_loss.reset_states()
            depict_enc_loss.reset_states()
            # shuffle the train set

            # computes P each update_interval epoch
            if epoch % self.update_interval == 0:
                # evaluate the clustering performance
                target_pred = self.clust_model.predict(x, verbose=0)
                if self.pred_normalizition_flag:
                    cluster_frequency = np.sum(target_pred, axis=0)  # avoid unbalanced assignment
                    target_pred = target_pred ** 2 / cluster_frequency
                    # y_prob = y_prob / np.sqrt(cluster_frequency)
                    target_pred = np.transpose(target_pred.T / np.sum(target_pred, axis=1))
                delta_label = np.sum(target_pred != target_pred_last).astype(np.float32) / target_pred.shape[0]
                target_pred_last = target_pred

                # check stop criterion
                if epoch > 0 and delta_label < self.tol:
                    if verbose:
                        print('delta_label ', delta_label, '< tol ', self.tol)
                        print('Reached tolerance threshold. Stopping training.')
                    self.log_stats(x, y, x_test, y_test, [0, 0, 0],
                                   epoch, log_writer, 'reached_stop_criterion')
                    break

            train_ds = tf.data.Dataset.from_tensor_slices((x, target_pred)) \
                .shuffle(x.shape[0], reshuffle_each_iteration=True) \
                .batch(self.batch_size).as_numpy_iterator()

            for x_batch, target_batch in train_ds:
                train_step(x_batch, target_batch)
                i += 1
                if i >= nb_steps:
                    break

            if verbose:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1, depict_enc_loss.result()))
            epoch += 1

            y_pred = self.log_stats(x, y,
                                    x_test, y_test,
                                    [depict_enc_loss.result(), clust_enc_loss.result(), train_enc_loss.result()],
                                    epoch,
                                    log_writer,
                                    'train')

        return epoch
