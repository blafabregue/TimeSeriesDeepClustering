"""
Based on Keras implementation https://github.com/XifengGuo/IDEC:
and article :
        Xifeng Guo, Long Gao, Xinwang Liu, Jianping Yin.
        Improved Deep Embedded Clustering with Local Structure Preservation. IJCAI 2017.

Original Author:
    Xifeng Guo. 2017.1.30
Author:
Baptiste Lafabregue 2019.25.04
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn import metrics
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans

from networks.trainer import Trainer


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim),
                                        initializer='glorot_uniform', name='clustering')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IDEC(Trainer):
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
                 optimizer=None):

        super(IDEC, self).__init__(dataset_name, classifier_name, encoder_model, batch_size, n_clusters, optimizer)

        self.keep_both_losses = keep_both_losses
        self.gamma = gamma
        self.alpha = alpha
        self.tol = tol
        self.update_interval = update_interval
        self.dec_model = None
        self.dec_loss = None

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

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.dec_model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        self.dec_loss = tf.keras.losses.KLDivergence()

    def load_weights(self, weights_path):
        """
        Load weights of IDEC model
        :param weights_path: path to load weights from
        """
        self.dec_model.load_weights(weights_path + '.tf')

    def save_weights(self, weights_path):
        """
        Save weights of IDEC model
        :param weights_path: path to save weights to
        """
        self.dec_model.save_weights(weights_path + '.tf')

    def get_trainer_name(self):
        """
        Return the name of the training method used
        :return: method name
        """
        if self.gamma == 0.0:
            return 'DEC'
        return self.__class__.__name__

    def predict_clusters(self, x, seeds=None):
        """
        Predict cluster labels using the output of clustering layer
        :param x: the data to evaluate
        :param seeds: seeds to initialize the K-Means if needed
        :return: the predicted cluster labels
        """
        q = self.dec_model.predict(x, verbose=0)
        return q.argmax(1), self.dec_model.get_layer(name='clustering').get_weights()

    @staticmethod
    def target_distribution(q):
        """
        Target distribution P which enhances the discrimination of soft label Q
        :param q: the Q tensor
        :return: the P tensor
        """
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def _run_training(self, x, y, x_test, y_test, nb_steps,
                      seeds, verbose, log_writer, dist_matrix=None):
        if seeds is not None:
            seeds_enc = self.extract_features(seeds)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, init=seeds_enc)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        x_pred = self.extract_features(x)
        y_pred = kmeans.fit_predict(x_pred)

        self.dec_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        if y is not None:
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            if verbose:
                print('ari kmeans: ', str(ari))
            self.log_stats(x, y, x_test, y_test, [0, 0, 0], 0, log_writer, 'init')

        q = self.dec_model.predict(x, verbose=0)
        p = self.target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        y_pred_last = y_pred

        i = 0  # Number of performed optimization steps
        epoch = 0  # Number of performed epochs

        # define the train function
        train_enc_loss = tf.keras.metrics.Mean(name='encoder train_loss')
        dec_enc_loss = tf.keras.metrics.Mean(name='dec train_loss')
        idec_enc_loss = tf.keras.metrics.Mean(name='idec train_loss')

        @tf.function
        def train_step(x_batch, p_batch):
            with tf.GradientTape() as tape:
                encoder_loss = self.encoder_model.loss.compute_loss(x_batch, training=True)
                encoding_x = self.dec_model(x_batch, training=True)
                dec_loss = tf.keras.losses.KLD(p_batch, encoding_x)
                loss = (1 - self.gamma) * encoder_loss + self.gamma * dec_loss
            gradients = tape.gradient(loss,
                                      self.encoder_model.get_trainable_variables() + self.dec_model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.encoder_model.get_trainable_variables() + self.dec_model.trainable_variables))

            train_enc_loss(encoder_loss)
            dec_enc_loss(loss)
            idec_enc_loss(loss)

        if verbose:
            print('start training')
        # idec training
        while i < nb_steps:
            train_enc_loss.reset_states()
            dec_enc_loss.reset_states()
            idec_enc_loss.reset_states()
            # shuffle the train set

            # computes P each update_interval epoch
            if epoch % self.update_interval == 0:
                q = self.dec_model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

                # check stop criterion
                if epoch > 0 and delta_label < self.tol:
                    if verbose:
                        print('delta_label ', delta_label, '< tol ', self.tol)
                        print('Reached tolerance threshold. Stopping training.')
                    self.log_stats(x, y, x_test, y_test, [0, 0, 0],
                                   epoch, log_writer, 'reached_stop_criterion')
                    break

            train_ds = tf.data.Dataset.from_tensor_slices((x, p)) \
                .shuffle(x.shape[0], reshuffle_each_iteration=True) \
                .batch(self.batch_size).as_numpy_iterator()

            for x_batch, p_batch in train_ds:
                train_step(x_batch, p_batch)
                i += 1
                if i >= nb_steps:
                    break

            if verbose:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1, idec_enc_loss.result()))
            epoch += 1

            y_pred = self.log_stats(x, y,
                                    x_test, y_test,
                                    [idec_enc_loss.result(), dec_enc_loss.result(), train_enc_loss.result()],
                                    epoch,
                                    log_writer,
                                    'train')

        return epoch
