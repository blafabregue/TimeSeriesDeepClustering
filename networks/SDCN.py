"""
Based on torch implementation https://github.com/bdy9527/SDCN
and article :
        Deyu Bo, Xiao Wang, Chuan Shi, Meiqi Zhu, Emiao Lu, and Peng Cui,
        Structural Deep Clustering Network

Author:
Baptiste Lafabregue 2019.25.04
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn import metrics
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Layer, ReLU, Reshape, GlobalMaxPool1D
from tensorflow.keras import Model, Input

import utils
from networks.IDEC import ClusteringLayer
from networks.trainer import Trainer


class GNNLayer(Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 adjency_matrix,
                 final=False,
                 **kwargs):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.final = final
        w_init = tf.initializers.GlorotUniform()
        self.weight = tf.Variable(initial_value=w_init(shape=(in_features, out_features), dtype=K.floatx()),
                                  trainable=True)
        self.adj = adjency_matrix
        super(GNNLayer, self).__init__(**kwargs)

    def call(self, features):
        support = K.dot(features, self.weight)
        output = tf.sparse.sparse_dense_matmul(self.adj, support)
        if self.final:
            output = tf.keras.activations.softmax(output)
        else:
            output = ReLU()(output)
        return output


class SDCN(Trainer):
    def __init__(self,
                 dataset_name,
                 classifier_name,
                 encoder_model,
                 keep_both_losses=True,
                 gamma=0.4,
                 n_clusters=10,
                 alpha=1.0,
                 tol=1e-3,
                 update_interval=5,
                 batch_size=10,
                 optimizer=None,
                 graph_path='.'
                 ):

        super(SDCN, self).__init__(dataset_name, classifier_name, encoder_model, batch_size, n_clusters, optimizer)

        self.keep_both_losses = keep_both_losses
        self.gamma = gamma
        self.alpha = alpha
        self.tol = tol
        self.update_interval = update_interval
        self.graph_path = graph_path
        self.pretrain_model = True
        self.sdcn_model = None
        self.dec_loss = None
        self.gnn_layers = []

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

        if not self.graph_path.endswith('.npy'):
            self.graph_path = self.graph_path + '/graph.npy'
            utils.construct_knn_graph(x, y, self.graph_path)
        adj = utils.load_graph(len(x), self.graph_path)

        inputs = Input((x.shape[1:]))
        # init gnn layers
        enc_output = self.encoder(inputs)
        # self._format_output(enc_output)
        h = Reshape((-1,))(inputs)
        ts_length = x.shape[1]
        channels = x.shape[2]
        output_shape_prev = K.int_shape(enc_output[0])[-1]
        h = GNNLayer(ts_length*channels, output_shape_prev, adj)(h)
        for i in range(0, len(enc_output) - 1):
            output_shape = K.int_shape(enc_output[i + 1])[-1]
            enc_layer = enc_output[i]
            if len(K.int_shape(enc_layer)) > 2:
                enc_layer = GlobalMaxPool1D()(enc_layer)
            h = GNNLayer(output_shape_prev, output_shape, adj)(tf.keras.layers.add([h, enc_layer]))
            output_shape_prev = output_shape
        enc_layer = enc_output[-1]
        gnn_output = GNNLayer(output_shape_prev, self.n_clusters,
                              adj, final=True)(tf.keras.layers.add([h, enc_layer]))

        dnn_output = ClusteringLayer(self.n_clusters, name='clustering')(enc_layer)
        self.sdcn_model = Model(inputs=inputs, outputs=[dnn_output, gnn_output])
        self.dec_loss = tf.keras.losses.KLDivergence()

    @staticmethod
    def _format_output(enc_output):
        for i in range(len(enc_output)):
            if len(K.int_shape(enc_output[i])) > 2:
                enc_output[i] = GlobalMaxPool1D()(enc_output[i])

    def load_weights(self, weights_path):
        """
        Load weights of IDEC model
        :param weights_path: path to load weights from
        """
        self.sdcn_model.load_weights(weights_path)

    def save_weights(self, weights_path):
        """
        Save weights of IDEC model
        :param weights_path: path to save weights to
        """
        self.sdcn_model.save_weights(weights_path)

    def extract_features(self, x):
        """
        Extract features from the encoder (before the clustering layer)
        :param x: the data to extract features from
        :return: the encoded features
        """
        return self.encoder.predict(x)[-1]

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

        self.sdcn_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        if y is not None:
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            if verbose:
                print('ari kmeans: ', str(ari))
            self.log_stats(x, y, x_test, y_test, [0, 0, 0], 0, log_writer, 'init')

        q, _ = self.sdcn_model.predict(x, verbose=0, batch_size=len(x))
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
                q, q_gcn = self.sdcn_model(x_batch, training=True)
                dec_loss = tf.keras.losses.KLD(p_batch, q)
                gcn_loss = tf.keras.losses.KLD(p_batch, q_gcn)
                loss = (1 - self.gamma) * encoder_loss + self.gamma * (dec_loss + gcn_loss)
            gradients = tape.gradient(loss, self.encoder_model.get_trainable_variables()
                                      + self.sdcn_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.encoder_model.get_trainable_variables()
                                               + self.sdcn_model.trainable_variables))

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
                q, _ = self.sdcn_model.predict(x, verbose=0, batch_size=len(x))
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

            # for this method we cannot use mini batch
            # train_ds = tf.data.Dataset.from_tensor_slices((x, p)) \
            #     .shuffle(x.shape[0], reshuffle_each_iteration=True) \
            #     .batch(self.x.shape[0]).as_numpy_iterator()
            #
            # for x_batch, p_batch in train_ds:
            train_step(x, p)
            i += 1

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
