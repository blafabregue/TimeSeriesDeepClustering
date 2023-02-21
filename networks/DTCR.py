"""
Based on tensorflow implementation https://github.com/qianlima-lab/DTCR
and article :
        Ma, Q., Zheng, J., Li, S., & Cottrell, G. W. (2019),
        Learning Representations for Time Series Clustering

Author:
Baptiste Lafabregue 2019.25.04
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn import metrics
from sklearn.cluster import KMeans

import utils
from networks.trainer import Trainer


class DTCR(Trainer):
    def __init__(self,
                 dataset_name,
                 classifier_name,
                 encoder_model,
                 keep_both_losses=True,
                 lamda=1,
                 n_clusters=10,
                 batch_size=10,
                 classification_dim=128,
                 update_interval=1,
                 optimizer=None):

        super(DTCR, self).__init__(dataset_name, classifier_name, encoder_model, batch_size, n_clusters, optimizer)

        self.keep_both_losses = keep_both_losses
        self.lamda = lamda
        self.update_interval = update_interval
        self.pretrain_model = True
        self.classification_dim = classification_dim
        self.classification_model = None
        self.classification_loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

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
            self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-3)

        # fake/real classification task
        inputs = self.encoder.inputs
        h = self.encoder(inputs)
        h = tf.keras.layers.Dense(self.classification_dim)(h)
        h = tf.keras.layers.Dense(2, activation='softmax')(h)
        self.classification_model = tf.keras.Model(inputs=inputs, outputs=h)

    def load_weights(self, weights_path):
        """
        Load weights of IDEC model
        :param weights_path: path to load weights from
        """
        self.encoder_model.load_weights(weights_path)

    def save_weights(self, weights_path):
        """
        Save weights of IDEC model
        :param weights_path: path to save weights to
        """
        self.encoder_model.save_weights(weights_path)

    def _construct_F(self, H, k):
        # u, s, vh = np.linalg.svd(H, full_matrices=True)
        # F = np.transpose(vh[:k])
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)

        y_pred = kmeans.fit_predict(H)
        F = np.zeros(shape=(self.n_clusters, len(y_pred)))
        for i in range(len(y_pred)):
            F[y_pred[i]][i] = 1.0
        F = [sub * 1 / np.sqrt(np.sum(sub)) for sub in F]
        F = np.transpose(F)
        return F

    def _compute_loss(self, H, F):
        H_transpose = K.transpose(H)
        F_transpose = K.transpose(F)

        # Tr(HT H) − Tr(FT HT H F)
        # loss = tf.linalg.trace(K.dot(H_transpose, H)) - \
        #        tf.linalg.trace(K.dot(K.dot(F_transpose, H_transpose),
        #                              K.dot(H, F)))
        HTH = K.dot(H_transpose, H)
        FTHTHF= K.dot(K.dot(F_transpose, HTH), F)
        loss = tf.linalg.trace(HTH) - tf.linalg.trace(FTHTHF)

        return loss

    def _run_training(self, x, y, x_test, y_test, nb_steps,
                      seeds, verbose, log_writer, dist_matrix=None):

        if seeds is not None:
            seeds_enc = self.extract_features(seeds)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, init=seeds_enc)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        x_pred = self.extract_features(x)
        y_pred = kmeans.fit_predict(x_pred)

        if y is not None:
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            if verbose:
                print('ari kmeans: ', str(ari))
            self.log_stats_encoder(x, y, x_test, y_test, [0, 0, 0], 0, log_writer, 'init')

        # initialize cluster centers using k-means
        if verbose:
            print('Initializing F (k-means loss).')

        x_pred = self.extract_features(x)
        H = np.transpose(x_pred)
        # first on is initialized with k-means
        F = self._construct_F(H, self.n_clusters)
        if seeds is not None:
            seeds_enc = self.extract_features(seeds)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, init=seeds_enc)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(x_pred)
        F = np.zeros(shape=(self.n_clusters, len(y_pred)))
        for i in range(len(y_pred)):
            F[y_pred[i]][i] = 1.0
        F = [sub * 1 / np.sqrt(np.sum(sub)) for sub in F]
        F = np.transpose(F)

        i = 0  # Number of performed optimization steps
        epoch = 0  # Number of performed epochs

        # define the train function
        train_enc_loss = tf.keras.metrics.Mean(name='encoder train_loss')
        dec_enc_loss = tf.keras.metrics.Mean(name='dec train_loss')
        idec_enc_loss = tf.keras.metrics.Mean(name='idec train_loss')

        @tf.function
        def train_step(x_batch, f_batch, fake_batch, classif_true):
            f_batch = tf.convert_to_tensor(f_batch)
            with tf.GradientTape() as tape:
                # reconstruction loss
                encoder_loss = self.encoder_model.loss.compute_loss(x_batch, training=True)
                # k-means loss
                encoding_x = K.transpose(self.encoder(x_batch))
                kmeans_loss = self._compute_loss(encoding_x, f_batch)
                # fake/real classificationloss
                classif_pred = self.classification_model(fake_batch)
                classification_loss = self.classification_loss(classif_true, classif_pred)
                loss = encoder_loss + classification_loss + self.lamda/2 * kmeans_loss
            gradients = tape.gradient(loss,
                                      self.encoder_model.get_trainable_variables() +
                                      self.classification_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.encoder_model.get_trainable_variables() +
                                               self.classification_model.trainable_variables))

            dec_enc_loss(loss)
            idec_enc_loss(loss)

            return encoder_loss, kmeans_loss, classification_loss

        if verbose:
            print('start training')
        while i < nb_steps:
            train_enc_loss.reset_states()
            dec_enc_loss.reset_states()
            idec_enc_loss.reset_states()
            fake_x = utils.generate_fake_samples(x)
            # shuffle the train set

            # computes P each update_interval epoch
            if epoch % self.update_interval == 0 and epoch != 0:
                print('update F')
                # H = np.transpose(self.encoder.predict(x))
                F = self._construct_F(self.encoder.predict(x), self.n_clusters)

                # # evaluate the clustering performance
                # delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                # F_last = F

                # check stop criterion
                # if epoch > 0 and delta_label < tol:
                #     if verbose:
                #         print('delta_label ', delta_label, '< tol ', tol)
                #         print('Reached tolerance threshold. Stopping training.')
                #     self.log_stats_encoder(x, y, x_test, y_test, [0, 0, 0],
                #                            epoch, log_writer, 'reached_stop_criterion')
                #     break

            train_ds = tf.data.Dataset.from_tensor_slices((x, F, fake_x)) \
                .shuffle(x.shape[0], reshuffle_each_iteration=True) \
                .batch(self.batch_size).as_numpy_iterator()

            for x_batch, f_batch, fake_batch in train_ds:
                # this is the labels for real/fake loss, it is always in the same order
                classif_true = np.concatenate([np.zeros(len(x_batch)),
                                              np.ones(len(x_batch))])
                classif_true = classif_true.astype(int)
                n_values = np.max(classif_true) + 1
                classif_true = np.eye(n_values)[classif_true]

                # train_step(x_batch, f_batch, np.concatenate([x_batch, fake_batch]))
                # encoder_loss, kmeans_loss = train_step(x_batch, f_batch, np.concatenate([x_batch, fake_batch]))
                encoder_loss, kmeans_loss, classification_loss = train_step(x_batch,
                # encoder_loss, classification_loss = train_step(x_batch,
                                                                            f_batch,
                                                                            np.concatenate([x_batch, fake_batch]),
                                                                            classif_true)
                # x_pred = self.encoder.predict(x_batch)
                # fake_pred = self.encoder.predict(fake_batch)

                # x_decod = self.encoder_model.autoencoder.decoder_predict(inputs=x_batch)
                # x_decod = np.transpose(np.reshape(x_decod, (x_decod.shape[0], x_decod.shape[1])))
                # x_batch_ = np.transpose(np.reshape(x_batch, (x_batch.shape[0], x.shape[1])))

                i += 1
                if i >= nb_steps:
                    break

            if verbose:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1, idec_enc_loss.result()))
            epoch += 1

            y_pred = self.log_stats_encoder(x, y,
                                   x_test, y_test,
                                   [idec_enc_loss.result(), dec_enc_loss.result(), train_enc_loss.result()],
                                   epoch,
                                   log_writer,
                                   'pretrain')

        return epoch
