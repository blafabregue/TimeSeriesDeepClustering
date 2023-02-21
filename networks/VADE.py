"""
Based on Keras implementation https://github.com/slim1017/VaDE
and article :
        Jiang, Z., Zheng, Y., Tan, H., Tang, B., & Zhou, H. (2016).
        Variational deep embedding: A generative approach to clustering

Author:
Baptiste Lafabregue 2019.25.04
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

from networks.trainer import Trainer
import utils


class VADE(Trainer):
    def __init__(self,
                 dataset_name,
                 classifier_name,
                 encoder_model,
                 decoder,
                 keep_both_losses=True,
                 n_clusters=10,
                 alpha=1.0,
                 batch_size=10,
                 optimizer=None):

        super(VADE, self).__init__(dataset_name, classifier_name, encoder_model, batch_size, n_clusters, optimizer)

        self.decoder = decoder
        if decoder is None:
            raise utils.CompatibilityException('architecture incompatible with VADE')
        self.keep_both_losses = keep_both_losses
        self.alpha = alpha
        self.n_clusters = n_clusters
        enc_shape = np.array(K.int_shape(encoder_model.encoder.output)).tolist()
        self.latent_dim = enc_shape[1] // 2
        self.theta_p = K.variable(np.ones(n_clusters) / n_clusters, name="pi")
        self.u_p = K.variable(np.zeros((self.latent_dim, n_clusters)), name="mu")
        self.lambda_p = K.variable(np.ones((self.latent_dim, n_clusters)), name="lambda")
        self.p_c_z_output = None

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

        gamma_layer = Lambda(self.get_gamma, output_shape=(self.n_clusters,))(self.encoder.output)
        self.p_c_z_output = Model(inputs=self.encoder.input, outputs=gamma_layer)

    def predict_clusters(self, x, seeds=None):
        """
        Predict cluster labels using the output of clustering layer
        :param x: the data to evaluate
        :param seeds: seeds to initialize the K-Means if needed
        :return: the predicted cluster labels
        """
        center_shape = list(x.shape)
        center_shape[0] = self.n_clusters
        features = self.p_c_z_output.predict(x, batch_size=1)
        return np.argmax(features, axis=1), np.zeros(tuple(center_shape))

    def get_gamma(self, z):
        mean, logvar = tf.split(z, num_or_size_splits=2, axis=1)
        mean_c = K.permute_dimensions(K.repeat(mean, self.n_clusters), (0, 2, 1))
        size_z = 1

        # do the same for parameters but for batch size
        u_t = K.repeat_elements(K.expand_dims(self.u_p, axis=0), size_z, 0)
        lambda_t = K.repeat_elements(K.expand_dims(self.lambda_p, axis=0), size_z, 0)
        theta_t = K.expand_dims(K.expand_dims(self.theta_p, axis=0), axis=0)
        theta_t = tf.ones((size_z, self.latent_dim, self.n_clusters), dtype=K.floatx()) * theta_t

        temp_p_c_z = K.exp(K.sum((K.log(theta_t) - 0.5 * K.log(2 * np.pi * lambda_t) -
                                  K.square(mean_c - u_t) / (2 * lambda_t)), axis=1))
        return temp_p_c_z / K.sum(temp_p_c_z, axis=-1, keepdims=True)

    def init_gmm_parameters(self, x):
        g = GaussianMixture(n_components=self.n_clusters, covariance_type='diag')
        sample = self.encoder.predict(x)
        sample = np.split(sample, 2, axis=1)[0]
        g.fit(sample)
        self.u_p.assign(g.means_.T)
        self.lambda_p.assign(g.covariances_.T)

    def _run_training(self, x, y, x_test, y_test, nb_steps,
                      seeds, verbose, log_writer, dist_matrix=None):

        i = 0  # Number of performed optimization steps
        epoch = 0  # Number of performed epochs
        self.init_gmm_parameters(x)

        # define the train function
        vae_loss = tf.keras.metrics.Mean(name='encoder train_loss')

        # @tf.function
        def train_step(batch, batch_size):
            with tf.GradientTape() as tape:
                mean, logvar = tf.split(self.encoder(batch), num_or_size_splits=2, axis=1)
                epsilon = K.random_normal(shape=mean.shape)
                z = mean + K.exp(logvar / 2) * epsilon
                # repeat each element per # of clusters
                mean_c = K.permute_dimensions(K.repeat(mean, self.n_clusters), (0, 2, 1))
                logvar_c = K.permute_dimensions(K.repeat(logvar, self.n_clusters), (0, 2, 1))
                z_c = K.permute_dimensions(K.repeat(z, self.n_clusters), (0, 2, 1))

                # do the same for parameters but for batch size
                u_t = K.repeat_elements(K.expand_dims(self.u_p, axis=0), self.batch_size, 0)
                lambda_t = K.repeat_elements(K.expand_dims(self.lambda_p, axis=0), self.batch_size, 0)
                theta_t = K.expand_dims(K.expand_dims(self.theta_p, axis=0), axis=0)
                theta_t = tf.ones((self.batch_size, self.latent_dim, self.n_clusters), dtype=K.floatx()) * theta_t

                x_logit = self.decoder(z, training=True)
                recon = self.alpha * K.sum(tf.keras.losses.mean_squared_error(batch, x_logit), axis=1)  # * original_dim

                p_c_z = K.exp(K.sum((K.log(theta_t) - 0.5 * K.log(2 * np.pi * lambda_t) -
                                     K.square(z_c - u_t) / (2 * lambda_t)), axis=1)) + 1e-10
                gamma = p_c_z / K.sum(p_c_z, axis=-1, keepdims=True)

                gamma_t = K.repeat(gamma, self.latent_dim)

                # print(gamma_t)
                # print(theta_t)
                # print(lambda_t)
                # print(u_t)
                # print(gamma)
                # print(logvar_c)
                # print((mean_c))
                a = K.sum(0.5 * gamma_t * (tf.cast(K.log(np.pi * 2), dtype=K.floatx()) +
                                           K.log(lambda_t) + K.exp(logvar_c) / lambda_t +
                                           K.square(mean_c - u_t) / lambda_t), axis=(1, 2))
                b = 0.5 * K.sum(logvar + 1, axis=-1)
                c = K.sum(K.log(K.repeat_elements(K.expand_dims(self.theta_p, axis=0), batch_size, 0)) * gamma, axis=-1)
                d = K.sum(K.log(gamma) * gamma, axis=-1)
                loss = recon + a - b - c + d
                # print(a)
                # print(b)
                # print(c)
                # print(d)
                # print(recon)
            trainable_var = self.encoder_model.get_trainable_variables() + [self.lambda_p, self.u_p, self.theta_p]
            gradients = tape.gradient(loss, trainable_var)
            self.optimizer.apply_gradients(zip(gradients, trainable_var))

            vae_loss(loss)

        if verbose:
            print('start training')

        while i < nb_steps:
            vae_loss.reset_states()
            # shuffle the train set
            train_ds = tf.data.Dataset.from_tensor_slices(x) \
                .shuffle(x.shape[0], reshuffle_each_iteration=True) \
                .batch(self.batch_size, drop_remainder=True).as_numpy_iterator()

            for x_batch in train_ds:
                train_step(x_batch, len(x_batch))
                i += 1
                if i >= nb_steps:
                    break

            if verbose:
                template = 'Epoch {}, Loss: {}'
                print(template.format(epoch + 1, vae_loss.result()))
            epoch += 1

            y_pred = self.log_stats(x, y,
                                    x_test, y_test,
                                    [vae_loss.result(), vae_loss.result(), vae_loss.result()],
                                    epoch,
                                    log_writer,
                                    'train')

        return epoch
