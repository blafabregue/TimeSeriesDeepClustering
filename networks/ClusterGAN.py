"""
Based on tensorflow implementation https://github.com/sudiptodip15/ClusterGAN
and article :
        Sudipto Mukherjee, Himanshu Asnani, Eugene Lin and Sreeram Kannan,
        ClusterGAN : Latent Space Clustering in Generative Adversarial Networks

Author:
Baptiste Lafabregue 2019.25.04
"""

import numpy as np
import os
import csv
import copy

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Activation

from networks.trainer import Trainer
import utils


def sample_z(batch, z_dim, sampler='one_hot', num_class=10, n_cat=1, label_index=None):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim - num_class * n_cat),
                          np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim - num_class), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return 0.15 * np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low=0, high=num_class, size=batch)
        return 0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index]


class ClusterGAN(Trainer):
    def __init__(self,
                 dataset_name,
                 classifier_name,
                 encoder,
                 generator,
                 discriminator,
                 n_clusters=10,
                 batch_size=10,
                 beta_cycle_gen=2,
                 beta_cycle_label=2,
                 optimizer=None):
        super(ClusterGAN, self).__init__(dataset_name, classifier_name, None, batch_size, n_clusters, optimizer)

        self.encoder = encoder
        self.discriminator = discriminator
        self.generator = generator
        self.beta_cycle_gen = beta_cycle_gen
        self.beta_cycle_label = beta_cycle_label
        self.allow_pretrain = False
        if optimizer is None:
            optimizer = tf.keras.optimizers.legacy.Adam(lr=1e-4)
        self.optimizer_discriminator = copy.deepcopy(optimizer)
        self.optimizer_generator = copy.deepcopy(optimizer)
        self.dim_gen = 0
        self.zdim = 0
        self.scale = 10

    def initialize_model(self, x, y, ae_weights=None):
        """
        Initialize the model for training
        :param ae_weights: arguments to let the encoder load its weights, None to pre-train the encoder
        """
        self.pretrain_model = False
        input_shape = K.int_shape(self.encoder.input)[1:]
        enc_input = Input(shape=input_shape)
        enc_output = self.encoder(enc_input)
        enc_shape = K.int_shape(self.encoder.output)
        self.dim_gen = enc_shape[1] - self.n_clusters
        self.zdim = enc_shape[1]
        self.xdim = x.shape[1] * x.shape[2]

        disc_input = Input(shape=input_shape)
        disc_output = self.discriminator(disc_input)
        disc_output = Dense(1)(disc_output)
        disc_output = Activation('sigmoid')(disc_output)
        self.discriminator = Model(inputs=disc_input, outputs=disc_output)

        logits = enc_output[:, self.dim_gen:]
        y = tf.nn.softmax(logits)
        self.encoder = Model(inputs=enc_input, outputs=[enc_output[:, 0:self.dim_gen], y, logits])

    def load_weights(self, weights_path):
        """
        Load weights of IDEC model
        :param weights_path: path to load weights from
        """

    def save_weights(self, weights_path):
        """
        Save weights of IDEC model
        :param weights_path: path to save weights to
        """

    def extract_features(self, x):
        """
        Extract features from the encoder (before the clustering layer)
        :param x: the data to extract features from
        :return: the encoded features
        """
        return self.encoder.predict(x)[0]

    def reconstruct_features(self, x):
        """
        Reconstruct features from the autoencoder (encode and decode)
        :param x: the data to reconstruct features from
        :return: the reconstructed features, None if not supported
        """
        return None

    def _run_training(self, x, y, x_test, y_test, nb_steps,
                      seeds, verbose, log_writer, dist_matrix=None):
        discriminator_steps = 4

        i = 0  # Number of performed optimization steps
        epoch = 0  # Number of performed epochs

        # define the train function
        discriminator_loss = tf.keras.metrics.Mean(name='discriminator train_loss')
        generator_loss = tf.keras.metrics.Mean(name='generator train_loss')

        @tf.function
        def train_discriminator(x_batch, y_real, y_fake):
            with tf.GradientTape() as tape:
                x_batch_size = K.int_shape(x_batch)[0]
                z = sample_z(x_batch_size, self.zdim, num_class=self.n_clusters)
                x_gen = self.generator(z)

                disc_pred = self.discriminator(x_batch)
                disc_pred_gen = self.discriminator(x_gen)
                real_loss = tf.keras.losses.binary_crossentropy(y_real, disc_pred)
                fake_loss = tf.keras.losses.binary_crossentropy(y_fake, disc_pred_gen)
                loss = real_loss + fake_loss

            gradients = tape.gradient(loss, self.discriminator.trainable_variables)
            self.optimizer_discriminator.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

            discriminator_loss(loss)

        @tf.function
        def train_generator():
            with tf.GradientTape() as tape:
                z = sample_z(self.batch_size, self.zdim, num_class=self.n_clusters)
                z_gen = z[:, 0:self.dim_gen]
                z_hot = z[:, self.dim_gen:]
                x_gen = self.generator(z)

                z_enc_gen, z_enc_label, z_enc_logits = self.encoder(x_gen)
                disc_pred_gen = self.discriminator(x_gen)

                # v_loss = tf.reduce_mean(disc_pred_gen)
                dtype = tf.float32
                if K.floatx() == 'float64':
                    dtype = tf.float64
                v_loss = tf.keras.losses.binary_crossentropy(disc_pred_gen, tf.ones((self.batch_size, 1), dtype=dtype))
                a = self.beta_cycle_gen * tf.reduce_mean(tf.square(z_gen - z_enc_gen))
                b = self.beta_cycle_label * tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=z_enc_logits, labels=z_hot))
                loss = v_loss + a + b

            gradients = tape.gradient(loss, self.encoder.trainable_variables + self.generator.trainable_variables)
            self.optimizer_generator.apply_gradients(
                zip(gradients, self.encoder.trainable_variables + self.generator.trainable_variables))

            generator_loss(loss)

        if verbose:
            print('start training')

        while i < nb_steps:
            discriminator_loss.reset_states()
            generator_loss.reset_states()

            train_ds = tf.data.Dataset.from_tensor_slices(x) \
                .shuffle(x.shape[0], reshuffle_each_iteration=True) \
                .batch(self.batch_size).as_numpy_iterator()
            sub_i = 0
            # Train discriminator
            for x_batch in train_ds:
                x_batch_size = K.int_shape(x_batch)[0]
                z = sample_z(x_batch_size, self.zdim, num_class=self.n_clusters)
                x_gen = self.generator(z)

                # disc_pred = self.discriminator(x_batch)
                # disc_pred_gen = self.discriminator(x_gen)
                train_discriminator(x_batch, np.ones(shape=(x_batch.shape[0], 1)),
                                    np.zeros(shape=(x_batch.shape[0], 1)))
                sub_i += 1
                if sub_i >= discriminator_steps:
                    break

            # Train generator
            train_generator()
            z = sample_z(self.batch_size, self.zdim, num_class=self.n_clusters)
            z_gen = z[:, 0:self.dim_gen]
            z_hot = z[:, self.dim_gen:]
            x_gen = self.generator(z)

            z_enc_gen, z_enc_label, z_enc_logits = self.encoder(x_gen)
            disc_pred_gen = self.discriminator(x_gen)

            a = tf.reduce_mean(disc_pred_gen) + self.beta_cycle_gen * tf.reduce_mean(tf.square(z_gen - z_enc_gen))
            b = self.beta_cycle_label * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=z_enc_logits, labels=z_hot))
            z_enc_gen = z_enc_gen.numpy()
            z_enc_logits = z_enc_logits.numpy()
            z_enc_label = z_enc_label.numpy()
            x_gen = np.reshape(x_gen.numpy(), (self.batch_size, -1))
            loss = a + b
            epoch += 1

            i += 1
            if i >= nb_steps:
                break

            if verbose:
                template = 'Epoch {}, Loss discriminator : {}, Loss generator : {}'
                print(template.format(epoch + 1, discriminator_loss.result(), generator_loss.result()))

            y_pred = self.log_stats(x, y,
                                    x_test, y_test,
                                    [discriminator_loss.result() + generator_loss.result(),
                                     discriminator_loss.result(), generator_loss.result()],
                                    epoch,
                                    log_writer,
                                    'train')

        return epoch
