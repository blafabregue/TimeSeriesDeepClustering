"""
Define the temporal Autoencoder proposed in:
Madiraju, N. S., Sadat, S. M., Fisher, D., & Karimabadi, H. (2018).
Deep temporal clustering: Fully unsupervised learning of time-domain features

Author:
Baptiste Lafabregue 2019.25.04
"""

import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model

from networks.encoders import LayersGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class AutoEncoder(LayersGenerator):
    """docstring for AutoEncoder"""

    def __init__(self,
                 x,
                 encoder_loss,
                 n_clusters,
                 n_filters_CNN=100,
                 kernel_size=10,
                 P=10,
                 n_filters_RNN_list=[50, 50],
                 alpha=1.0,
                 nb_steps=2000):
        self.n_clusters = n_clusters
        self.L = x.shape[1]
        self.C = x.shape[2]
        self.n_filters_CNN = n_filters_CNN
        self.kernel_size = kernel_size
        self.P = P
        self.n_filters_RNN_list = n_filters_RNN_list
        self.alpha = alpha
        self.input_ = layers.Input(shape=(None, self.C))

        super().__init__(x, encoder_loss)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.1,
            decay_steps=nb_steps//4,
            decay_rate=0.1,
            staircase=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    def _create_encoder(self, x):

        with tf.name_scope('encoder'):
            h = layers.Conv1D(filters=self.n_filters_CNN,
                              kernel_size=self.kernel_size,
                              padding='same')(self.input_)
            # TODO: check alpha from leakyrelu
            h = layers.LeakyReLU()(h)
            h = layers.MaxPool1D(pool_size=self.kernel_size,
                                 strides=self.P)(h)
            for i in range(len(self.n_filters_RNN_list)):
                h = layers.Bidirectional(layers.LSTM(self.n_filters_RNN_list[i], return_sequences=True))(h)
            # h = layers.Bidirectional(layers.LSTM(self.n_filters_RNN_list[-1]))(h)
            h = layers.Flatten()(h)

        return Model(inputs=self.input_, outputs=h), None

    def _create_decoder(self, x):

        with tf.name_scope('decoder'):
            decoder_input = layers.Input(shape=(self.enc_shape[1]))
            upsampled = layers.Reshape((self.enc_shape[1], 1, 1))(decoder_input)

            upsampled = tf.image.resize(upsampled, [self.L + self.kernel_size - 1, 1])
            upsampled_shape = K.int_shape(upsampled)
            upsampled = layers.Reshape((upsampled_shape[1], upsampled_shape[2]))(upsampled)

            decode = layers.Conv1D(filters=self.C, kernel_size=self.kernel_size)(upsampled)

        return Model(inputs=decoder_input, outputs=decode), None

