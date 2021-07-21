"""
Define the Bi-directional RNN autoencoder (GRU, LSTM and RNN gates are supported)

Author:
Baptiste Lafabregue 2019.25.04
"""
import os
import numpy as np

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
                 n_filters_RNN_list=[50, 50],
                 latent_dim=None,
                 cell_type='LSTM'
                 ):
        self.n_clusters = n_clusters
        self.L = x.shape[1]
        self.C = x.shape[2]
        self.cell_type = cell_type
        assert self.cell_type in ['GRU', 'LSTM', 'RNN']
        self.n_filters_RNN_list = n_filters_RNN_list
        if latent_dim is not None:
            self.n_filters_RNN_list[-1] = latent_dim//2
        self.input_ = layers.Input(shape=(None, self.C))

        super().__init__(x, encoder_loss)

    def _create_encoder(self, x):

        with tf.name_scope('encoder'):
            h = self.input_
            for i in range(len(self.n_filters_RNN_list)-1):
                if self.cell_type == 'LSTM':
                    h = layers.Bidirectional(layers.LSTM(self.n_filters_RNN_list[i], return_sequences=True))(h)
                elif self.cell_type == 'GRU':
                    h = layers.Bidirectional(layers.GRU(self.n_filters_RNN_list[i], return_sequences=True))(h)
                elif self.cell_type == 'RNN':
                    h = layers.Bidirectional(layers.RNN(self.n_filters_RNN_list[i], return_sequences=True))(h)

            if self.cell_type == 'LSTM':
                h = layers.Bidirectional(layers.LSTM(self.n_filters_RNN_list[-1]))(h)
            elif self.cell_type == 'GRU':
                h = layers.Bidirectional(layers.GRU(self.n_filters_RNN_list[-1]))(h)
            elif self.cell_type == 'RNN':
                h = layers.Bidirectional(layers.RNN(self.n_filters_RNN_list[-1]))(h)

        return Model(inputs=self.input_, outputs=h), None

    def _create_decoder(self, x):

        with tf.name_scope('decoder'):
            decoder_input = layers.Input(shape=(self.enc_shape[1]))
            h = layers.RepeatVector(self.L)(decoder_input)
            decoder_filters = np.flip(self.n_filters_RNN_list)
            for i in range(len(decoder_filters)):
                if self.cell_type == 'LSTM':
                    h = layers.Bidirectional(layers.LSTM(decoder_filters[i], return_sequences=True))(h)
                elif self.cell_type == 'GRU':
                    h = layers.Bidirectional(layers.GRU(decoder_filters[i], return_sequences=True))(h)
                elif self.cell_type == 'RNN':
                    h = layers.Bidirectional(layers.RNN(decoder_filters[i], return_sequences=True))(h)

            decode = layers.TimeDistributed(layers.Dense(self.C))(h)

        return Model(inputs=decoder_input, outputs=decode), None

