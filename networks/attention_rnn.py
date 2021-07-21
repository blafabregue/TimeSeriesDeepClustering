"""
Define a Bi-directional GRU autoencoder with attention mechanism.
Based on implementation https://gitlab.irstea.fr/dino.ienco/detsec
and article:
    Dino Ienco, Roberto Interdonato.
    Deep Multivariate Time Series Embedding Clustering via Attentive-Gated Autoencoder. PAKDD 2020: 318-329

Author:
Baptiste Lafabregue 2019.25.04
"""
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, InputSpec

from networks.encoders import LayersGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class AttentionLayer(Layer):
    """
    AttentionLayer apply a self-attention mechanism to abidirectionel RNN output

    # Arguments
        n_units : number of units returned by the previous (RNN) layer
        att_size : size of attention mechanism weights
    # Input shape
        (<batch_size>, <Time_steps>, nunits)
    # Output shape
        (<batch_size>, att_size)
    """

    def __init__(self, n_units, att_size, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AttentionLayer, self).__init__(**kwargs)
        self.n_units = n_units
        self.att_size = att_size
        # self.input_spec = InputSpec(ndim=2)
        self.W_omega = None
        self.b_omega = None
        self.u_omega = None

    def build(self, input_shape):
        # input_dim = input_shape[1]
        initializer = tf.keras.initializers.RandomNormal(stddev=0.1)
        # self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.W_omega = self.add_weight(shape=(self.n_units, self.att_size),
                                       initializer=initializer, name="W_omega")
        self.b_omega = self.add_weight(shape=(self.att_size,),
                                       initializer=initializer, name="b_omega")
        self.u_omega = self.add_weight(shape=(self.att_size,),
                                       initializer=initializer, name="u_omega")
        self.built = True

    def call(self, input, **kwargs):
        # outputs = tf.stack(input_list, axis=1)

        v = tf.tanh(tf.tensordot(input, self.W_omega, axes=1) + self.b_omega)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, self.u_omega, axes=1)  # (B,T) shape
        alphas = tf.nn.softmax(vu)  # (B,T) shape also

        output = tf.reduce_sum(input * tf.expand_dims(alphas, -1), 1)
        output = tf.reshape(output, [-1, self.n_units])
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_units

    def get_config(self):
        config = {'n_units': self.n_units, 'att_size': self.att_size}
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AutoEncoder(LayersGenerator):
    """docstring for AutoEncoder"""

    def __init__(self,
                 x,
                 encoder_loss,
                 n_clusters,
                 latent_dim=None,
                 ):
        self.n_clusters = n_clusters
        self.L = x.shape[1]
        self.C = x.shape[2]
        if latent_dim is None:
            # we fix the rnn size based on train set size
            self.n_filters_RNN = 64
            if x.shape[0] > 250:
                self.n_filters_RNN = 512
        else:
            self.n_filters_RNN = latent_dim

        self.input_ = layers.Input(shape=(None, self.C))
        self.gate = layers.Dense(self.n_filters_RNN, activation='sigmoid')

        super().__init__(x, encoder_loss)

    def _create_encoder(self, x):

        with tf.name_scope('encoder'):
            h = self.input_
            forward_layer = layers.GRU(self.n_filters_RNN, return_sequences=True)(h)
            backward_layer = layers.GRU(self.n_filters_RNN, activation='relu', return_sequences=True,
                                        go_backwards=True)(h)

            h_att_fw = AttentionLayer(self.n_filters_RNN, self.n_filters_RNN)(forward_layer)
            h_att_bw = AttentionLayer(self.n_filters_RNN, self.n_filters_RNN)(backward_layer)

            h = self.gate(h_att_fw) * h_att_fw + self.gate(h_att_bw) * h_att_bw
        return Model(inputs=self.input_, outputs=h), None

    def _create_decoder(self, x):

        with tf.name_scope('decoder'):
            decoder_input = layers.Input(shape=(self.enc_shape[1]))
            h = layers.RepeatVector(self.L)(decoder_input)
            h = layers.Bidirectional(layers.GRU(self.n_filters_RNN // 2, return_sequences=True))(h)

            decode = layers.TimeDistributed(layers.Dense(self.C))(h)

        return Model(inputs=decoder_input, outputs=decode), None
