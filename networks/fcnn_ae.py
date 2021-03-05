"""
Define the fully convolutional autoencoder

Author:
Baptiste Lafabregue 2019.25.04
"""

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Activation, Dropout
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D

from networks.encoders import LayersGenerator


class AutoEncoder(LayersGenerator):
    """docstring for AutoEncoder"""

    def __init__(self,
                 x,
                 encoder_loss,
                 filters=[64, 64, 64],
                 kernels=[3, 5, 9],
                 latent_dim=10,
                 activation='relu',
                 dropout_rate=0
                 ):
        self.filters = np.array(filters)
        self.kernels = np.array(kernels)
        self.latent_dim = latent_dim
        self.activation = activation
        self.dropout_rate = dropout_rate

        super().__init__(x, encoder_loss, support_joint_training=True)

    def _create_encoder(self, x):
        i = Input(shape=x.shape[1:])
        layers_outputs = []
        h = i
        if self.dropout_rate > 0:
            h = Dropout(self.dropout_rate)(h)
        for filter, kernel in zip(self.filters, self.kernels):
            h = Conv1D(filters=filter, kernel_size=int(kernel), padding='same', strides=1)(h)
            if self.dropout_rate > 0:
                h = Dropout(self.dropout_rate)(h)
            h = BatchNormalization()(h)
            h = Activation(activation=self.activation)(h)
            layers_outputs.append(h)
        h = GlobalAveragePooling1D()(h)
        h = Dense(self.latent_dim)(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)

    def _create_decoder(self, x):
        out_channels = 1
        layers_outputs = []

        for s in x.shape[1:]:
            out_channels *= s
        i = Input(batch_shape=self.enc_shape)
        h = Dense(out_channels)(i)
        h = Reshape((x.shape[1:]))(h)
        layers_outputs.append(h)
        for filter, kernel in zip(np.flip(self.filters), np.flip(self.kernels)):
            h = Conv1D(filters=filter, kernel_size=int(kernel), padding='same', strides=1)(h)
            h = Activation(activation=self.activation)(h)
            layers_outputs.append(h)
        h = Conv1D(filters=x.shape[-1], kernel_size=int(self.kernels[-1]), padding='same', strides=1)(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)
