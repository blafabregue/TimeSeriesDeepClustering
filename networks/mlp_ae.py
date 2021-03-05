"""
Define the multi layer perceptron autoencoder

Author:
Baptiste Lafabregue 2019.25.04
"""

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Dropout

from networks.encoders import LayersGenerator


class AutoEncoder(LayersGenerator):
    """docstring for AutoEncoder"""

    def __init__(self,
                 x,
                 encoder_loss,
                 layers=[500, 500, 2000],
                 latent_dim=10,
                 dropout_rate=0
                 ):
        self.layers = np.array(layers)
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        super().__init__(x, encoder_loss, support_joint_training=True)

    def _create_encoder(self, x):
        i = Input(shape=x.shape[1:])
        layers_outputs = []
        h = Flatten()(i)
        if self.dropout_rate > 0:
            h = Dropout(self.dropout_rate)(h)
        for layer in self.layers:
            h = Dense(layer)(h)
            if self.dropout_rate > 0:
                h = Dropout(self.dropout_rate)(h)
            layers_outputs.append(h)
        h = Dense(self.latent_dim)(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)

    def _create_decoder(self, x):
        layers_outputs = []
        i = Input(batch_shape=self.enc_shape)
        h = i
        for layer in np.flip(self.layers):
            h = Dense(layer)(h)
            layers_outputs.append(h)
        out_channels = 1
        for s in x.shape[1:]:
            out_channels *= s
        h = Dense(out_channels)(h)
        h = Reshape((x.shape[1:]))(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)
