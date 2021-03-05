"""
Define the ReseNet autoencoder.
Based on keras Hassan Fawaz implementation https://github.com/hfawaz/dl-4-tsc
and on article:
Wang, Z., Yan, W., & Oates, T. (2016).
Time series classification from scratch with deep neural networks: a strong baseline

Author:
Baptiste Lafabregue 2019.25.04
"""
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Reshape, Activation, Dropout, add
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D

from networks.encoders import LayersGenerator


class AutoEncoder(LayersGenerator):
    """docstring for AutoEncoder"""

    def __init__(self,
                 x,
                 encoder_loss,
                 filters=[64, 128, 128],
                 kernels=[8, 5, 3],
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

    def _create_res_block(self, i, n_features):
        conv_x = Conv1D(filters=n_features, kernel_size=int(self.kernels[0]), padding='same')(i)
        conv_x = BatchNormalization()(conv_x)
        conv_x = Activation('relu')(conv_x)

        conv_y = Conv1D(filters=n_features, kernel_size=int(self.kernels[1]), padding='same')(conv_x)
        conv_y = BatchNormalization()(conv_y)
        conv_y = Activation('relu')(conv_y)

        conv_z = Conv1D(filters=n_features, kernel_size=int(self.kernels[2]), padding='same')(conv_y)
        conv_z = BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = Conv1D(filters=n_features, kernel_size=1, padding='same')(i)
        shortcut_y = BatchNormalization()(shortcut_y)

        output_block_1 = add([shortcut_y, conv_z])
        output_block_1 = Activation('relu')(output_block_1)

        return output_block_1

    def _create_encoder(self, x):
        i = Input(shape=x.shape[1:])
        layers_outputs = []
        h = i

        for n_features in self.filters:
            h = self._create_res_block(h, n_features)
            if self.dropout_rate > 0:
                h = Dropout(self.dropout_rate)(h)
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
        for n_features in np.flip(self.filters):
            h = self._create_res_block(h, n_features)
            layers_outputs.append(h)
        h = Conv1D(filters=x.shape[-1], kernel_size=int(self.kernels[-1]), padding='same', strides=1)(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)
