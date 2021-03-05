"""
Based on torch implementation https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
and article:
Franceschi, J. Y., Dieuleveut, A., & Jaggi, M. (2019).
Unsupervised scalable representation learning for multivariate time series

Author:
Baptiste Lafabregue 2019.25.04
"""
import inspect
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Dense, Input, Reshape, Dropout
from tensorflow.keras.layers import GlobalMaxPool1D, Layer, LeakyReLU
from tensorflow.keras import initializers

import tensorflow_addons as tfa

from networks.encoders import LayersGenerator


def compute_adaptive_dilations(time_size):
    last_dilations=1
    dilations = [last_dilations]
    rate = 4
    if time_size < 50:
        rate = 2

    while True:
        if last_dilations > time_size/2:
            break
        last_dilations *= rate
        dilations.append(last_dilations)

    return dilations


class CausalResidualBlock(Layer):

    def __init__(self,
                 nb_filters,
                 kernel_size,
                 dilation_rate,
                 final=False,
                 kernel_initializer=None,
                 **kwargs):
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        if self.kernel_initializer is None:
            self.kernel_initializer = initializers.VarianceScaling((1.0/3), distribution="uniform")
        self.final = final
        self.layers_outputs = []
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(CausalResidualBlock, self).__init__(**kwargs)

    def build(self, input_shape):

        self.final = input_shape[-1] != self.nb_filters
        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = 'conv1D_{}'.format(k)
                with K.name_scope(name):
                    self._add_and_activate_layer(
                        tfa.layers.WeightNormalization(
                            Conv1D(filters=self.nb_filters,
                                   kernel_size=self.kernel_size,
                                   dilation_rate=self.dilation_rate,
                                   kernel_initializer=self.kernel_initializer,
                                   bias_initializer=self.kernel_initializer,
                                   padding='causal',
                                   name=name)
                            , data_init=False
                        )
                    )
                self._add_and_activate_layer(LeakyReLU(alpha=0.01))

            if self.final:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'conv1D_3'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   kernel_initializer=self.kernel_initializer,
                                                   padding='same',
                                                   name=name)
                # else:
                #     self.shape_match_conv = Lambda(lambda x: x, name='identity')
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            # if self.final:
            #     name = 'activ'
            #     with K.name_scope(name):
            #         self.final_activation = Activation('relu')
            #         # self.final_activation = Activation(LeakyReLU(alpha=0.01))
            #         self.final_activation.build(self.res_output_shape)  # probably isn't necessary
            # else:
            #     self.final_activation = None

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)

            super(CausalResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        self.layers_outputs = [x]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
            self.layers_outputs.append(x)

        if self.final:
            x2 = self.shape_match_conv(inputs)
        else:
            x2 = inputs
        self.layers_outputs.append(x2)
        res_x = layers.add([x2, x])
        self.layers_outputs.append(res_x)

        if self.final_activation is not None:
            res_act_x = self.final_activation(res_x)
            self.layers_outputs.append(res_act_x)
        else:
            res_act_x = res_x
        return res_act_x

    def compute_output_shape(self, input_shape):
        return self.res_output_shape

    def get_config(self):
        config = super(CausalResidualBlock, self).get_config()
        config.update({'dilation_rate': self.dilation_rate})
        config.update({'nb_filters': self.nb_filters})
        config.update({'kernel_size': self.kernel_size})
        config.update({'kernel_initializer': self.kernel_initializer})
        config.update({'final': self.final})
        return config


class AutoEncoder(LayersGenerator):
    """docstring for AutoEncoder"""

    def __init__(self,
                 x,
                 encoder_loss,
                 nb_filters=10,
                 depth=1,
                 reduced_size=10,
                 latent_dim=10,
                 kernel_size=4,
                 dilations=None,
                 dropout_rate=0
                 ):
        self.nb_filters = nb_filters
        self.reduced_size = reduced_size
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dilation_depth = depth
        self.dropout_rate = dropout_rate
        if dilations is None:
            self.dilations = [2**i for i in range(depth)]
        self.dilation_depth = len(self.dilations)

        super().__init__(x, encoder_loss, support_joint_training=True)

    def _create_encoder(self, x):

        i = Input(batch_shape=(None, None, x.shape[2]))
        dilation_size = self.dilations[0]
        layers_outputs = []

        h = i
        if self.dropout_rate > 0:
            h = Dropout(self.dropout_rate)(h)
        for k in range(self.dilation_depth):
            h = CausalResidualBlock(
                nb_filters=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_size,
                name='residual_block_{}'.format(k)
            )(h)
            layers_outputs.append(h)
            # build newest residual block
            dilation_size = self.dilations[k]

        # Last layer
        h = CausalResidualBlock(
            nb_filters=self.reduced_size,
            kernel_size=self.kernel_size,
            dilation_rate=dilation_size,
            name='residual_block_{}'.format(self.dilation_depth)
        )(h)

        h = GlobalMaxPool1D()(h)
        if self.dropout_rate > 0:
            h = Dropout(self.dropout_rate)(h)
        init = initializers.VarianceScaling((1.0 / 3), distribution="uniform")
        h = Dense(self.latent_dim, kernel_initializer=init, bias_initializer=init)(h)
        if x.shape[2]:
            h = layers.Activation('sigmoid')(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)

    def _create_decoder(self, x):

        # Initial dilation size
        # we invert the order of dilations
        dilation_size = self.dilations[-1]
        dense_size = x.shape[1]*x.shape[2]
        layers_outputs = []

        i = Input(shape=(self.enc_shape[1],))
        h = Dense(dense_size)(i)
        h = Reshape((x.shape[1], x.shape[2]))(h)
        layers_outputs.append(h)
        for k in range(self.dilation_depth):
            h = CausalResidualBlock(
                nb_filters=self.nb_filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_size,
                name='decoder_residual_block_{}'.format(k)
            )(h)
            layers_outputs.append(h)
            dilation_size = self.dilations[-(k+1)]

        h = CausalResidualBlock(
            nb_filters=x.shape[2],
            kernel_size=self.kernel_size,
            dilation_rate=dilation_size,
            name='residual_block_{}'.format(self.dilation_depth)
        )(h)
        layers_outputs.append(h)

        return Model(inputs=i, outputs=h), Model(inputs=i, outputs=layers_outputs)
