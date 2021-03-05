"""
Define autoencoder abstract classes

Author:
Baptiste Lafabregue 2019.25.04
"""

import tensorflow.keras.backend as K
from tensorflow.keras import Model
import tensorflow as tf

import numpy as np
import os


class EncoderWrapper(object):
    def __init__(self, encoder_clear, encoder_noisy):
        self.encoder_clear = encoder_clear
        self.encoder_noisy = encoder_noisy

    def __call__(self, inputs, clear=True, *args, **kwargs):
        if clear:
            return self.encoder_clear(inputs, *args, **kwargs)
        return self.encoder_noisy(inputs, *args, **kwargs)

    def predict(self, inputs, *args, **kwargs):
        return self.encoder_clear.predict(inputs, *args, **kwargs)


class EncoderModel(object):
    def __init__(self, encoder, autoencoder, optimizer, loss, name, batch_size=10, nb_steps=100):
        self.encoder = encoder
        self.autoencoder = autoencoder
        self.optimize = optimizer
        self.loss = loss
        self.name = name
        self.batch_size = batch_size,
        self.nb_steps = nb_steps

    def load_weights(self, paths):
        if self.autoencoder is not None:
            self.autoencoder.load_weights(paths+self.name)
        else:
            self.encoder.load_weights(paths+self.name+'_encoder.h5')

    def save_weights(self, paths):
        if self.autoencoder is not None:
            self.autoencoder.save_weights(paths+self.name)
        else:
            self.encoder.save_weights(paths+self.name+'_encoder.h5')

    def exists(self, paths):
        if self.autoencoder is not None:
            return self.autoencoder.exists(paths+self.name)
        return os.path.exists(paths+self.name+'_encoder.h5')

    def get_trainable_variables(self):
        if self.autoencoder is not None:
            trainable_variables = self.autoencoder.get_trainable_variables()
        else:
            trainable_variables = self.encoder.trainable_variables
        return trainable_variables

    def get_name(self):
        return self.name

    def summary(self):
        if self.autoencoder is not None:
            self.autoencoder.summary()
        else:
            self.encoder.summary()


class LayersGenerator(object):
    def __init__(self, x, encoder_loss, support_joint_training=False):
        self.encoder_loss = encoder_loss
        self.optimizer = None
        self.encoder, self.all_layers_encoder = self._create_encoder(x)
        self.discriminator, _ = self._create_encoder(x)
        self.enc_shape = np.array(K.int_shape(self.encoder.output)).tolist()
        if encoder_loss == 'vae':
            self.enc_shape[1] //= 2
        self.decoder, self.all_layers_decoder = self._create_decoder(x)
        self.autoencoder = self._create_autoencoder()
        self.support_joint_training = support_joint_training

    def get_encoder(self):
        return self.encoder

    def get_all_layers_encoder(self):
        return self.all_layers_encoder

    def get_all_layers_decoder(self):
        return self.all_layers_decoder

    def get_decoder(self):
        return self.decoder

    def get_discriminator(self):
        return self.discriminator

    def get_auto_encoder(self):
        return self.autoencoder

    def get_optimizer(self):
        return self.optimizer

    def support_joint_training(self):
        return self.support_joint_training

    def _create_encoder(self, x) -> (Model, Model):
        pass

    def _create_decoder(self, x) -> (Model, Model):
        pass

    def _create_autoencoder(self):
        return AutoencoderModel(self.encoder, self.decoder)


class AutoencoderModel(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.use_vae = False

    def __call__(self, inputs, *args, **kwargs):
        return self.decoder(self.encoder(inputs))

    def decoder_predict(self, inputs=None, encoding=None):
        assert(inputs is not None or encoding is not None)
        if encoding is not None:
            return self.decoder.predict(encoding)
        encoded = self.encoder.predict(inputs)
        if self.use_vae:
            mean, logvar = tf.split(encoded, num_or_size_splits=2, axis=1)
            epsilon = K.random_normal(shape=mean.shape)
            encoded = mean + K.exp(logvar / 2) * epsilon
        return self.decoder.predict(encoded)

    def get_trainable_variables(self):
        trainable_variables = self.encoder.trainable_variables
        if self.decoder is not None:
            trainable_variables += self.decoder.trainable_variables
        return trainable_variables

    def save_weights(self, paths):
        self.encoder.save_weights(paths+'_encoder.h5')
        if self.decoder is not None:
            self.decoder.save_weights(paths+'_decoder.h5')

    def load_weights(self, paths):
        self.encoder.load_weights(paths+'_encoder.h5')
        if self.decoder is not None:
            self.decoder.load_weights(paths+'_decoder.h5')

    def exists(self, paths):
        decoder_exists = True
        if self.decoder is not None:
            decoder_exists = os.path.exists(paths+'_decoder.h5')
        encoder_exists = os.path.exists(paths+'_encoder.h5')
        return decoder_exists and encoder_exists

    def summary(self):
        self.encoder.summary()
        if self.decoder is not None:
            self.decoder.summary()

    def set_use_vae(self, use_vae):
        self.use_vae = use_vae


class RnnAutoencoderModel(AutoencoderModel):
    def __init__(self, encoder, decoder):
        super(RnnAutoencoderModel, self).__init__(encoder, decoder)

    def __call__(self, inputs, *args, **kwargs):
        return self.decoder([inputs, np.zeros((inputs.shape[0], 1, inputs.shape[2]))], *args, **kwargs)

    def decoder_predict(self, inputs=None, encoding=None):
        return self.decoder.predict([inputs, np.zeros((inputs.shape[0], 1, inputs.shape[2]))])

    def get_trainable_variables(self):
        return self.decoder.trainable_variables

    def save_weights(self, paths):
        self.decoder.save_weights(paths+'_autoencoder.tf')

    def load_weights(self, paths):
        self.decoder.load_weights(paths+'_autoencoder.tf')

    def exists(self, paths):
        return os.path.exists(paths+'_autoencoder.tf')

    def summary(self):
        self.decoder.summary()
