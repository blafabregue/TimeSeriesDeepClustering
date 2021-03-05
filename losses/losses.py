"""
Define loss used in the experiment
The triplet loss is based on torch implementation https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
and article:
Franceschi, J. Y., Dieuleveut, A., & Jaggi, M. (2019).
Unsupervised scalable representation learning for multivariate time series

Author:
Baptiste Lafabregue 2019.25.04
"""
import numpy as np

from tensorflow.keras import backend as K
import tensorflow as tf

import utils


class TripletLoss(object):
    def __init__(self, encoder, train_set, compared_length,
                 nb_random_samples, negative_penalty, fixed_time_dim=False):
        self.encoder = encoder
        self.train_set = train_set
        self.compared_length = compared_length
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.fixed_time_dim = fixed_time_dim

    def compute_loss(self, batch, noisy_batch=None, training=True):
        batch_size = batch.shape[0]
        train_size = self.train_set.shape[0]
        length = min(self.compared_length, self.train_set.shape[1])
        fixed_length = self.train_set.shape[1]

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = np.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )

        # Choice of length of positive and negative samples
        length_pos_neg = np.random.randint(1, high=length + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = np.random.randint(
            length_pos_neg, high=length + 1
        )  # Length of anchors
        beginning_batches = np.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = np.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = np.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )
        anchor = K.concatenate([batch[j: j + 1,
                                beginning_batches[j]: beginning_batches[j] + random_length,
                                :,
                                ] for j in range(batch_size)], axis=0)
        if self.fixed_time_dim:
            anchor = tf.pad(anchor, tf.constant([[0, 0], [0, fixed_length-random_length], [0, 0]]))

        representation = self.encoder(
            anchor,
            training=training
        )  # Anchors representations

        positive = K.concatenate([batch[j: j + 1,
                                  end_positive[j] - length_pos_neg: end_positive[j],
                                  :,
                                  ] for j in range(batch_size)], axis=0)
        if self.fixed_time_dim:
            positive = tf.pad(positive, tf.constant([[0, 0], [0, fixed_length-length_pos_neg], [0, 0]]))
        positive_representation = self.encoder(
            positive,
            training=training
        )  # Positive samples representations

        size_representation = K.int_shape(representation)[1]
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -K.mean(K.log(K.sigmoid(K.batch_dot(
            K.reshape(representation, (batch_size, 1, size_representation)),
            K.reshape(positive_representation, (batch_size, size_representation, 1)))
        )))

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative = K.concatenate([self.train_set[samples[i, j]: samples[i, j] + 1]
                                      [:,
                                      beginning_samples_neg[i, j]:beginning_samples_neg[i, j] + length_pos_neg,
                                      :] for j in range(batch_size)], axis=0)
            if self.fixed_time_dim:
                negative = tf.pad(negative, tf.constant([[0, 0], [0, fixed_length-length_pos_neg], [0, 0]]))
            negative_representation = self.encoder(
                negative,
                training=training
            )
            loss += multiplicative_ratio * -K.mean(
                K.log(K.sigmoid(-K.batch_dot(
                    K.reshape(representation, (batch_size, 1, size_representation)),
                    K.reshape(negative_representation, (batch_size, size_representation, 1))
                )))
            )

        return loss


class MSELoss(object):
    def __init__(self, autoencoder):
        self.autoencoder = autoencoder
        self.loss = tf.keras.losses.MeanSquaredError()

    def compute_loss(self, batch, noisy_batch=None, training=True):
        if noisy_batch is None:
            noisy_batch = batch
        decoding = self.autoencoder(noisy_batch, training=training)
        # y_pred = ops.convert_to_tensor(decoding)
        #         # y_true = math_ops.cast(batch, y_pred.dtype)
        #         # return K.mean(math_ops.squared_difference(y_pred, y_true))
        return self.loss(batch, decoding)


class JointLearningLoss(object):
    def __init__(self, layers_generator, hlayer_loss_param=0.1):
        if not layers_generator.support_joint_training:
            raise utils.CompatibilityException('architecture incompatible with Joint Learning loss')
        self.encoder = layers_generator.get_all_layers_encoder()
        self.decoder = layers_generator.get_all_layers_decoder()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.hlayer_loss_param = hlayer_loss_param

    def compute_loss(self, batch, noisy_batch=None, training=True):
        if noisy_batch is None:
            noisy_batch = batch
        encoding = self.encoder(noisy_batch, training=training)
        decoding = self.decoder(encoding[-1], training=training)

        loss = 0
        for i in range(len(encoding) - 1):
            loss += self.hlayer_loss_param*self.loss(encoding[i], decoding[-2 - i])

        loss += self.loss(batch, decoding[-1])
        return loss


class VAELoss(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        if K.floatx() == 'float64':
            dtype = tf.dtypes.float64
        else:
            dtype = tf.dtypes.float32
        self.dtype = dtype

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        log2pi = tf.cast(log2pi, self.dtype)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.cast(tf.exp(-logvar), self.dtype) + logvar + log2pi),
                             axis=raxis)

    def compute_loss(self, batch, noisy_batch=None, training=True):
        mean, logvar = tf.split(self.encoder(batch), num_or_size_splits=2, axis=1)
        epsilon = K.random_normal(shape=mean.shape)
        z = mean + K.exp(logvar / 2) * epsilon
        x_logit = self.decoder(z, training=training)

        # we use the classic mse because values are z-normalized (so not necessarily between 0 and 1)
        recon = K.sum(tf.keras.losses.mean_squared_error(batch, x_logit), axis=1)

        kl = 0.5 * K.sum(K.exp(logvar) + K.square(mean) - 1. - logvar, axis=1)

        return recon + kl


class SiameseTSLoss(object):
    def __init__(self, autoencoder1, autoencoder2, filter1, filter2):
        self.autoencoder1 = autoencoder1
        self.autoencoder2 = autoencoder2
        self.filter1 = filter1
        self.filter2 = filter2
        self.loss = tf.keras.losses.MeanSquaredError()

    def compute_loss(self, batch, noisy_batch=None, training=True):
        if noisy_batch is None:
            noisy_batch = batch
        decoding1 = self.autoencoder1(noisy_batch, training=training)
        loss1 = self.loss(self.filter1(batch), decoding1)
        decoding2 = self.autoencoder2(noisy_batch, training=training)
        loss2 = self.loss(self.filter2(batch), decoding2)
        return loss1 + loss2


class CombinedLoss(object):
    def __init__(self, losses, weights=None):
        self.losses = losses
        self.weights = weights
        if self.weights is None:
            self.weights = np.ones_like(self.losses)

    def compute_loss(self, batch, noisy_batch=None, training=True):
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss.compute_loss(batch, noisy_batch=noisy_batch, training=training)
        return total_loss
