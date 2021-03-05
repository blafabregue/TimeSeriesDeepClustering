"""
Based on tensorflow implementation https://github.com/qianlima-lab/DTCR
and article :
        Ma, Q., Zheng, J., Li, S., & Cottrell, G. W. (2019),
        Learning Representations for Time Series Clustering

Author:
Baptiste Lafabregue 2019.25.04
"""

import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow as tf

from networks.encoders import RnnAutoencoderModel
from networks.encoders import LayersGenerator


class FormatLayer(layers.Layer):

    def __init__(self, n_steps, dilation_rate):
        super(FormatLayer, self).__init__()
        self.n_steps = n_steps
        self.dilation_rate = dilation_rate

    def call(self, inputs):
        # make the length of inputs divide 'rate', by using zero-padding
        EVEN = (self.n_steps % self.dilation_rate) == 0
        if not EVEN:
            # Create a tensor in shape (batch_size, input_dims), which all elements are zero.
            # This is used for zero padding
            zero_tensor = K.zeros_like(inputs[0])
            zero_tensor = K.expand_dims(zero_tensor, axis=0)
            dilated_n_steps = self.n_steps // self.dilation_rate + 1
            # print("=====> %d time points need to be padded. " % (dilated_n_steps * self.dilation_rate - self.n_steps))
            # print("=====> Input length for sub-RNN: %d" % dilated_n_steps)
            for i_pad in range(dilated_n_steps * self.dilation_rate - self.n_steps):
                inputs = K.concatenate((inputs, zero_tensor), axis=0)
        else:
            dilated_n_steps = self.n_steps // self.dilation_rate
            # print("=====> Input length for sub-RNN: %d" % dilated_n_steps)
        # now the length of 'inputs' divide rate
        # reshape it in the format of a list of tensors
        # the length of the list is 'dilated_n_steps'
        # the shape of each tensor is [batch_size * rate, input_dims]
        # by stacking tensors that "colored" the same

        # Example:
        # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
        # zero-padding --> [x1, x2, x3, x4, x5, 0]
        # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
        # which the length is the ceiling of n_steps/rate
        # print('format call inputs '+str(inputs))
        dilated_inputs = [tf.concat(tf.unstack(inputs[i * self.dilation_rate:(i + 1) * self.dilation_rate]),
                                    axis=0) for i in range(dilated_n_steps)]

        # print('format call dilated '+str(dilated_inputs))
        dilated_inputs = tf.stack(dilated_inputs)
        return dilated_inputs

    def compute_output_shape(self, input_shape):
        int_input_shape = input_shape.as_list()
        dilated_n_steps = int(np.ceil(self.n_steps / self.dilation_rate))

        shape2 = None
        if int_input_shape[1] is not None:
            shape2 = self.dilation_rate * int_input_shape[1]

        output_shape = tf.TensorShape([dilated_n_steps, shape2,
                                       int_input_shape[2]])
        return output_shape


class ReformatLayer(layers.Layer):

    def __init__(self, n_steps, dilation_rate):
        super(ReformatLayer, self).__init__()
        self.n_steps = n_steps
        self.dilation_rate = dilation_rate

    def call(self, inputs):
        # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
        # split each element of the outputs from size [batch_size*rate, input_dims] to
        # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
        # print("input 12 : "+str(inputs))
        inputs = tf.unstack(inputs)
        splitted_outputs = [tf.split(output, self.dilation_rate, axis=0)
                            for output in inputs]
        # print("splitted "+str(splitted_outputs))
        unrolled_outputs = [output
                            for sublist in splitted_outputs for output in sublist]
        # unrolled_outputs = tf.unstack(splitted_outputs)
        # remove padded zeros
        outputs = unrolled_outputs[:self.n_steps]
        # print("outputs "+str(outputs))

        outputs = tf.stack(outputs)
        # print("stack "+str(outputs))

        return outputs

    def compute_output_shape(self, input_shape):
        int_input_shape = input_shape.as_list()

        shape2 = None
        if int_input_shape[1] is not None:
            shape2 = int_input_shape[1] // self.dilation_rate

        output_shape = tf.TensorShape([self.n_steps, shape2,
                                       int_input_shape[2]])
        return output_shape


def _construct_dilated_rnn(input, cell, dilation_rate, n_steps, return_sequences=True, bidirectional=True):

    if dilation_rate < 0 or dilation_rate >= n_steps:
        raise ValueError('The \'dilation rate\' is lower than the number of time steps.')

    h = FormatLayer(n_steps, dilation_rate)(input)

    # building a dilated RNN with formated (dilated) inputs
    rnn = layers.RNN(cell, return_sequences=return_sequences, time_major=True, return_state=True)
    if bidirectional:
        rnn = layers.Bidirectional(rnn)
    outputs = rnn(h)
    h, state = outputs[0], outputs[1:]

    if return_sequences:
        h = ReformatLayer(n_steps, dilation_rate)(h)

    return h, state


def reformat_representation_layer(inputs, dilation_rate):
    splitted_outputs = [tf.split(i, dilation_rate, axis=0) for i in inputs]
    return [tf.reduce_sum(s, axis=0) for s in splitted_outputs]


class ReformatRepresentationLayer(layers.Layer):

    def __init__(self, dilation_rate, batch_size):
        super(ReformatRepresentationLayer, self).__init__()
        self.dilation_rate = dilation_rate
        self.batch_size = batch_size

    def call(self, inputs):
        # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
        # split each element of the outputs from size [batch_size*rate, input_dims] to
        # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
        splitted_outputs = tf.split(inputs, self.dilation_rate, axis=0)
        # then we do a reduce sum to sum all tensors to one [batch_size, input_dims] tensor
        outputs = tf.reduce_sum(splitted_outputs, axis=0)
        return outputs

    def compute_output_shape(self, input_shape):
        int_input_shape = input_shape.as_list()
        output_shape = tf.TensorShape([self.batch_size, int_input_shape[1]])
        return output_shape


class TimeMajorLayer(layers.Layer):
    """
    This Layer reformat input to the shape that standard RNN can take (Time Major).

    Inputs:
        x -- a tensor of shape (batch_size, n_steps, channel_dim).
    Outputs:
        x_reformat -- a list of 'n_steps' tensors, each has shape (batch_size, input_dims).
    """

    def __init__(self, n_steps, channel_dim):
        super(TimeMajorLayer, self).__init__()
        self.n_steps = n_steps
        self.channel_dim = channel_dim

    def call(self, inputs):
        # permute batch_size and n_steps
        x_ = tf.transpose(inputs, [1, 0, 2])
        # reshape to (n_steps*batch_size, input_dims)
        x_ = tf.reshape(x_, [-1, self.channel_dim])
        # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
        x_reformat = tf.split(x_, self.n_steps, 0)
        x_reformat = tf.stack(x_reformat)

        return x_reformat


class BatchMajorLayer(layers.Layer):
    """
    This Layer reformat input to standard shape (Batch Major).

    Inputs:
        x  -- a list of 'n_steps' tensors, each has shape (batch_size, input_dims).
    Outputs:
        x_reformat -- a tensor of shape (batch_size, n_steps, channel_dim).
    """

    def __init__(self, n_steps, channel_dim, batch_size):
        super(BatchMajorLayer, self).__init__()
        self.n_steps = n_steps
        self.channel_dim = channel_dim
        self.batch_size = batch_size

    def call(self, inputs):
        # permute batch_size and n_steps
        x_ = tf.transpose(inputs, [1, 0, 2])
        # reshape to (n_steps*batch_size, input_dims)
        x_ = tf.reshape(x_, [-1, self.channel_dim])
        # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
        n_batch = self.batch_size
        if K.int_shape(x_)[0] is not None:
            n_batch = K.int_shape(x_)[0]//self.n_steps
        x_reformat = tf.split(x_, n_batch, 0)
        x_reformat = tf.stack(x_reformat)

        return x_reformat


def _append_states(concat_states, new_state, bidirectional=True):
    if bidirectional:
        tmp_states = []
        nb_states = int(len(new_state) / 2)
        for i in range(nb_states):
            tmp = tf.concat([new_state[i], new_state[nb_states + i]], axis=-1)
            tmp_states.append(tmp)
        new_state = tmp_states

    if len(concat_states) < len(new_state):
        for i in range(len(new_state)-len(concat_states)): concat_states.append([])
    for i, si in zip(range(len(new_state)), new_state):
        concat_states[i].append(si)


class AutoEncoder(LayersGenerator):
    """docstring for AutoEncoder"""

    def __init__(self,
                 x,
                 encoder_loss,
                 batch_size=10,
                 nb_RNN_units=[100, 50, 50],
                 dilations=[1, 4, 16],
                 cell_type='GRU',
                 alpha=1.0,
                 nb_steps=2000,
                 bidirectional=True):
        self.input_shape = x.shape[1:]
        if len(self.input_shape) > 2:
            self.input_shape = self.input_shape[1:]
        # we only consider the time and channel dimensions
        self.L = self.input_shape[0]
        self.C = self.input_shape[1]
        self.batch_size = batch_size
        self.nb_RNN_units = nb_RNN_units
        self.cell_type = cell_type
        assert self.cell_type in ['GRU', 'LSTM', 'RNN']
        self.dilations = dilations
        if self.dilations is None:
            self.dilations = [2**i for i in range(len(nb_RNN_units))]
        self.alpha = alpha
        self.bidirectional = bidirectional
        self.inputs_ = layers.Input(shape=(self.L, self.C))
        self.encoder = None
        self._decoder = None
        # the decoder should be designed to fit general framework to ba added
        self.decoder = None
        self.nb_steps = nb_steps

        state = self._encoder_network()
        self._decoder_network(init_state=state)
        self.autoencoder = RnnAutoencoderModel(self.encoder, self._decoder)

        # self.input_batch_size = tf.placeholder(tf.int32, shape=[])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.1,
            decay_steps=nb_steps//4,
            decay_rate=0.1,
            staircase=True)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    def _construct_encoder_cells(self):
        """
        This function contructs a list of cells for the encoder.
        """
        # define cells
        cells = []
        for u in self.nb_RNN_units:
            if self.cell_type == "RNN":
                cell = layers.SimpleRNNCell(u)
            elif self.cell_type == "LSTM":
                cell = layers.LSTMCell(u)
            elif self.cell_type == "GRU":
                cell = layers.GRUCell(u)
            cells.append(cell)

        return cells

    def _encoder_network(self):

        cells = self._construct_encoder_cells()
        concat_states = [[]]

        with tf.name_scope('encoder'):
            h = TimeMajorLayer(self.L, self.C)(self.inputs_)

            for cell, dilation in zip(cells[:-1], self.dilations[:-1]):
                scope_name = 'dRNN_'+str(dilation)
                h, tmp_state = _construct_dilated_rnn(h, cell, dilation, self.L,
                                                      bidirectional=self.bidirectional)
                _append_states(concat_states, reformat_representation_layer(tmp_state, dilation))

            h, tmp_state = _construct_dilated_rnn(h, cells[-1], self.dilations[-1], self.L,
                                                  return_sequences=False, bidirectional=self.bidirectional)
            _append_states(concat_states, reformat_representation_layer(tmp_state, self.dilations[-1]))

            h = ReformatRepresentationLayer(self.dilations[-1], self.batch_size)(h)

            # h = BatchMajorLayer(self.nb_steps, self.C, self.batch_size)(h)
        self.encoder = Model(inputs=self.inputs_, outputs=h)

        return [tf.concat(sub_states, axis=1) for sub_states in concat_states]

    def _construct_decoder_cell(self, bidirectional=True):
        """
        This function contructs a cell for the decoder.
        """
        nb_units = 0
        # compute nb units based on encoder units
        for u in self.nb_RNN_units:
            nb_units += u

        if bidirectional:
            nb_units *= 2

        if self.cell_type == "RNN":
            cell = layers.SimpleRNNCell(nb_units)
        elif self.cell_type == "LSTM":
            cell = layers.LSTMCell(nb_units)
        elif self.cell_type == "GRU":
            cell = layers.GRUCell(nb_units)

        return cell

    def _decoder_network(self, init_state=None):

        with tf.name_scope('decoder'):
            decoder_inputs = layers.Input(shape=(1, self.C))
            tmp_inputs = decoder_inputs

            decode = layers.RNN(self._construct_decoder_cell(bidirectional=self.bidirectional),
                                return_sequences=True, return_state=True)
            decode_dense = layers.Dense(self.C)

        all_outputs = []
        for _ in range(self.L):
            # Run the decoder on one timestep
            outputs = decode(tmp_inputs, initial_state=init_state)
            output = outputs[0]
            output = decode_dense(output)
            # Store the current prediction (we will concatenate all predictions later)
            all_outputs.append(output)
            # Reinject the outputs as inputs for the next loop iteration
            # as well as update the states
            tmp_inputs = output
            init_state = outputs[1:]

        decoder_outputs = layers.Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # decoder_input_data = np.zeros((num_samples, 1, num_decoder_tokens))

        self._decoder = Model(inputs=[self.inputs_, decoder_inputs], outputs=decoder_outputs)
