# Copyright 2016 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: npuichigo@gmail.com (zhangyuchao)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import sys
import tensorflow as tf


class TfModel(object):
    """A deep RNN model, for use of acoustic or duration modeling."""

    def __init__(self, rnn_cell, dnn_depth, dnn_num_hidden, rnn_depth, rnn_num_hidden,
                 output_size, bidirectional=False, rnn_output=False, cnn_output=False,
                 look_ahead=5, mdn_output=False, mix_num=1, name="acoustic_model"):
        """Constructs a TfModel.

        Args:
            rnn_cell: Type of rnn cell including rnn, gru and lstm
            dnn_depth: Number of DNN layers.
            dnn_num_hidden: Number of hidden units in each DNN layer.
            rnn_depth: Number of RNN layers.
            rnn_num_hidden: Number of hidden units in each RNN layer.
            output_size: Size of the output layer on top of the DeepRNN.
            bidirectional: Whether to use bidirectional rnn.
            rnn_output: Whether to use ROL(Rnn Output Layer).
            cnn_output: Whether to use COL(Cnn Output Layer).
            look_ahead: Look ahead window size, used together with cnn_output.
            mdn_output: Whether to interpret last layer as mixture density layer.
            mix_num: Number of gaussian mixes in mdn layer.
            name: Name of the module.
        """

        super(TfModel, self).__init__()

        if rnn_cell == "rnn":
            self._cell_fn = tf.contrib.rnn.BasicRNNCell
        elif rnn_cell == "gru":
            self._cell_fn = tf.contrib.rnn.GRUBlockCellV2
        elif rnn_cell == "lstm":
            self._cell_fn =  tf.contrib.rnn.LSTMBlockCell
        elif rnn_cell == "fused_lstm":
            self._cell_fn = tf.contrib.rnn.LSTMBlockFusedCell
        else:
            raise ValueError("model type not supported: {}".format(rnn_cell))

        self._rnn_cell = rnn_cell
        self._dnn_depth = dnn_depth
        self._dnn_num_hidden = dnn_num_hidden
        self._rnn_depth = rnn_depth
        self._rnn_num_hidden = rnn_num_hidden
        self._output_size = output_size
        self._bidirectional = bidirectional
        self._rnn_output = rnn_output
        self._cnn_output = cnn_output
        self._look_ahead = look_ahead
        self._mdn_output = mdn_output
        self._mix_num = mix_num

        self._input_module = [
            tf.layers.Dense(units=self._dnn_num_hidden,
                            activation=tf.nn.relu,
                            name="linear_input_{}".format(i))
            for i in range(self._dnn_depth)
        ]

        if not self._bidirectional:
            if rnn_cell == "fused_lstm":
                self._rnns = [
                    self._cell_fn(self._rnn_num_hidden,
                                  name="{0}_{1}".format(rnn_cell, i))
                    for i in range(self._rnn_depth)
                ]
            else:
                self._rnns = tf.nn.rnn_cell.MultiRNNCell([
                    self._cell_fn(self._rnn_num_hidden,
                                  name="{0}_{1}".format(rnn_cell, i))
                    for i in range(self._rnn_depth)
                ])
        else:
            if rnn_cell == "fused_lstm":
                self._rnns = {
                    "fw": [
                        self._cell_fn(self._rnn_num_hidden,
                                      name="{0}_fw_{1}".format(rnn_cell, i))
                        for i in range(self._rnn_depth)
                    ],
                    "bw": [
                        tf.contrib.rnn.TimeReversedFusedRNN(
                            self._cell_fn(self._rnn_num_hidden,
                                          name="{0}_bw_{1}".format(rnn_cell, i)))
                        for i in range(self._rnn_depth)
                    ],
                }
            else:
                self._rnns = {
                    "fw": tf.nn.rnn_cell.MultiRNNCell([
                        self._cell_fn(self._rnn_num_hidden,
                                      name="{0}_fw_{1}".format(rnn_cell, i))
                        for i in range(self._rnn_depth)
                    ]),
                    "bw": tf.nn.rnn_cell.MultiRNNCell([
                        self._cell_fn(self._num_hidden,
                                      name="{0}_bw_{1}".format(rnn_cell, i))
                        for i in range(self._rnn_depth)
                    ]),
                }

        # If mdn output is used, output size should be mix_num * (2 * output_dim + 1).
        if self._mdn_output:
            output_size = self._mdn_output_size = self._mix_num * (2 * self._output_size + 1)
        else:
            output_size = self._output_size

        if self._rnn_output and self._cnn_output:
            raise ValueError("rnn_output and cnn_output cannot be "
                             "specified at the same time.")
        if self._rnn_output:
            self._output_module = tf.contrib.rnn.BasicRNNCell(
                output_size, activation=tf.identity)
        elif self._cnn_output:
            self._output_module = {
                "linear": tf.layers.Dense(output_size, name="linear"),
                "cnn": tf.layers.Conv2D(
                    filters=1,
                    kernel_size=(self._look_ahead, 1),
                    padding="VALID",
                    name="cnn_output")
            }
        else:
            self._output_module = tf.layers.Dense(output_size, name="linear_output")

    def __call__(self, input_sequence, input_length):
        """Builds the deep LSTM model sub-graph.

        Args:
        input_sequence: A 3D Tensor with padded input sequence data.
        input_length. Actual length of each sequence in padded input data.

        Returns:
            Tuple of the Tensor of output logits for the batch, with dimensions
            `[truncation_length, batch_size, output_size]`, and the
            final state of the unrolled core,.
        """
        output_sequence = input_sequence
        for layer in self._input_module:
            output_sequence = layer(output_sequence)

        if not self._bidirectional:
            if self._rnn_cell == 'fused_lstm':
                output_sequence = tf.transpose(output_sequence, [1, 0, 2])

                new_states = []
                for cell in self._rnns:
                    output_sequence, new_state = cell(
                        inputs=output_sequence,
                        sequence_length=input_length,
                        dtype=tf.float32)
                    new_states.append(new_state)

                output_sequence = tf.transpose(output_sequence, [1, 0, 2])
                final_state = tuple(new_states)
            else:
                output_sequence, final_state = tf.nn.dynamic_rnn(
                    cell=self._rnns,
                    inputs=output_sequence,
                    sequence_length=input_length,
                    dtype=tf.float32)
        else:
            if self._rnn_cell == 'fused_lstm':
                output_sequence = tf.transpose(output_sequence, [1, 0, 2])

                fw_new_states, bw_new_states = [], []
                for i in range(self._rnn_depth):
                    fw_output, fw_new_state = self._rnns["fw"][i](
                        inputs=output_sequence,
                        sequence_length=input_length,
                        dtype=tf.float32)
                    fw_new_states.append(fw_new_state)

                    bw_output, bw_new_state = self._rnns["bw"][i](
                        inputs=output_sequence,
                        sequence_length=input_length,
                        dtype=tf.float32)
                    bw_new_states.append(bw_new_state)
                    output_sequence = tf.concat([fw_output, bw_output], axis=-1)

                final_state_fw = tuple(fw_new_states)
                final_state_bw = tuple(bw_new_states)

                output_sequence = tf.transpose(output_sequence, [1, 0, 2])
                final_state = (final_state_fw, final_state_bw)
            else:
                outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=self._unpack_cell(self._rnns["fw"]),
                    cells_bw=self._unpack_cell(self._rnns["bw"]),
                    inputs=output_sequence,
                    sequence_length=input_length,
                    dtype=tf.float32)

                output_sequence, final_state_fw, final_state_bw = outputs
                final_state = (final_state_fw, final_state_bw)

        if self._rnn_output:
            output_sequence_logits, _ = tf.nn.dynamic_rnn(
                cell=self._output_module,
                inputs=output_sequence,
                sequence_length=input_length,
                dtype=tf.float32)
        elif self._cnn_output:
            if not self._bidirectional:
                for i in range(self._look_ahead - 1):
                    output_sequence = tf.pad(output_sequence,
                                             [[0, 0], [0, 1], [0, 0]],
                                             "SYMMETRIC")
            else:
                for i in range(int((self._look_ahead - 1) / 2)):
                    output_sequence = tf.pad(output_sequence,
                                             [[0, 0], [1, 1], [0, 0]],
                                             "SYMMETRIC")
            output_sequence = self._output_module["linear"](output_sequence)
            output_sequence_logits = tf.squeeze(self._output_module["cnn"](
                tf.expand_dims(output_sequence, -1)))
        else:
            output_sequence_logits = self._output_module(output_sequence)

        return output_sequence_logits, final_state

    def loss(self, logits, target, length):
        """Returns loss.

        Args:
            logits: model output.
            target: target.

        Returns:
            RMSE or MDN loss for a sequence of logits. The loss will be averaged.
        """
        if not self._mdn_output:
            # Compute mse for each frame.
            loss = tf.reduce_sum(
                0.5 * tf.square(logits - target), axis=[2])
            # Mask the loss.
            mask = tf.cast(
                tf.sequence_mask(length, tf.shape(logits)[1]), tf.float32)
            loss *= mask
            # Average over actual sequence lengths.
            loss = tf.reduce_mean(
                tf.reduce_sum(loss, axis=[1]) / tf.cast(length, tf.float32))
            return loss
        else:
            # Mask the logits sequence.
            mask = tf.cast(
                tf.sequence_mask(length, tf.shape(logits)[1]), tf.float32)
            logits *= tf.expand_dims(mask, 2)
            logits = tf.reshape(logits, [-1, self._mdn_output_size])
            target = tf.reshape(target, [-1, self._output_size])

            out_pi, out_mu, out_sigma = self._get_mixture_coef(logits, self._mix_num)

            all_mix_prob = []
            for i in xrange(self._mix_num):
                pi = out_pi[:, i : (i + 1)]
                mu = out_mu[:, i * self._output_size : (i + 1) * self._output_size]
                sigma = out_sigma[:, i * self._output_size : (i + 1) * self._output_size]

                tmp = tf.multiply(tf.square(target - mu), tf.reciprocal(sigma))
                xEx = -0.5 * tf.reduce_sum(tmp, 1, keep_dims=True)
                normaliser = tf.reduce_sum(tf.log(sigma), 1, keep_dims=True)
                exponent = xEx + tf.log(pi) - normaliser
                all_mix_prob.append(exponent)

            all_mix_prob = tf.concat(all_mix_prob, 1)
            max_exponent = tf.reduce_max(all_mix_prob, 1, keep_dims=True)
            mod_exponent = all_mix_prob - max_exponent

            loss = -tf.reduce_mean(
                max_exponent + tf.log(tf.reduce_sum(tf.exp(mod_exponent), 1, keep_dims=True)))
            return loss

    def _get_mixture_coef(self, logits, mix_num, var_floor=0.01):
        # pi1, pi2, pi3...
        out_pi = logits[:, :mix_num]  #pi1,pi2,pi3...
        # sigma1, sigma2, sigma3...
        out_mu = logits[:, mix_num:(mix_num + mix_num * self._output_size)]
        # mu1, mu2, mu3...
        out_sigma = logits[:, (mix_num + mix_num * self._output_size):]

        out_pi = tf.nn.softmax(out_pi)
        out_sigma = tf.exp(out_sigma)
        out_sigma = tf.maximum(var_floor, out_sigma)

        return out_pi, out_mu, out_sigma

    def _unpack_cell(self, cell):
        """Unpack the cells because the stack_bidirectional_dynamic_rnn
        expects a list of cells, one per layer."""
        if isinstance(cell, tf.nn.rnn_cell.MultiRNNCell):
            return cell._cells
        else:
            return [cell]
