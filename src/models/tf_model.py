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
import sonnet as snt
import tensorflow as tf


class TfModel(snt.AbstractModule):
    """A deep RNN model, for use of acoustic or duration modeling."""

    def __init__(self, rnn_cell, num_hidden, dnn_depth, rnn_depth, output_size,
                 bidirectional=False, rnn_output=False, cnn_output=False,
                 look_ahead=5, mdn_output=False, mix_num=1, name="acoustic_model"):
        """Constructs a TfModel.

        Args:
            rnn_cell: Type of rnn cell including rnn, gru and lstm
            num_hidden: Number of hidden units in each RNN layer.
            dnn_depth: Number of DNN layers.
            rnn_depth: Number of RNN layers.
            output_size: Size of the output layer on top of the DeepRNN.
            bidirectional: Whether to use bidirectional rnn.
            rnn_output: Whether to use ROL(Rnn Output Layer).
            cnn_output: Whether to use COL(Cnn Output Layer).
            look_ahead: Look ahead window size, used together with cnn_output.
            mdn_output: Whether to interpret last layer as mixture density layer.
            mix_num: Number of gaussian mixes in mdn layer.
            name: Name of the module.
        """

        super(TfModel, self).__init__(name=name)

        if rnn_cell == 'rnn':
            self._cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif rnn_cell == 'gru':
            self._cell_fn = tf.nn.rnn_cell.GRUCell
        elif rnn_cell == 'lstm':
            self._cell_fn = tf.nn.rnn_cell.LSTMCell
        else:
            raise ValueError("model type not supported: {}".format(rnn_cell))

        self._num_hidden = num_hidden
        self._dnn_depth = dnn_depth
        self._rnn_depth = rnn_depth
        self._output_size = output_size
        self._bidirectional = bidirectional
        self._rnn_output = rnn_output
        self._cnn_output = cnn_output
        self._look_ahead = look_ahead
        self._mdn_output = mdn_output
        self._mix_num = mix_num

        with self._enter_variable_scope():
            self._input_module = snt.nets.MLP(
                output_sizes=[self._num_hidden] * self._dnn_depth,
                activation=tf.nn.relu,
                activate_final=True,
                name="mlp_input")

            #self._quasi_rnns = [QuasiRNN(filter_width=2,
            #                             num_hidden=256,
            #                             pool_type='fo',
            #                             name="qusi_rnn_{}".format(i))
            #                    for i in xrange(4)]

            if not self._bidirectional:
                self._rnns = [
                    self._cell_fn(self._num_hidden)
                    for i in range(self._rnn_depth)
                ]
                self._core = tf.nn.rnn_cell.MultiRNNCell(self._rnns)
            else:
                self._core = {
                    "fw": tf.nn.rnn_cell.MultiRNNCell([
                        self._cell_fn(self._num_hidden)
                        for i in range(self._rnn_depth)
                    ]),
                    "bw": tf.nn.rnn_cell.MultiRNNCell([
                        self._cell_fn(self._num_hidden)
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
                self._output_module = tf.nn.rnn_cell.BasicRNNCell(
                    output_size, activation=tf.identity)
            elif self._cnn_output:
                self._output_module = {
                    "linear": snt.Linear(output_size, name="linear"),
                    "cnn": snt.Conv2D(
                        output_channels=1,
                        kernel_shape=(self._look_ahead, 1),
                        padding="VALID",
                        name="cnn_output")
                }
            else:
                self._output_module = snt.Linear(output_size, name="linear_output")

    def _build(self, input_sequence, input_length):
        """Builds the deep LSTM model sub-graph.

        Args:
        input_sequence: A 3D Tensor with padded input sequence data.
        input_length. Actual length of each sequence in padded input data.

        Returns:
            Tuple of the Tensor of output logits for the batch, with dimensions
            `[truncation_length, batch_size, output_size]`, and the
            final state of the unrolled core,.
        """

        batch_input_module = snt.BatchApply(self._input_module)
        output_sequence = batch_input_module(input_sequence)

        #for layer_id in xrange(2):
        #    quasi_rnn = self._quasi_rnns[layer_id]
        #    output_sequence, final_state = quasi_rnn(output_sequence, input_length)

        if not self._bidirectional:
            output_sequence, final_state = tf.nn.dynamic_rnn(
                cell=self._core,
                inputs=output_sequence,
                sequence_length=input_length,
                dtype=tf.float32)
        else:
            outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=self._unpack_cell(self._core["fw"]),
                cells_bw=self._unpack_cell(self._core["bw"]),
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
                for i in xrange(self._look_ahead - 1):
                    output_sequence = tf.pad(output_sequence,
                                             [[0, 0], [0, 1], [0, 0]],
                                             "SYMMETRIC")
            else:
                for i in xrange((self._look_ahead - 1) / 2):
                    output_sequence = tf.pad(output_sequence,
                                             [[0, 0], [1, 1], [0, 0]],
                                             "SYMMETRIC")
            batch_output_module = snt.BatchApply(self._output_module["linear"])
            output_sequence = batch_output_module(output_sequence)
            output_sequence_logits = tf.squeeze(self._output_module["cnn"](
                tf.expand_dims(output_sequence, -1)))
        else:
            batch_output_module = snt.BatchApply(self._output_module)
            output_sequence_logits = batch_output_module(output_sequence)

        return output_sequence_logits, final_state

    def cost(self, logits, target, length):
        """Returns cost.

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

            finetune_cost = -tf.reduce_mean(
                max_exponent + tf.log(tf.reduce_sum(tf.exp(mod_exponent), 1, keep_dims=True)))

            return finetune_cost

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
