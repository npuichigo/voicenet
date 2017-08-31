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


class RecurrentPooling(tf.nn.rnn_cell.RNNCell):
  """The recurrent pooling component in QuasiRNN.
  Args:
    num_units: int, The number of units in the RNN cell.
    pool_type: str, The type of pooling, must be either f, fo or ifo.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, num_units, pool_type, reuse=None):
    super(RecurrentPooling, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._pool_type = pool_type

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """fo-pooling: output = new_state = f * state + (1-f) * inputs."""
    with tf.variable_scope("{}_pooling".format(self._pool_type)):
        if self._pool_type == 'f':
            z, f = tf.split(inputs, len(self._pool_type) + 1, 1)
            output = new_state = tf.multiply(f, state) + tf.multiply(1 - f, z)
            return (output, output)
        elif self._pool_type == 'fo':
            z, f, o = tf.split(inputs, len(self._pool_type) + 1, 1)
            new_state = tf.multiply(f, state) + tf.multiply(1 - f, z)
            output = tf.multiply(o, new_state)
            return (output, new_state)
        elif self._pool_type == 'ifo':
            z, i, f, o = tf.split(inputs, len(self._pool_type) + 1, 1)
            new_state = tf.multiply(f, state) + tf.multiply(i, z)
            output = tf.multiply(o, new_state)
            return (output, new_state)
        else:
            raise ValueError('Pool type must be either f, fo or ifo.')


class QuasiRNN(snt.AbstractModule):
    """Implementation of Quasi-Recurrent Neural Network"""

    def __init__(self, filter_width, num_hidden, pool_type='fo', zone_out=0.0,
                 name="quasi_rnn"):
        """Constructs a quasi_rnn.

        Args:
            filter_width: Filter width of the convolutional component.
            num_hidden: Number of hidden units in each RNN layer.
            pool_type: Types of pool component. (f-pooling, fo-pooling, ifo-pooling)
            zoon_out: The dropout rate.
            name: Name of the module.
        """

        super(QuasiRNN, self).__init__(name=name)

        self._filter_width = filter_width
        self._num_hidden = num_hidden
        assert pool_type in ['f', 'fo', 'ifo']
        self._pool_type = pool_type
        self._zone_out = zone_out

        with self._enter_variable_scope():
            self._cnn_component = snt.Conv1D(
                output_channels=self._num_hidden * (len(self._pool_type) + 1),
                kernel_shape=self._filter_width,
                padding="VALID",
                name="cnn_component")

            self._pooling_component = RecurrentPooling(self._num_hidden, self._pool_type)


    def _build(self, input_sequence, input_length):
        """Builds the quasi_rnn model sub-graph."""

        # Padding before apply convolution component.
        num_pads = self._filter_width - 1
        padded_inputs = tf.pad(input_sequence,
                               [[0, 0], [num_pads, 0], [0, 0]],
                               "CONSTANT")
        # Apply convolution component.
        conv_results = self._cnn_component(padded_inputs)
        gates = tf.split(conv_results, len(self._pool_type) + 1, 2)
        # Apply nonlinearities.
        gates[0] = tf.tanh(gates[0])
        for i in range(1, len(gates)):
            gates[i] = tf.sigmoid(gates[i])
        conv_results = tf.concat(gates, 2)

        # Apply recurrent pooling component.
        output_sequence, final_state = tf.nn.dynamic_rnn(
            cell=self._pooling_component,
            inputs=conv_results,
            sequence_length=input_length,
            dtype=tf.float32)

        return output_sequence, final_state
