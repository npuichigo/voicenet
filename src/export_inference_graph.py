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

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

from models.tf_model import TfModel
from tensorflow.python.tools import freeze_graph
from utils.utils import pp, show_all_variables

# Basic model parameters as external flags.
FLAGS = None


def main(_):
    if not FLAGS.output_file:
        raise ValueError('You must supply the path to save to with --output_file')

    # Read cmvn to do reverse mean variance normalization.
    cmvn = np.load(os.path.join(FLAGS.data_dir, "train_cmvn.npz"))
    with tf.Graph().as_default() as graph:
        model = TfModel(
            rnn_cell=FLAGS.rnn_cell,
            dnn_depth=FLAGS.dnn_depth,
            dnn_num_hidden=FLAGS.dnn_num_hidden,
            rnn_depth=FLAGS.rnn_depth,
            rnn_num_hidden=FLAGS.rnn_num_hidden,
            output_size=FLAGS.output_dim,
            bidirectional=FLAGS.bidirectional,
            rnn_output=FLAGS.rnn_output,
            cnn_output=FLAGS.cnn_output,
            look_ahead=FLAGS.look_ahead,
            mdn_output=FLAGS.mdn_output,
            mix_num=FLAGS.mix_num,
            name="tf_model")

        input_sequence = tf.placeholder(name='input', dtype=tf.float32,
                                        shape=[None, FLAGS.input_dim])
        length = tf.expand_dims(tf.shape(input_sequence)[0], 0)

        # Apply normalization for input before inference.
        mean_inputs = tf.constant(cmvn["mean_inputs"], dtype=tf.float32)
        stddev_inputs = tf.constant(cmvn["stddev_inputs"], dtype=tf.float32)
        input_sequence = (input_sequence - mean_inputs) / stddev_inputs
        input_sequence = tf.expand_dims(input_sequence, 0)

        output_sequence_logits, final_state = model(input_sequence, length)

        # Apply reverse cmvn for output after inference
        mean_labels = tf.constant(cmvn["mean_labels"], dtype=tf.float32)
        stddev_labels = tf.constant(cmvn["stddev_labels"], dtype=tf.float32)
        output_sequence_logits = output_sequence_logits * stddev_labels + mean_labels
        output_sequence_logits = tf.squeeze(output_sequence_logits)
        output_sequence_logits = tf.identity(output_sequence_logits,
                                             name=FLAGS.output_node_name)

        show_all_variables()

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        if ckpt:
            saver = tf.train.Saver()
        else:
            tf.logging.warning("Cannot find checkpoint in {}".format(args.checkpoint))
            sys.exit(-1)

        freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=graph.as_graph_def(),
            input_saver_def=saver.as_saver_def(),
            input_checkpoint=ckpt.model_checkpoint_path,
            output_node_names=FLAGS.output_node_name,
            restore_op_name=None,
            filename_tensor_name=None,
            output_graph=FLAGS.output_file,
            clear_devices=True,
            initializer_nodes="",
            variable_names_blacklist=None)

        tf.logging.info("Inference graph has been written to %s" % FLAGS.output_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dim',
        type=int,
        default=145,
        help='The dimension of inputs.'
    )
    parser.add_argument(
        '--output_dim',
        type=int,
        default=75,
        help='The dimension of outputs.'
    )
    parser.add_argument(
        '--rnn_cell',
        type=str,
        default='fused_lstm',
        help='Rnn cell types including rnn, gru and lstm.'
    )
    parser.add_argument(
        '--bidirectional',
        type=_str_to_bool,
        default=False,
        help='Whether to use bidirectional layers.'
    )
    parser.add_argument(
        '--dnn_depth',
        type=int,
        default=2,
        help='Number of layers of dnn model.'
    )
    parser.add_argument(
        '--dnn_num_hidden',
        type=int,
        default=128,
        help='Number of hidden units to use.'
    )
    parser.add_argument(
        '--rnn_depth',
        type=int,
        default=3,
        help='Number of layers of rnn model.'
    )
    parser.add_argument(
        '--rnn_num_hidden',
        type=int,
        default=64,
        help='Number of hidden units to use.'
    )
    parser.add_argument(
        '--rnn_output',
        type=_str_to_bool,
        default=False,
        help='Whether to use rnn as the output layer.'
    )
    parser.add_argument(
        '--cnn_output',
        type=_str_to_bool,
        default=False,
        help='Whether to use cnn as the output layer.'
    )
    parser.add_argument(
        '--look_ahead',
        type=int,
        default=5,
        help='Number of steps to look ahead in cnn output layer.',
    )
    parser.add_argument(
        '--mdn_output',
        type=_str_to_bool,
        default=False,
        help='Whether to use mdn as the output layer.'
    )
    parser.add_argument(
        '--mix_num',
        type=int,
        default=1,
        help='Number of gaussian mixes in mdn output layer.',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='exp/acoustic/frozen_acoustic.pb',
        help='Where to save the resulting file to.'
    )
    parser.add_argument(
        '--output_node_name',
        type=str,
        default='output',
        help='Name of output node.'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Name of model checkpoint path.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/',
        help='Directory of train, val and test data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    pp.pprint(FLAGS.__dict__)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
