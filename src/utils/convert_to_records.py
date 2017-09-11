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
"""Converts data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from io_funcs.tfrecords_io import make_sequence_example
from utils import read_binary_file

tf.logging.set_verbosity(tf.logging.INFO)


def calculate_cmvn(name):
    """Calculate mean and var."""
    tf.logging.info("Calculating mean and var of %s" % name)
    config_filename = open(os.path.join(FLAGS.config_dir, name + '.lst'))

    inputs_frame_count, labels_frame_count = 0, 0
    for line in config_filename:
        utt_id, inputs_path, labels_path = line.strip().split()
        tf.logging.info("Reading utterance %s" % utt_id)
        inputs = read_binary_file(inputs_path, FLAGS.input_dim)
        labels = read_binary_file(labels_path, FLAGS.output_dim)
        if inputs_frame_count == 0:    # create numpy array for accumulating
            ex_inputs = np.sum(inputs, axis=0)
            ex2_inputs = np.sum(inputs**2, axis=0)
            ex_labels = np.sum(labels, axis=0)
            ex2_labels = np.sum(labels**2, axis=0)
        else:
            ex_inputs += np.sum(inputs, axis=0)
            ex2_inputs += np.sum(inputs**2, axis=0)
            ex_labels += np.sum(labels, axis=0)
            ex2_labels += np.sum(labels**2, axis=0)
        inputs_frame_count += len(inputs)
        labels_frame_count += len(labels)

    mean_inputs = ex_inputs / inputs_frame_count
    stddev_inputs = np.sqrt(ex2_inputs / inputs_frame_count - mean_inputs**2)
    stddev_inputs[stddev_inputs < 1e-20] = 1e-20

    mean_labels = ex_labels / labels_frame_count
    stddev_labels = np.sqrt(ex2_labels / labels_frame_count - mean_labels**2)
    stddev_labels[stddev_labels < 1e-20] = 1e-20

    cmvn_name = os.path.join(FLAGS.output_dir, name + "_cmvn.npz")
    np.savez(cmvn_name,
             mean_inputs=mean_inputs,
             stddev_inputs=stddev_inputs,
             mean_labels=mean_labels,
             stddev_labels=stddev_labels)
    config_filename.close()
    tf.logging.info("Wrote to %s" % cmvn_name)


def convert_to(name, apply_cmvn=True):
    """Converts a dataset to tfrecords."""
    cmvn = np.load(os.path.join(FLAGS.output_dir, "train_cmvn.npz"))
    config_file = open(os.path.join(FLAGS.config_dir, name + ".lst"))
    for line in config_file:
        if name != 'test':
            utt_id, inputs_path, labels_path = line.strip().split()
        else:
            utt_id, inputs_path = line.strip().split()
        tfrecords_name = os.path.join(FLAGS.output_dir, name,
                                      utt_id + ".tfrecords")
        with tf.python_io.TFRecordWriter(tfrecords_name) as writer:
            tf.logging.info(
                "Writing utterance %s to %s" % (utt_id, tfrecords_name))
            inputs = read_binary_file(inputs_path, FLAGS.input_dim).astype(np.float64)
            if name != 'test':
                labels = read_binary_file(labels_path, FLAGS.output_dim).astype(np.float64)
            else:
                labels = None
            if apply_cmvn:
                inputs = (inputs - cmvn["mean_inputs"]) / cmvn["stddev_inputs"]
                if labels is not None:
                    labels = (labels - cmvn["mean_labels"]) / cmvn["stddev_labels"]
            ex = make_sequence_example(inputs, labels)
            writer.write(ex.SerializeToString())

    config_file.close()


def main(unused_argv):
    # Convert to Examples and write the result to TFRecords.
    calculate_cmvn("train")    # use training data to calculate mean and var

    convert_to("train", apply_cmvn=True)
    convert_to("valid", apply_cmvn=True)
    convert_to("test", apply_cmvn=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/',
        help='Directory to write the converted result'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='config',
        help='Directory to load train, valid and test lists'
    )
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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
