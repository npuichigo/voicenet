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

import collections
import math
import os
import sys
import sonnet as snt
import tensorflow as tf

class BatchedInput(
    collections.namedtuple("BatchedInput",
                          ("initializer", "input_sequence", "target_sequence",
                           "input_sequence_length", "target_sequence_length"))):
    pass

class SequenceDataset(snt.AbstractModule):
    """Sequence dataset provider."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    def __init__(self, subset, config_dir, data_dir, batch_size, input_size,
                 output_size, num_threads=4, output_buffer_size=None,
                 use_bucket=False, bucket_width=100, num_buckets=10,
                 infer=False, name="sequence_dataset"):
        if subset not in [self.TRAIN, self.VALID, self.TEST]:
            raise ValueError("subset should be %s, %s, or %s. Received %s instead."
                             % (self.TRAIN, self.VALID, self.TEST, subset))

        super(SequenceDataset, self).__init__(name=name)

        self._config_dir = config_dir
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._input_size = input_size
        self._output_size = output_size
        self._num_threads = num_threads
        self._use_bucket = use_bucket
        self._bucket_width = bucket_width
        self._num_buckets = num_buckets
        self._infer = infer
        self._tfrecords_lst = self._read_config_file(subset)
        self._num_batches = int(math.ceil(len(self._tfrecords_lst) / float(self._batch_size)))

        if not output_buffer_size:
            self._output_buffer_size = 1000 + batch_size * (num_threads + 1)
        else:
            self._output_buffer_size = output_buffer_size

    def _build(self):
        if not self._infer:
            def _parse_function(serialized_example):
                sequence_features = {
                    'inputs': tf.FixedLenSequenceFeature(shape=[self._input_size],
                                                         dtype=tf.float32),
                    'labels': tf.FixedLenSequenceFeature(shape=[self._output_size],
                                                         dtype=tf.float32)}

                _, sequence = tf.parse_single_sequence_example(
                    serialized_example, sequence_features=sequence_features)
                return sequence['inputs'], sequence['labels']

            dataset = tf.contrib.data.TFRecordDataset(self._tfrecords_lst)
            # Parse tfrecords example.
            dataset = dataset.map(
                _parse_function,
                num_threads=self._num_threads,
                output_buffer_size = self._output_buffer_size)
            # Randomly shuffle.
            dataset = dataset.shuffle(
                self._output_buffer_size,
                seed=tf.random_uniform((), maxval=777, dtype=tf.int64))
            # Add in sequence lengths.
            dataset = dataset.map(
                lambda src, tgt: (src, tf.shape(src)[0], tgt, tf.shape(tgt)[0]),
                num_threads=self._num_threads,
                output_buffer_size = self._output_buffer_size)

            # Dynamic padding.
            def batching_func(x):
                return x.padded_batch(
                    self._batch_size,
                    # The first and third entries are the source and target
                    # line rows; these have unknown-length vectors.
                    # The second and fourth entries are the source and target
                    # row sizes; these are scalars.
                    padded_shapes=(
                        tf.TensorShape([None, self._input_size]),   # input
                        tf.TensorShape([]),                         # input_len
                        tf.TensorShape([None, self._output_size]),  # target
                        tf.TensorShape([])))                        # target_len

            if self._use_bucket:
                def key_func(unused_1, src_len, unused_2, tgt_len):
                    # Calculate bucket_width by maximum source sequence length.
                    # Pairs with length [0, bucket_width) go to bucket 0, length
                    # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs
                    # with length over ((num_bucket-1) * bucket_width) words all
                    # go into the last bucket.

                    # Bucket sentence pairs by the length of their source
                    # sentence and target sentence.
                    bucket_id = tf.maximum(src_len // self._bucket_width,
                                           tgt_len // self._bucket_width)
                    return tf.to_int64(tf.minimum(self._num_buckets, bucket_id))

                def reduce_func(unused_key, windowed_data):
                    return batching_func(windowed_data)

                batched_dataset = dataset.group_by_window(
                    key_func=key_func,
                    reduce_func=reduce_func,
                    window_size=self._batch_size)
            else:
                batched_dataset = batching_func(dataset)
            batched_iter = batched_dataset.make_initializable_iterator()
            (input_seq, input_seq_len, target_seq, target_seq_len) = (
                batched_iter.get_next())

            return BatchedInput(
                initializer=batched_iter.initializer,
                input_sequence=input_seq,
                input_sequence_length=input_seq_len,
                target_sequence=target_seq,
                target_sequence_length=target_seq_len)
        else:
            def _parse_function(serialized_example):
                sequence_features = {
                    'inputs': tf.FixedLenSequenceFeature(shape=[self._input_size],
                                                         dtype=tf.float32)}

                _, sequence = tf.parse_single_sequence_example(
                    serialized_example, sequence_features=sequence_features)
                return sequence['inputs']

            dataset = tf.contrib.data.TFRecordDataset(self._tfrecords_lst)
            # Parse tfrecords example.
            dataset = dataset.map(_parse_function)
            # Add in sequence lengths.
            dataset = dataset.map(lambda src: (src, tf.shape(src)[0]))

            def batching_func(x):
                return x.padded_batch(
                    self._batch_size,
                        # The first entry is the source line rows;
                        # these have unknown-length vectors.
                        # The second entry is the source row sizes;
                        # these are scalars.
                    padded_shapes=(
                        tf.TensorShape([None, self._input_size]),   # input
                        tf.TensorShape([])))                        # input_len

            batched_dataset = batching_func(dataset)
            batched_iter = batched_dataset.make_initializable_iterator()
            (input_seq, input_seq_len) = batched_iter.get_next()

            return BatchedInput(
                initializer=batched_iter.initializer,
                input_sequence=input_seq,
                input_sequence_length=input_seq_len,
                target_sequence=None,
                target_sequence_length=None)

    def _read_config_file(self, name):
        file_name = os.path.join(self._config_dir, name + ".lst")
        if not tf.gfile.Exists(file_name):
            tf.logging.fatal('File does not exist %s', file_name)
            sys.exit(-1)
        config_file = open(file_name)
        tfrecords_lst = []
        for line in config_file:
            utt_id = line.strip().split()[0]
            tfrecords_name = os.path.join(
                self._data_dir, name, utt_id + ".tfrecords")
            if not tf.gfile.Exists(tfrecords_name):
                tf.logging.fatal('TFrecords does not exist %s', tfrecords_name)
                sys.exit(-1)
            tfrecords_lst.append(tfrecords_name)
        return tfrecords_lst

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def tfrecords_lst(self):
        return self._tfrecords_lst
