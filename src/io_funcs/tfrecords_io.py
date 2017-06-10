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

"""Utility functions for working with tf.train.SequenceExamples."""

import tensorflow as tf


def make_sequence_example(inputs, labels=None):
    """Returns a SequenceExample for the given inputs and labels(optional).
    Args:
        inputs: A list of input vectors. Each input vector is a list of floats.
        labels(optional): A list of label vectors. Each label vector is a list of floats.
    Returns:
        A tf.train.SequenceExample containing inputs and labels(optional).
    """
    if labels is not None:
        input_features = [
            tf.train.Feature(float_list=tf.train.FloatList(value=input_))
            for input_ in inputs]
        label_features = [
            tf.train.Feature(float_list=tf.train.FloatList(value=label))
            for label in labels]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features),
            'labels': tf.train.FeatureList(feature=label_features)
        }
    else:
        input_features = [
            tf.train.Feature(float_list=tf.train.FloatList(value=input_))
            for input_ in inputs]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features)
        }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


def get_padded_batch(file_list, batch_size, input_size, output_size,
                     num_enqueuing_threads, num_epochs, infer):
    """Reads batches of SequenceExamples from TFRecords and pads them.
    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.
    Args:
        file_list: A list of paths to TFRecord files containing SequenceExamples.
        batch_size: The number of SequenceExamples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size, num_steps, input_size].
        num_enqueuing_threads: The number of threads to use for enqueuing
            SequenceExamples.
    Returns:
        inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
        labels: A tensor of shape [batch_size, num_steps] of float32s.
        lengths: A tensor of shape [batch_size] of int32s. The lengths of each
            SequenceExample before padding.
    """
    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=num_epochs, shuffle=(not infer))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    if not infer:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32),
            'labels': tf.FixedLenSequenceFeature(shape=[output_size],
                                                 dtype=tf.float32)}

        _, sequence = tf.parse_single_sequence_example(
            serialized_example, sequence_features=sequence_features)

        length = tf.shape(sequence['inputs'])[0]

        capacity = 1000 + (num_enqueuing_threads + 1) * batch_size
        #queue = tf.PaddingFIFOQueue(
        #    capacity=capacity,
        #    dtypes=[tf.float32, tf.float32, tf.int32],
        #    shapes=[(None, input_size), (None, output_size), ()])

        #enqueue_ops = [queue.enqueue([sequence['inputs'],
        #                              sequence['labels'],
        #                              length])] * num_enqueuing_threads

        #tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
        #return queue.dequeue_up_to(batch_size)

        return tf.train.batch(
            [sequence['inputs'], sequence['labels'], length],
            batch_size=batch_size,
            num_threads=num_enqueuing_threads,
            capacity=capacity,
            shapes=[(None, input_size), (None, output_size), ()],
            dynamic_pad=True,
            allow_smaller_final_batch=True)
    else:
        sequence_features = {
            'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                                 dtype=tf.float32)}

        _, sequence = tf.parse_single_sequence_example(
            serialized_example, sequence_features=sequence_features)

        length = tf.shape(sequence['inputs'])[0]

        capacity = 1
        queue = tf.PaddingFIFOQueue(
            capacity=capacity,
            dtypes=[tf.float32, tf.int32],
            shapes=[(None, input_size), ()])

        enqueue_ops = [queue.enqueue([sequence['inputs'],
                                      length])] * num_enqueuing_threads

        tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
        return queue.dequeue_up_to(batch_size)


def get_spliced_batch(file_list, batch_size, input_size, output_size,
                      left_splice=4, right_splice=4, num_enqueuing_threads=4,
                      num_epochs=None, infer=False):
    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=num_epochs, shuffle=(not infer))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    sequence_features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                             dtype=tf.float32),
        'labels': tf.FixedLenSequenceFeature(shape=[output_size],
                                             dtype=tf.float32)}

    _, sequence = tf.parse_single_sequence_example(
        serialized_example, sequence_features=sequence_features)

    inputs = splice_feats(sequence['inputs'], left_splice, right_splice)
    labels = sequence['labels']

    min_after_dequeue = 1000
    capacity = min_after_dequeue + (num_enqueuing_threads + 1) * batch_size
    queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        dtypes=[tf.float32, tf.float32],
        shapes=[(input_size * (left_splice + 1 + right_splice),), (output_size,)])

    enqueue_ops = [queue.enqueue_many([inputs,
                                       labels])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    return queue.dequeue_up_to(batch_size)


def splice_feats(feats, left_splice, right_splice):
    feats_list = []
    row = tf.shape(feats)[0]
    for i in range(left_splice, 0, -1):
        sliced_feats = tf.slice(feats, [0, 0], [row - i, -1])
        for j in range(i):
            sliced_feats = tf.pad(sliced_feats, [[1, 0], [0, 0]], mode='SYMMETRIC')
        feats_list.append(sliced_feats)

    feats_list.append(feats)

    for i in range(1, right_splice + 1):
        sliced_feats = tf.slice(feats, [i, 0], [-1, -1])
        for j in range(i):
            sliced_feats = tf.pad(sliced_feats, [[0, 1], [0, 0]], mode='SYMMETRIC')
        feats_list.append(sliced_feats)
    return tf.concat(feats_list, 1)
