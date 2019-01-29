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
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import *


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    model_file = "frozen_acoustic.pb"
    input_layer = "input"
    output_layer = "output"
    num_steps = 40

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--num_steps", help="number of inference steps to run")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer
    if args.num_steps:
        num_steps = args.num_steps

    graph = load_graph(model_file)
    input_seq = np.random.rand(1000, 425).astype(np.float32)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    with tf.Session(graph=graph) as sess:
        # Warm up
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: input_seq})

        time_used = 0.0
        for i in range(num_steps):
            time_start = time.time()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: input_seq})
            time_end = time.time()
            time_used += time_end - time_start

    tf.logging.info("Input shape: %s" % str(input_seq.shape))
    tf.logging.info("Output shape: %s " % str(results.shape))

    tf.logging.info(results)
    tf.logging.info("Time used: %.3fs" % (time_used / num_steps))
