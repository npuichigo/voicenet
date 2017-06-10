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

import sys
import os
import math
import sonnet as snt
import tensorflow as tf

sys.path.append(os.path.dirname(sys.path[0]))
from io_funcs.tfrecords_io import get_padded_batch


class SequenceDataset(snt.AbstractModule):
    """Sequence dataset provider."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    def __init__(self, subset, config_dir, data_dir, batch_size,
                 input_size, output_size, num_enqueuing_threads=8,
                 num_epochs=None, infer=False, name="sequence_dataset"):
        if subset not in [self.TRAIN, self.VALID, self.TEST]:
            raise ValueError("subset should be %s, %s, or %s. Received %s instead."
                             % (self.TRAIN, self.VALID, self.TEST, subset))

        super(SequenceDataset, self).__init__(name=name)

        self._config_dir = config_dir
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._input_size = input_size
        self._output_size = output_size
        self._num_enqueuing_threads = num_enqueuing_threads
        self._num_epochs = num_epochs
        self._infer = infer
        self._tfrecords_lst = self.read_config_file(subset)
        self._num_batches = int(math.ceil(len(self._tfrecords_lst) / self._batch_size))

    def _build(self):
        if not self._infer:
            input_sequence, target_sequence, length = get_padded_batch(
                file_list=self._tfrecords_lst,
                batch_size=self._batch_size,
                input_size=self._input_size,
                output_size=self._output_size,
                num_enqueuing_threads=self._num_enqueuing_threads,
                num_epochs=self._num_epochs,
                infer=self._infer)
            return input_sequence, target_sequence, length
        else:
            input_sequence, length = get_padded_batch(
                file_list=self._tfrecords_lst,
                batch_size=self._batch_size,
                input_size=self._input_size,
                output_size=self._output_size,
                num_enqueuing_threads=self._num_enqueuing_threads,
                num_epochs=self._num_epochs,
                infer=self._infer)
            return input_sequence, length

    def read_config_file(self, name):
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
