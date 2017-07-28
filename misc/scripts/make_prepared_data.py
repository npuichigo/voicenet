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
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(sys.path[0])), 'src'))
from utils.utils import read_binary_file, write_binary_file


def main(unused_argv):
    input_filenames = sorted(os.listdir(FLAGS.input_dir))
    output_filenames = sorted(os.listdir(FLAGS.output_dir))
    assert(len(input_filenames) == len(output_filenames))

    if not os.path.exists('prepared_label'):
        os.mkdir('prepared_label')
    if not os.path.exists('prepared_cmp'):
        os.mkdir('prepared_cmp')

    for i in range(len(input_filenames)):
        basename = os.path.splitext(input_filenames[i])[0]
        print("processing %s" % basename)

        if FLAGS.input_dim:
            input_mat = read_binary_file(
                os.path.join(FLAGS.input_dir, input_filenames[i]), dimension=FLAGS.input_dim)
        else:
            input_mat = np.loadtxt(os.path.join(FLAGS.input_dir, input_filenames[i]))

        if FLAGS.output_dim:
            output_mat = read_binary_file(
                os.path.join(FLAGS.output_dir, output_filenames[i]), dimension=FLAGS.output_dim)
        else:
            output_mat = np.loadtxt(os.path.join(FLAGS.output_dir, output_filenames[i]))

        frame_num = min(input_mat.shape[0], output_mat.shape[0])

        write_binary_file(input_mat[:frame_num, :],
            os.path.join('prepared_label', basename + '.lab'))
        write_binary_file(output_mat[:frame_num, :],
            os.path.join('prepared_cmp', basename + '.cmp'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir',
        help='Directory to read data.'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to write data.'
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=None,
        help='The dimension of binary format inputs, None for text format.'
    )
    parser.add_argument(
        '--output_dim',
        type=int,
        default=None,
        help='The dimension of binary format outputs, None for text format.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
