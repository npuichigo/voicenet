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
import argparse
import os
import sys
import numpy as np
from scipy import signal

sys.path.append('../../src')

from utils.utils import read_binary_file, write_binary_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir',
        type=str,
        default='exp/acoustic/test',
        help="Directory of inferred test results.",
    )
    FLAGS, unparsed = parser.parse_known_args()

    inf_float = -1.0e+10

    if not os.path.exists(os.path.join(FLAGS.dir, "mgc")):
        os.mkdir(os.path.join(FLAGS.dir, 'mgc'))
    if not os.path.exists(os.path.join(FLAGS.dir, "lf0")):
        os.mkdir(os.path.join(FLAGS.dir, "lf0"))
    if not os.path.exists(os.path.join(FLAGS.dir, "bap")):
        os.mkdir(os.path.join(FLAGS.dir, "bap"))

    in_scp = open("config/test.lst")
    for line in in_scp:
        id = line.strip().split()[0]
        cmp_mat = read_binary_file(
            os.path.join(FLAGS.dir, "cmp", id + ".cmp"), dimension=75)

        mgc = signal.convolve2d(
            cmp_mat[:, : 60], [[1.0 / 3], [1.0 / 3], [1.0 / 3]], mode="same", boundary="symm")
        vuv = cmp_mat[:, 60]
        lf0 = cmp_mat[:, 65]
        bap = cmp_mat[:, 70:]

        lf0[vuv < 0.5] = inf_float

        write_binary_file(mgc, os.path.join(FLAGS.dir, "mgc", id + ".mgc"))
        write_binary_file(lf0, os.path.join(FLAGS.dir, "lf0", id + ".lf0"))
        write_binary_file(bap, os.path.join(FLAGS.dir, "bap", id + ".bap"))
