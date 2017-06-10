#!/usr/bin/env python
# coding=utf-8
#
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

import sys
import struct
import numpy as np

cmvn_read_buffer = open(sys.argv[1], 'rb')
cmvn_write_buffer = sys.argv[1] + '_text'
header = struct.unpack('<xcccc', cmvn_read_buffer.read(5))
if header[0] != "B":
    print "Input .ark file is not binary"; exit(1)
if header[1] == "C":
    print "Input .ark file is compressed"; exit(1)

rows = 0; cols= 0
m, rows = struct.unpack('<bi', cmvn_read_buffer.read(5))
n, cols = struct.unpack('<bi', cmvn_read_buffer.read(5))

tmp_mat = np.frombuffer(cmvn_read_buffer.read(rows * cols * 4), dtype=np.float32)
cmvn_stats = np.reshape(tmp_mat, (rows, cols))

cmvn_stats.tofile(cmvn_write_buffer, sep=' ', format='%g')

cmvn_read_buffer.close()
