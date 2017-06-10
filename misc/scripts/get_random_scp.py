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

import random
import sys

train_ratio = 0.85
valid_ratio = 0.1
test_ratio = 0.05

raw = 'raw'

label_scp_dir = raw + '/prepared_label/label_scp/'
param_scp_dir = raw + '/prepared_cmp/param_scp/'
lst_dir = 'config/'

label_scp = open(label_scp_dir + 'all.scp')
param_scp = open(param_scp_dir + 'all.scp')

label_train = open(label_scp_dir + 'train.scp','w')
label_valid = open(label_scp_dir + 'valid.scp','w')
label_test = open(label_scp_dir + 'test.scp','w')
param_train = open(param_scp_dir + 'train.scp','w')
param_valid = open(param_scp_dir + 'valid.scp','w')
param_test = open(param_scp_dir + 'test.scp','w')

lst_train = open(lst_dir + 'train.lst','w')
lst_valid = open(lst_dir + 'valid.lst','w')
lst_test = open(lst_dir + 'test.lst','w')

lists_label = label_scp.readlines()
lists_param = param_scp.readlines()

if len(lists_label) != len(lists_param):
    print "scp files have unequal lengths"
    sys.exit(1)

lists = range(len(lists_label))
random.seed(0)
random.shuffle(lists)

train_num = int(train_ratio * len(lists))
valid_num = int(valid_ratio * len(lists))
test_num = int(test_ratio * len(lists))
train_lists = sorted(lists[: train_num])
valid_lists = sorted(lists[train_num: (train_num + valid_num)])
test_lists = sorted(lists[(train_num + valid_num):])

for i in xrange(len(lists)):
    line_label = lists_label[i]
    line_param = lists_param[i]
    line_lst = line_label.strip() + ' ' + line_param.split()[1] + '\n'
    if i in valid_lists:
        label_valid.write(line_label)
        param_valid.write(line_param)
        lst_valid.write(line_lst)
    elif i in test_lists:
        label_test.write(line_label)
        param_test.write(line_param)
        lst_test.write(line_label)
    else:
        label_train.write(line_label)
        param_train.write(line_param)
        lst_train.write(line_lst)
