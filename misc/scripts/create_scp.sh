#!/bin/bash
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

if [[ $# -ne 1 ]]; then
    echo "Usage: ./create_scp.sh [label|param]"
    exit 1
fi

if [[ "$1" != label && "$1" != param ]]; then
    echo "Usage: ./create_scp.sh [label|param]"
    exit 1
fi

dir=${1}_scp

ext=
if [[ "$1" == label ]]; then
    ext=lab
else
    ext=cmp
fi

set -euo pipefail

[[ ! -e $dir ]] && mkdir -p $dir

> ${dir}/all.scp
for filename in *.${ext}; do
    basename=${filename%.*}
    echo $basename $(pwd)/$filename >> ${dir}/all.scp 
done
