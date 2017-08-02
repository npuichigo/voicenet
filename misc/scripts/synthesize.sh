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

echo "$0 $@"  # Print the command line for logging

# top voicenet directory
voicenet_dir=$(dirname $(dirname $(pwd)))

# tools directory
world="${voicenet_dir}/tools/bin/World"
sptk="${voicenet_dir}/tools/bin/SPTK-3.9"

if [ $# != 1 ]; then
  echo "Usage: synthesize.sh <dir>"
  exit 1
fi

# Output features directory
out_dir=$1

mgc_dir="${out_dir}/mgc"
bap_dir="${out_dir}/bap"
lf0_dir="${out_dir}/lf0"
syn_dir="${out_dir}/syn_dir"
syn_wav_dir="${out_dir}/syn_wav"

mkdir -p ${syn_dir}
mkdir -p ${syn_wav_dir}

# initializations
fs=16000

if [ "$fs" -eq 16000 ]
then
nFFTHalf=1024
alpha=0.58
fi

if [ "$fs" -eq 48000 ]
then
nFFTHalf=2048
alpha=0.77
fi

mcsize=59
order=4

for file in $mgc_dir/*.mgc #.mgc
do
    filename="${file##*/}"
    file_id="${filename%.*}"

    echo $file_id

    ### WORLD Re-synthesis -- reconstruction of parameters ###

    ### convert lf0 to f0 ###
    $sptk/sopr -magic -1.0E+10 -EXP -MAGIC 0.0 ${lf0_dir}/$file_id.lf0 | $sptk/x2x +fa > ${syn_dir}/$file_id.syn.f0a
    $sptk/x2x +ad ${syn_dir}/$file_id.syn.f0a > ${syn_dir}/$file_id.syn.f0

    ### convert mgc to sp ###
    $sptk/mgc2sp -a $alpha -g 0 -m $mcsize -l $nFFTHalf -o 2 ${mgc_dir}/$file_id.mgc | $sptk/sopr -d 32768.0 -P | $sptk/x2x +fd > ${syn_dir}/$file_id.syn.sp

    ### convert bap to ap ###
    $sptk/mgc2sp -a $alpha -g 0 -m $order -l $nFFTHalf -o 2 ${bap_dir}/$file_id.bap | $sptk/sopr -d 32768.0 -P | $sptk/x2x +fd > ${syn_dir}/$file_id.syn.ap

    $world/synthesis ${syn_dir}/$file_id.syn.f0 ${syn_dir}/$file_id.syn.sp ${syn_dir}/$file_id.syn.ap ${syn_wav_dir}/$file_id.syn.wav
done

#rm -rf $syn_dir
