#!/bin/bash

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

current_working_dir=$(pwd)
voicenet_dir=$(dirname $(dirname $current_working_dir))

stage=0
raw=raw
data=data
config=config
dir=exp/acoustic
add_delta=false
kaldi_format=false
export_graph=true

set -euo pipefail

[ ! -e $data ] && mkdir -p $data
[ ! -e $config ] && mkdir -p $config

# Make train test val data
if [ $stage -le 0 ]; then
  echo "Creating file list with scp format"
  cd ${raw}/prepared_label/ && ${voicenet_dir}/misc/scripts/create_scp.sh label && cd $current_working_dir
  cd ${raw}/prepared_cmp/ && ${voicenet_dir}/misc/scripts/create_scp.sh param && cd $current_working_dir

  echo "Randomly selecting train test val data and create config file"
  python ${voicenet_dir}/misc/scripts/get_random_scp.py

  if $kaldi_format; then
    [ -f $voicenet_dir/misc/scripts/kaldi_path.sh ] && . $voicenet_dir/misc/scripts/kaldi_path.sh;
    for x in train test val; do
    {
      convert-binary-to-matrix "ark:${raw}/prepared_label/label_scp/${x}.scp" "ark,scp:${data}/${x}_label.ark,${data}/${x}_label.scp"
      convert-binary-to-matrix "ark:${raw}/prepared_cmp/param_scp/${x}.scp" "ark,scp:${data}/${x}_param.ark,${data}/${x}_param.scp"
    }
    done

    # Add delta features
    if $add_delta; then
      for x in train test val; do
      {
        add-deltas --delta-window=1 "ark:${data}/${x}_param.ark" "ark,scp:${data}/${x}_param_delta.ark,${data}/${x}_param_delta.scp"
      }
    done
    fi

    # Do CMVN
    compute-cmvn-stats --binary=true scp:${data}/train_label.scp $dir/label_cmvn
    compute-cmvn-stats --binary=true scp:${data}/train_param.scp $dir/param_cmvn
    python $voicenet_dir/misc/scripts/convert_binary_cmvn_to_text.py ${dir}/param_cmvn
  else
    # Tfrecords format
    [ ! -e $data/train ] && mkdir -p $data/train
    [ ! -e $data/valid ] && mkdir -p $data/valid
    [ ! -e $data/test ] && mkdir -p $data/test
    # You should change the dimensions here to match your own dataset
    python ${voicenet_dir}/src/utils/convert_to_records_parallel.py --input_dim=425 --output_dim=75
  fi
fi

# Train nnet with cross-validation
if [ $stage -le 1 ]; then
  [ ! -e $dir ] && mkdir -p $dir
  [ ! -e $dir/nnet ] && mkdir -p $dir/nnet
  echo "Training nnet"
  python $voicenet_dir/src/run_tts.py --save_dir=$dir "$@"
fi

# Decode nnet
if [ $stage -le 2 ]; then
  [ ! -e $dir/test/cmp ] && mkdir -p $dir/test/cmp
  echo "Decoding nnet"
  # Disable gpu for decoding
  CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=1 python $voicenet_dir/src/run_tts.py --decode --save_dir=$dir "$@"
fi

# Vocoder synthesis
if [ $stage -le 3 ]; then
  echo "Synthesizing wav"
  python $voicenet_dir/misc/scripts/split_cmp.py --dir=$dir/test
  sh $voicenet_dir/misc/scripts/synthesize.sh $dir/test
fi

# Export graph for inference
if [ $stage -le 4 ]; then
  if $export_graph; then
    echo "Exporting graph"
    CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=1 python $voicenet_dir/src/export_inference_graph.py  --output_file=$dir/frozen_acoustic.pb --checkpoint_path=$dir/nnet "$@"
  fi
fi
