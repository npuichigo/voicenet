# Voicenet: Speech Synthesis Platform
* [Overview](#overview)
* [Installation Instructions](#installation-instructions)
* [Getting Started](#getting-started)

## Overview

**Voicenet** is an open source speech synthesis framework based on tensorflow
and sonnet. This flexible architecture lets you validate your new neural network
based acoustic model quickly in your experiments, and owing to the deployment
capability of tensorflow, we think it's easy to deploy new  algorithms and
experiments online for serving.

## Installation Instructions

We have simplify the dependencies of voicenet, so you will need to install tensorflow>=v1.7.0 and progress only.
```shell
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository with `git clone https://github.com/npuichigo/voicenet.git`
2. Go to `voicenet/tools` and run the script `compile_tools.sh` to compile third_party tools
3. Go to `voicenet/egs/slt_arctic/` and run the script `run.sh`
4. For your own dataset, just make a new directory under `voicenet/egs/`, copy `voicenet/egs/local` and `voicenet/egs/run.sh` to the new workspace

**WARNING**:You should change the training parameters for your own dataset. For the purpose of demostration, `batch_size` is set to one in `voicenet/egs/slt_arctic/`.

**IMPORTANT**:We remove the dependency of sonnet in the latest version of voicenet. The main reason is that we want to keep track of tensorflow's rapid updates. In addition, starting in tensorflow 1.2, dataset iterator is added for reading data into tensorflow. In using tensorflow's dataset api, iterators of dataset_train and dataset_valid can be merged into one iterator, which can be switched between different datasets conveniently, so variable reuse is no longer needed.

```python
model = LSTM(...)
iterator = tf.contrib.data.Iterator.from_structure(
    dataset_train.batched_dataset.output_types,
    dataset_train.batched_dataset.output_shapes)
input, _ = iterator.get_next()
output = model(input)
```
