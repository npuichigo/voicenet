**Voicenet** is an open source speech synthesis framework based on tensorflow 
and sonnet. This flexible architecture lets you validate your new neural network 
based acoustic model quickly in your experiments, and owing to the deployment 
capability of tensorflow, we think it's easy to deploy new  algorithms and 
experiments online for serving. 

* [Installation Instructions](#installation-instructions)
* [Getting Started](#getting-started)
* [Why We Introduce Sonnet](#why-we-introduce-sonnet)
* [General Principles of Sonnet](#general-principles-of-sonnet)


## Installation Instructions

To install voicenet, you will need to install tensorflow>=v1.2.0 and sonnet.
*See [Installing dependencies](https://github.com/npuichigo/voicenet/blob/master/INSTALL.md) for instructions on how to install all 
the dependencies needed to use our framework.*

## Getting Started

1. Clone this repository with `git clone https://github.com/npuichigo/voicenet.git`
2. Go to `voicenet/egs/slt_arctic/` and run the script `run.sh`
3. For your own dataset, just make a new directory under `voicenet/egs/`, copy `voicenet/egs/local` and `voicenet/egs/run.sh` to the
new workspace

**WARNING**:You should change the training parameters for your own dataset. For the purpose of demostration, 
`batch_size` is set to one in `voicenet/egs/slt_arctic/`.

## Why We Introduce Sonnet

On training acoustic models in speech synthesis, we commonly do one cross-validation
iteration right after one training iteration, so variables reuse is important. However
we usually write inelegent codes like:

```python
train_model = LSTM(train_input)
# train_model and valid_model should share variables
scope.reuse_variables()
valid_model = LSTM(valid_input)
```

In order to construct models sharing variables on different inputs, we
need to use `scope.reuse_variables()` explicitly, which makes codes hard to
read and unchaste. Sonnet moves codes which construct the computation graph
to __call__ method of model class, using tf.make_template to enable variable
sharing, so it's just right for us to write concise codes with variable reusing
ability:

```python
model = LSTM(...)
train_output = model(train_input)
# variables are shared automatically
valid_output = model(valid_output)
```

**IMPORTANT**:Starting in tensorflow 1.2, dataset iterator is added for reading data into tensorflow. In using tensorflow's dataset api, iterators of dataset_train and dataset_valid can be merged into one iterator, which can be switched between different datasets conveniently, so variable reuse is no longer needed. However, we still preserve sonnet dependence for its useful abstraction and features, see *General Principles of Sonnet* for details.

```python
model = LSTM(...)
iterator = tf.contrib.data.Iterator.from_structure(
    dataset_train.batched_dataset.output_types,
    dataset_train.batched_dataset.output_shapes)
input, _ = iterator.get_next()
output = model(input)
```

## General Principles of Sonnet 

The main principle of Sonnet is to first _construct_ Python objects which
represent some part of a neural network, and then separately _connect_ these
objects into the TensorFlow computation graph. The objects are subclasses of
`sonnet.AbstractModule` and as such are referred to as `Modules`.

Modules may be connected into the graph multiple times, and any variables
declared in that module will be automatically shared on subsequent connection
calls. Low level aspects of TensorFlow which control variable sharing, including
specifying variable scope names, and using the `reuse=` flag, are abstracted
away from the user.

Separating configuration and connection allows easy construction of higher-order
Modules, i.e., modules that wrap other modules. For instance,
the `BatchApply` module merges a number of leading dimensions of a tensor into
a single dimension, connects a provided module, and then splits the leading
dimension of the result to match the input.
At construction time, the inner module is passed in as an argument to the
`BatchApply` constructor. At run time, the module first performs a reshape
operation on inputs, then applies the module passed into the constructor, and
then inverts the reshape operation.

An additional advantage of representing Modules by Python objects is that it
allows additional methods to be defined where necessary. An example of this is
a module which, after construction, may be connected in a variety of ways while
maintaining weight sharing. For instance, in the case of a generative model, we
may want to sample from the model, or calculate the log probability of a given
observation. Having both connections simultaneously requires weight sharing, and
so these methods depend on the same variables. The variables are conceptually
owned by the object, and are used by different methods of the module.
