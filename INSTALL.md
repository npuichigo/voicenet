# INSTALL

## Installation instructions

*See [Installing TensorFlow](https://www.tensorflow.org/install/) for instructions on how to install our release binaries or how to build from source.*

Sonnet can be installed from pip, with or without GPU support, which is compatible with Linux/Mac OS X and Python 2.7 and
3.{4,5,6}. The version of TensorFlow installed must be at least 1.2. 

To install sonnet, run:

```shell
$ pip install dm-sonnet
```

Sonnet will work with both the CPU and GPU version of tensorflow, but to allow
for that it does not list Tensorflow as a requirement, so you need to install
Tensorflow separately if you haven't already done so.

If Sonnet was already installed, uninstall prior to calling `pip install` on
the wheel file:

```shell
$ pip uninstall dm-sonnet
```

You can verify that Sonnet has been successfully installed by, for example,
trying out the resampler op:

```shell
$ cd ~/
$ python
>>> import sonnet as snt
>>> import tensorflow as tf
>>> snt.resampler(tf.constant([0.]), tf.constant([0.]))
```

The expected output should be:

```shell
<tf.Tensor 'resampler/Resampler:0' shape=(1,) dtype=float32>
```

However, if an `ImportError` is raised then the C++ components were not found.
Ensure that you are not importing the cloned source code (i.e. call python
outside of the cloned repository) and that you have uninstalled Sonnet prior to
installing the wheel file.
