# INSTALL

## Installation instructions

*See [Installing TensorFlow](https://www.tensorflow.org/install/) for instructions on how to install our release binaries or how to build from source.*

To install Sonnet, you will need to compile the library using bazel against
the TensorFlow header files. You should have installed TensorFlow by
following the [TensorFlow installation instructions](https://www.tensorflow.org/install/).

This installation is compatible with Linux/Mac OS X and Python 2.7 and 3.4. The version
of TensorFlow installed must be at least 1.0.1. Installing Sonnet supports the
[virtualenv installation mode](https://www.tensorflow.org/install/install_linux#installing_with_virtualenv)
of TensorFlow, as well as the [native pip install](https://www.tensorflow.org/install/install_linux#installing_with_native_pip).

### Install bazel

Ensure you have a recent version of bazel (>= 0.4.5 ). If not, follow
[these directions](https://bazel.build/versions/master/docs/install.html).

### (virtualenv TensorFlow installation) Activate virtualenv

If using virtualenv, activate your virtualenv for the rest of the installation,
otherwise skip this step:

```shell
$ source $VIRTUALENV_PATH/bin/activate # bash, sh, ksh, or zsh
$ source $VIRTUALENV_PATH/bin/activate.csh  # csh or tcsh
```

### Configure TensorFlow Headers

First clone the Sonnet source code with TensorFlow as a submodule:

```shell
$ git clone --recursive https://github.com/deepmind/sonnet
```

and then call `configure`:

```shell
$ cd sonnet/tensorflow
$ ./configure
$ cd ../
```

You can choose the suggested defaults during the TensorFlow configuration.
Note: This will not modify your existing installation of TensorFlow. This step
is necessary so that Sonnet can build against the TensorFlow headers.

### Build and run the installer

Run the install script to create a wheel file in a temporary directory:

```shell
$ mkdir /tmp/sonnet
$ bazel build --config=opt :install
$ ./bazel-bin/install /tmp/sonnet
```

`pip install` the generated wheel file:

```shell
$ pip install /tmp/sonnet/*.whl
```

If Sonnet was already installed, uninstall prior to calling `pip install` on
the wheel file:

```shell
$ pip uninstall sonnet
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
