Welcome to Intel® Data Analytics Acceleration Library Extensions home pages!
===============================================================================

Intel® DAAL Extensions aim at providing reusable and easily understood APIs for Data Science and Machine Learning communities. It alleviates usage of the [Intel® Data Analytics Acceleration Library (Intel® DAAL)](http://software.intel.com/en-us/intel-daal) and abstracts away some of the inner tedious API details. The main focus of the library is interpretability, ease of use and immediate applicability of the Intel DAAL primitives, data structures and functionality.

The main emphasis is put on reusing and enabling usage of the [PyDAAL](https://anaconda.org/intel/pydaal) wrapper and [Neural Nets package](https://software.intel.com/sites/products/documentation/doclib/daal/daal-user-and-reference-guides/daal_prog_guide/GUID-E8F6E40F-60EC-422B-8D46-492E110BB0BD.htm) in a *pythonic* way for Python-centric technical computing workflows, frameworks and environments, e.g. [Anaconda](https://www.continuum.io/anaconda-overview) or [Scikit-Learn](http://scikit-learn.org>).

Intel® DAAL Extensions is open source and also available in the Intel channel on [Anaconda](http://anaconda.org). It is currently supported for Python 2.7, 3.5 and 3.6 on Linux/MacOS/Windows. It is recommended to use [Intel® Distribution for Python](https://software.intel.com/en-us/distribution-for-python) in conjunction with Intel® DAAL Extensions.

Another important focus is on connecting Intel® DAAL to other very prominent and recognized frameworks, *e.g.* converting [TensorFlow](https://www.tensorflow.org) or [Caffe](http://caffe.berkeleyvision.org) specific ops and models to the Intel DAAL primitives and models. Here we aim at providing a unified approach for describing any computational model in the Intel DAAL specific terms. It makes it easy to assemble any fully initialized model, for instance a Deep Convolutional Net, in terms of the Intel DAAL primitives out of Tensorflow/Caffe checkpoints or model files. 

Usage
------------------

As a simple example of various features provided by Intel® DAAL Extensions, a few lines of code below demonstrate how to train a [Deep Convolutional Net](https://en.wikipedia.org/wiki/Convolutional_neural_network) on some superficial dataset using [pydaalcontrib](pydaal-contrib) module: 

```python

from pydaalcontrib.nn import *
from pydaalcontrib.model.nn import (
  Model, Relu, SigmoidCrossEntropy, Gaussian, Uniform, Xavier, TruncatedGaussian
)
from daal.algorithms.optimization_solver import adagrad
from numpy.testing import assert_equal
import numpy as np

x = conv2d([3,3,6], strides=[2,2], paddings=[2,2])
x = x.with_weights_initializer(Gaussian(0, 1./100))
x = x.with_biases_initializer(Uniform(0, 0))
x = x(max_pool2d([2,2]))(Relu())(dropout(.5))
x = x(conv2d([3,3,12], strides=[2,2], paddings=[2,2]))
x = x.with_weights_initializer(Xavier())
x = x.with_biases_initializer(Xavier())
x = x(max_pool2d([2,2]))(Relu())(dropout(.5))
x = x(conv2d([3,3,24], strides=[2,2], paddings=[2,2]))
x = x.with_weights_initializer(TruncatedGaussian(0, 1./100).with_bounds(-2, 2))
x = x.with_biases_initializer(Uniform(0, 0))
x = x(avg_pool2d([2,2]))(Relu())
x = x(fc(10))(lrn())(fc(1))
x = x(SigmoidCrossEntropy())

model = Model(x)

num_epochs = 10
np.random.seed(0)
labels = np.append(np.zeros(100), np.ones(100))
data = np.random.rand(200, 3, 60, 60)
data[:100] = data[:100]*-1

net = DAALNet().build(model).with_solver(adagrad.Batch())

for i in range(num_epochs):
  perm = np.random.permutation(200)
  net.train(data[perm], labels[perm], batch_size=10)

with net.predict(data) as predictions:
  assert_equal(np.rint(predictions).ravel(), labels)
```

Please find more examples of using [pydaalcontrib](pydaal-contrib) and [pydaaltensorflow](pydaal-tensorflow) modules in these jupyter [noteboooks](docs/source/examples/notebooks).

Installation
----------------

Issue the following commands to get started on Linux:

```bash

git clone https://github.com/01org/daal-extensions.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/daal-extensions/pydaal-contrib
export PYTHONPATH=$PYTHONPATH:$(pwd)/daal-extensions/pydaal-tensorflow
export PYTHONPATH=$PYTHONPATH:$(pwd)/daal-extensions/pydaal-caffe
```

Afterwards, importing `pydaalcontrib`, `pydaaltensorflow` and `pydaalcaffe` should work in Python:

```bash

python -c 'import pydaalcontrib'
```

For a proper installation, run the following command (from the corresponding module's root folder) to install any of the aforementioned python modules, *e.g.* `pydaalcontrib`:

```bash

python setup.py install
```

or, if you have pip, by executing the following command:

```bash

pip install -e .
```

Installation may require superuser priviliges.

Docker images
------------------

It is possible to use Intel® DAAL Extensions from within a Docker image by building it using the provided Dockerfile within each module. 

Please execute first from within ``daal-extensions/pydaal-contrib`` directory:

```bash

docker build -t intel/pydaal-contrib .
```

Then you can build other images by inhereting from the tag: ``intel/pydaal-contrib``.