Welcome to Intel® Data Analytics Acceleration Library Extensions documentation!
===============================================================================

Intel® DAAL Extensions aim at providing reusable and easily understood APIs for Data Science and Machine Learning communities. It alleviates usage of the `Intel® Data Analytics Acceleration Library (Intel® DAAL) <http://software.intel.com/en-us/intel-daal>`__ and abstracts away some of the inner tedious API details. The main focus of the library is interpretability, ease of use and immediate applicability of the Intel DAAL primitives, data structures and functionality.

The main emphasis is put on reusing and enabling usage of the `PyDAAL <https://anaconda.org/intel/pydaal>`__ wrapper and `Neural Nets package <https://software.intel.com/sites/products/documentation/doclib/daal/daal-user-and-reference-guides/daal_prog_guide/GUID-E8F6E40F-60EC-422B-8D46-492E110BB0BD.htm>`__ in a *pythonic* way for Python-centric technical computing workflows, frameworks and environments, e.g. `Anaconda <https://www.continuum.io/anaconda-overview>`__ or `Scikit-Learn <http://scikit-learn.org>`__.

Intel® DAAL Extensions is open source and also available in the Intel channel on `Anaconda <http://anaconda.org>`__. It is currently supported for Python 2.7, 3.5 and 3.6 on Linux/MacOS/Windows. It is recommended to use `Intel® Distribution for Python <https://software.intel.com/en-us/distribution-for-python>`__ in conjunction with Intel® DAAL Extensions.

Another important focus is on connecting Intel® DAAL to other very prominent and recognized frameworks, *e.g.* converting `TensorFlow <https://www.tensorflow.org>`__ or `Caffe <http://caffe.berkeleyvision.org>`__ specific ops and models to the Intel DAAL primitives and models. Here we aim at providing a unified approach for describing any computational model in the Intel DAAL specific terms. It makes it easy to assemble any fully initialized model, for instance a Deep Convolutional Net, in terms of the Intel DAAL primitives out of Tensorflow/Caffe checkpoints or model files. 

Example
------------------

As a simple example of various features provided by Intel® DAAL Extensions, a few lines of code below demonstrate how to train a `Deep Convolutional Net <https://en.wikipedia.org/wiki/Convolutional_neural_network>`__ on some superficial dataset using :py:mod:`pydaalcontrib` module: 

.. literalinclude:: /examples/deep_nn.py
    :language: python
    :emphasize-lines: 24,32,36,38

Please find more examples of using :py:mod:`pydaalcontrib` :ref:`module <examples/notebooks/DAAL extensions demo.ipynb>` and :py:mod:`pydaaltensorflow` :ref:`module <examples/notebooks/Tensorflow to DAAL conversion demo.ipynb>` in our `Jupyter <http://jupyter.readthedocs.io>`__ notebooks.

Quick setup
----------------

Issue the following commands to get started on Linux:

.. code-block:: bash

    git clone https://github.com/01org/daal-extensions.git
    export PYTHONPATH=$PYTHONPATH:$(pwd)/daal-extensions/pydaal-contrib
    export PYTHONPATH=$PYTHONPATH:$(pwd)/daal-extensions/pydaal-tensorflow
    export PYTHONPATH=$PYTHONPATH:$(pwd)/daal-extensions/pydaal-caffe

Afterwards, importing :py:mod:`pydaalcontrib`, :py:mod:`pydaaltensorflow` and :py:mod:`pydaalcaffe` should work in Python:

.. code-block:: bash

    python -c 'import pydaalcontrib'

For a proper installation, run the following command (from the corresponding module's root folder) to install any of the aforementioned python modules, *e.g.* :py:mod:`pydaalcontrib`:

.. code-block:: bash

    python setup.py install

or, if you have pip, by executing the following command:

.. code-block:: bash

    pip install -e .

Installation may require superuser priviliges.

Docker images
------------------

It is possible to use Intel® DAAL Extensions from within a Docker image by building it using the provided Dockerfile within each module. 

Please execute first from within ``daal-extensions/pydaal-contrib`` directory:

.. code-block:: bash

    docker build -t intel/pydaal-contrib . 

Then you can build other images by inhereting from the tag: ``intel/pydaal-contrib``. 

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :hidden:
   :maxdepth: 4

   pydaal-contrib/pydaalcontrib
   pydaal-tensorflow/pydaaltensorflow
   pydaal-caffe/pydaalcaffe