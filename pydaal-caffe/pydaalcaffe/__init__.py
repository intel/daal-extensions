# Copyright 2014-2017 Intel Corporation All Rights Reserved.
#
# The source code,  information  and material  ("Material") contained  herein is
# owned by Intel Corporation or its  suppliers or licensors,  and  title to such
# Material remains with Intel  Corporation or its  suppliers or  licensors.  The
# Material  contains  proprietary  information  of  Intel or  its suppliers  and
# licensors.  The Material is protected by  worldwide copyright  laws and treaty
# provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
# modified, published,  uploaded, posted, transmitted,  distributed or disclosed
# in any way without Intel's prior express written permission.  No license under
# any patent,  copyright or other  intellectual property rights  in the Material
# is granted to  or  conferred  upon  you,  either   expressly,  by implication,
# inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
# property rights must be express and approved by Intel in writing.
#
# Unless otherwise agreed by Intel in writing,  you may not remove or alter this
# notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
# suppliers or licensors in any way.

"""
PyDAAL-Caffe's main module
-------------------------------

Designed as a contribution to Intel DAAL library :py:mod:`pydaalcaffe` module aims at 
providing reusable and concise API for converting `Caffe <http://caffe.berkeleyvision.org>`__ 
specific layers/ops and variables into the Intel DAAL specific fully initialized ops/models.

Provides
	1. Helper functions, like :py:func:`transform`, :py:func:`transform_proto` and :py:func:`transform_model` to traverse and convert Caffe graphs and models.  
	2. :obj:`DAALNet` class which helps to instantiate, manage, use and convert (Deep) Neural Networks implemented using Caffe.
	3. Basic support functionality for loading and dumping PyDAAL models (included from :py:mod:`pydaalcontrib`).

Suported layers in the latest `protobuf format <https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto>`__ 
-------------------------------------------------------------------------------------------------------------------------
	1. `Convolution Layer <http://caffe.berkeleyvision.org/tutorial/layers/convolution.html>`__
	2. `Pooling Layer <http://caffe.berkeleyvision.org/tutorial/layers/pooling.html>`__
	3. `Spatial Pyramid Pooling Layer <http://caffe.berkeleyvision.org/tutorial/layers/spp.html>`__
	4. `Deconvolution Layer <http://caffe.berkeleyvision.org/tutorial/layers/deconvolution.html>`__
	5. `Fully Connected Layer <http://caffe.berkeleyvision.org/tutorial/layers/innerproduct.html>`__
	6. `Dropout Layer <http://caffe.berkeleyvision.org/tutorial/layers/dropout.html>`__
	7. `Local Response Normalization Layer <http://caffe.berkeleyvision.org/tutorial/layers/lrn.html>`__
	8. `Batch Normalization Layer <http://caffe.berkeleyvision.org/tutorial/layers/batchnorm.html>`__
	9. `Batch Normalization <http://caffe.berkeleyvision.org/tutorial/layers/batchnorm.html>`__ + `Scale <http://caffe.berkeleyvision.org/tutorial/layers/scale.html>`__ Layers 
	10. `ReLU Layer <http://caffe.berkeleyvision.org/tutorial/layers/relu.html>`__
	11. `PReLU Layer <http://caffe.berkeleyvision.org/tutorial/layers/prelu.html>`__
	12. `ELU Layer <http://caffe.berkeleyvision.org/tutorial/layers/elu.html>`__
	13. `Sigmoid Layer <http://caffe.berkeleyvision.org/tutorial/layers/sigmoid.html>`__
	14. `TanH Layer <http://caffe.berkeleyvision.org/tutorial/layers/tanh.html>`__
	15. `Absolute Value Layer <http://caffe.berkeleyvision.org/tutorial/layers/abs.html>`__
	16. `BNLL Layer <http://caffe.berkeleyvision.org/tutorial/layers/bnll.html>`__
	17. `Flatten Layer <http://caffe.berkeleyvision.org/tutorial/layers/flatten.html>`__
	18. `Reshape Layer <http://caffe.berkeleyvision.org/tutorial/layers/reshape.html>`__
	19. `Split Layer <http://caffe.berkeleyvision.org/tutorial/layers/split.html>`__
	20. `Concat Layer <http://caffe.berkeleyvision.org/tutorial/layers/concat.html>`__
	21. `Softmax Layer <http://caffe.berkeleyvision.org/tutorial/layers/softmax.html>`__
	22. `Eltwise Layer <http://caffe.berkeleyvision.org/tutorial/layers/eltwise.html>`__
	23. `Softmax with Loss Layer <http://caffe.berkeleyvision.org/tutorial/layers/softmaxwithloss.html>`__
	24. `Sigmoid Cross-Entropy Loss Layer <http://caffe.berkeleyvision.org/tutorial/layers/sigmoidcrossentropyloss.html>`__
	
Examples
--------
>>> from pydaalcaffe import DAALNet
>>> import pydaalcaffe as pydaal  
>>> import caffe
...
>>> caffe_model = caffe.NetSpec()
...
>>> with open('deploy.prototxt', 'w') as f:
...    f.write(str(caffe_model.to_proto()))
...
>>> model = pydaal.transform_proto('deploy.prototxt')
...
>>> net = DAALNet().build(model, caffe_model_path='...')
...
>>> with net.predict(test_data) as predictions:
... # do something with Caffe->DAAL `predictions`
"""

from .net import DAALNet
from .ops import transform, transform_proto, transform_model
from pydaalcontrib.helpers import dump_model, load_model

__author__ = "Vilen Jumutc"
__version__ = "2017.0"

__all__ = ['DAALNet', 'transform', 'transform_proto', 'transform_model', 'dump_model', 'load_model']