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
PyDAAL-TensorFlow's main module
-------------------------------

Designed as a contribution to Intel DAAL library :py:mod:`pydaaltensorflow` module aims at 
providing reusable and concise API for converting `TensorFlow <https://www.tensorflow.org>`__ 
specific ops and variables into the Intel DAAL specific fully initialized ops/models.

Provides
	1. Helper functions, like :py:func:`transform` and to :py:func:`transform_all` to traverse and convert TF graphs.  
	2. :obj:`DAALNet` class which helps to instantiate, manage, use and convert (Deep) Neural Networks from TF.
	3. Basic support functionality for loading and dumping PyDAAL models (included from :py:mod:`pydaalcontrib`).

Suported ops/layers
-------------------
	1. Fully Connected layer/op (based on `tf.matmul <https://www.tensorflow.org/api_docs/python/tf/matmul>`__ op)
	2. Convolution layer/op (based on `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/conv2d>`__ and `tf.nn.bias_add <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/bias_add>`__ ops)
	3. Deconvolution layer/op (based on `tf.nn.conv2d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/conv2d_transpose>`__ op)
	4. Dropout layer/op (based on `tf.nn.dropout <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/dropout>`__ op)
	5. Fused Batch Normalization layer/op (based on `tf.nn.fused_batch_norm <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/fused_batch_norm>`__ op)
	6. Concat layer/op (based on `tf.concat <https://www.tensorflow.org/versions/master/api_docs/python/tf/concat>`__ op)
	7. Reshape layer/op (based on `tf.reshape <https://www.tensorflow.org/versions/master/api_docs/python/tf/reshape>`__ op)
	8. Maximum Pooling layer/op (based on `tf.nn.max_pool <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/max_pool>`__ op)
	9. 3D Maximum Pooling layer/op (based on `tf.nn.max_pool3d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/max_pool3d>`__ op)
	10. Average Pooling layer/op (based on `tf.nn.avg_pool <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/avg_pool>`__ op)
	11. 3D Average Pooling layer/op (based on `tf.nn.avg_pool3d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/avg_pool3d>`__ op)
	12. Softplus layer/op (based on `tf.nn.softplus <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/softplus>`__ op)
	13. Softmax layer/op (based on `tf.nn.softmax <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/softmax>`__ op)
	14. Sigmoid layer/op (based on `tf.sigmoid <https://www.tensorflow.org/versions/master/api_docs/python/tf/sigmoid>`__ op)
	15. Relu layer/op (based on `tf.nn.relu <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/relu>`__ op)
	16. Elu layer/op (based on `tf.nn.elu <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/elu>`__ op)
	17. Tanh layer/op (based on `tf.tanh <https://www.tensorflow.org/versions/master/api_docs/python/tf/tanh>`__ op)
	18. AddN layer/op (based on `tf.add_n <https://www.tensorflow.org/versions/master/api_docs/python/tf/add_n>`__ op)
	19. LRN layer/op (based on `tf.nn.local_response_normalization <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/local_response_normalization>`__ op)
	20. (Sparse) Softmax Cross-Entropy Loss layer/op (based on `tf.losses.softmax_cross_entropy <https://www.tensorflow.org/versions/master/api_docs/python/tf/losses/softmax_cross_entropy>`__ and `tf.losses.sparse_softmax_cross_entropy <https://www.tensorflow.org/versions/master/api_docs/python/tf/losses/sparse_softmax_cross_entropy>`__ ops)

Examples
--------
>>> from pydaaltensorflow import DAALNet
>>> import pydaaltensorflow as pydaal  
>>> import tensorflow as tf
...
>>> def activation_net(x):
... 	x = tf.tanh(x)
... 	x = tf.sigmoid(x)
... 	x = tf.nn.softmax(x)
... 	x = tf.nn.softplus(x)
... 	return tf.nn.relu(x)
...
>>> pred = activation_net(x)
>>> model = pydaal.transform(pred)
>>> pydaal.dump_model(model, checkpoint_dir)
...
>>> net = DAALNet().build(checkpoint_dir)
...
>>> with net.predict(test_data) as predictions:
... # do something with TF->DAAL `predictions`
"""

from .net import DAALNet
from .ops import transform, transform_all
from pydaalcontrib.helpers import dump_model, load_model

__author__ = "Vilen Jumutc"
__version__ = "2017.0"

__all__ = ['DAALNet', 'transform', 'transform_all', 'dump_model', 'load_model']