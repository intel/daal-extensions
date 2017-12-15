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
PyDAAL-contrib's module for Neural Networks
-------------------------------------------

Provides
	1. :obj:`DAALNet` class which helps to instantiate, manage and use (Deep) Neural Networks.
	2. Different support ops for Neural Networks, e.g. :py:func:`conv2d` or :py:func:`softmax_cross_entropy`.

Examples
--------
>>> from pydaalcontrib.model.nn import *
>>> from pydaalcontrib.nn import *
...
>>> DAALNet().build(model).train(...)
"""

from .net import DAALNet
from .ops import (
	fc, lc2d, conv2d, transposed_conv2d, max_pool2d, max_pool1d, avg_pool2d, stochastic_pool2d,
	avg_pool1d, concatenate, reshape, softmax_cross_entropy, dropout, lrn, lcn, elementwise_sum
)

__all__ = ['DAALNet', 'fc', 'lc2d', 'conv2d', 'transposed_conv2d', 'max_pool2d', 'max_pool1d', 'concatenate', 'elementwise_sum',
		   'avg_pool2d', 'avg_pool1d', 'stochastic_pool2d', 'reshape', 'softmax_cross_entropy', 'dropout', 'lrn', 'lcn']