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

from pydaaltensorflow.net import _shape_tensor_to_daal_format
from numpy.testing import assert_array_equal
from pydaalcontrib.model.nn import *
import numpy as np
import unittest

class TestFormat(unittest.TestCase):
	def setUp(self):
		np.random.seed(0)
		
	def test_conv2d_reshaping(self):
		tf_tensor = np.random.randn(10, 10, 3, 32)
		daal_tensor = _shape_tensor_to_daal_format(tf_tensor, Conv2D())

		self.assertEqual(daal_tensor.shape, (32, 3, 10, 10))

	def test_fc_reshaping(self):
		tf_tensor = np.random.randn(1000, 2000)
		daal_tensor = _shape_tensor_to_daal_format(tf_tensor, FullyConnected())

		self.assertEqual(daal_tensor.shape, (2000, 1000))

	def test_no_reshaping(self):
		tf_tensor = np.random.randn(1000, 2000)
		daal_tensor = _shape_tensor_to_daal_format(tf_tensor, Softmax())

		assert_array_equal(daal_tensor, tf_tensor)