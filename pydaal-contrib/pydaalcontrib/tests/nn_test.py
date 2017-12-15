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

from pydaalcontrib.nn import *
from pydaalcontrib.model.nn import *
from pydaalcontrib.helpers import initialize_weights, initialize_input
from daal.algorithms.neural_networks.layers.batch_normalization import forward
from daal.algorithms.optimization_solver import adagrad
from daal.algorithms.neural_networks import prediction
from numpy.testing import assert_allclose, assert_array_equal

import numpy as np
import unittest

class TestDAALNet(unittest.TestCase):

	def setUp(self):
		np.random.seed(0)
		perm = np.random.permutation(200)

		self.labels = np.append(np.zeros(100), np.ones(100))
		self.labels = self.labels[perm]
		
		self.data = np.random.rand(100, 3, 60, 60)
		self.data = np.vstack((self.data, -self.data))
		self.data = self.data[perm]

	def test_topology_not_set(self):
		"""
		Test which reflects ValueError when topology is not set
		"""
		with self.assertRaises(ValueError):
			DAALNet().train(None, None)

	def test_wrong_data(self):
		"""
		Test which reflects ValueError while passing data=None and labels=None to DAALNet.train(...)
		"""
		with self.assertRaises(ValueError):
			DAALNet().build(Model()).train(None, None)

	def test_wrong_labels(self):
		"""
		Test which reflects ValueError while passing labels=None to DAALNet.train(...)
		"""
		with self.assertRaises(ValueError):
			DAALNet().build(Model()).train(np.random.rand(100), None)

	def test_spatial_pyramid_pooling(self):
		"""
		Test which reflects a successful forward-pass of the spatial pyramid 
		"""		
		model = Model(MaxPyramidPooling2D().with_pyramid_height(3))
		net = DAALNet().build(model, trainable=False)

		with net.predict(self.data) as predictions:
			assert predictions.shape == (200, 63)

	def test_lc2dnet(self):
		"""
		Test which reflects a successful forward/backward pass (one epoch) of a Deep Convolutional Net (with 2D Locally Connected nodes)
		"""
		x = lc2d([3,3,6], strides=[1,1], paddings=[2,2])
		x = x(Relu())(max_pool2d([2,2]))(dropout(.1))
		x = x(lc2d([3,3,12], strides=[1,1], paddings=[2,2]))
		x = x(Relu())(lcn())(avg_pool2d([2,2]))
		x = x(fc(10))(lrn())(fc(2))
		x = x(softmax_cross_entropy())
		
		model = Model(x)

		net = DAALNet().build(model).with_solver(adagrad.Batch())
		
		for i in range(3):
		    net.train(self.data, self.labels, batch_size=10)

		with net.predict(self.data) as predictions:
			assert np.all(np.isfinite(predictions))
			assert_array_equal(np.argmax(predictions,1), self.labels)

	def test_convnet_with_graph(self):
		"""
		Test which reflects a successful forward/backward pass (one epoch) of a Deep Convolutional Net (based on Graph)
		"""
		input = Identity()

		x = input(conv2d([3,3,32], strides=[1,1], paddings=[2,2]))
		x = x(max_pool2d([2,2]))(Relu())(dropout(.5))(lrn())

		y = input(conv2d([3,3,64], strides=[1,1], paddings=[2,2]))
		y = y(max_pool2d([2,2]))(Relu())(dropout(.5))(lrn())

		z = concatenate([x, y])
		z = z(fc(10))(lrn())(fc(2))
		z = z(softmax_cross_entropy())
		
		model = Model(z)

		net = DAALNet().build(model)
		net.train(self.data, self.labels)

		with net.predict(self.data) as predictions:
			assert np.all(np.isfinite(predictions))

	def test_mlp(self):
		"""
		Test which reflects a successful forward/backward pass (one epoch) of an ordinary MLP
		"""
		x = fc(2)(Relu())(softmax_cross_entropy())
		model = Model(x)

		net = DAALNet().build(model)
		net.train(self.data, self.labels, batch_size=10, learning_rate=.1)

		with net.predict(self.data) as predictions:
			assert np.all(np.isfinite(predictions))
			assert_array_equal(np.argmax(predictions,1), self.labels)

	def test_mlp_with_initializer(self):
		"""
		Test which reflects a successful forward/backward pass (one epoch) of an ordinary MLP with Uniform and Xavier initializers
		"""
		fc1 = fc(2).with_weights_initializer(Xavier())
		fc1 = fc1.with_biases_initializer(Uniform(0, 0))
		x = fc1(Relu())(softmax_cross_entropy())
		model = Model(x)

		assert fc1.weights().has_initializer()
		assert fc1.biases().has_initializer()

		net = DAALNet().build(model)
		net.train(self.data, self.labels, batch_size=10, learning_rate=.1)

		with net.predict(self.data) as predictions:
			assert np.all(np.isfinite(predictions))
			assert_array_equal(np.argmax(predictions,1), self.labels)

	def test_batch_normalization(self):
		"""
		Test which reflects an actual behavior of the Batch Normalization
		"""
		model = Model(BatchNormalization())

		data = np.array([[ 0.5, 0.7, 0.6 ],
			             [ 0.5, 0.4, 0.6 ],
			             [ 0.4, 0.8, 0.9 ]], dtype=np.float32)

		mean = np.mean(data, axis=0)
		variance = np.sum((data - mean)**2, axis=0) / (data.shape[0]-1)
		normalized = (data - mean)/np.sqrt(variance + 1e-5)

		parameter = prediction.Parameter(); parameter.batchSize = 3
		args = {'rebuild': {'data_dims': [3, 3], 'parameter': parameter}}

		net = DAALNet().build(model, trainable=False, **args)
		initialize_weights(net.model, 0, np.ones_like(mean), np.zeros_like(variance), False)
		initialize_input(net.model, 0, variance, forward.populationVariance, False)
		initialize_input(net.model, 0, mean, forward.populationMean, False)

		with net.predict(data, rebuild=False) as predictions:
			assert_allclose(predictions, normalized, rtol=1e-5)

	def test_elementwise_sum(self):
		"""
		Test which reflects an actual behavior of the Elementwise Sum
		"""
		x = Split().with_nsplits(2)
		op = ElementwiseSum()
		x(Identity())(op)
		x(Identity())(op)

		model = Model(op)
		data = np.random.randn(10, 3)

		net = DAALNet().build(model, trainable=False)

		with net.predict(data) as predictions:
			assert_allclose(predictions, data + data, rtol=1e-5)

		coeffs = np.random.randn(2)
		x = Split().with_nsplits(2)
		op = elementwise_sum(coeffs)
		x(Identity())(op)
		x(Identity())(op)

		model = Model(op)
		data = np.random.randn(10, 3)

		net = DAALNet().build(model, trainable=False)

		with net.predict(data) as predictions:
			assert_allclose(predictions, coeffs[0]*data + coeffs[1]*data, rtol=1e-5)