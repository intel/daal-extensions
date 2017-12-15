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

import pydaaltensorflow as pydaal
from pydaaltensorflow import DAALNet
from numpy.testing import assert_allclose, assert_array_equal

from .conv2d_case import run_case as run_case1
from .pool_case import run_case as run_case2
from .activation_case import run_case as run_case3
from .mlp_case import run_case as run_case4
from .conv2d_transpose_case import run_case as run_case5
from .conv2d_concat_case import run_case as run_case6
from .lrn_case import run_case as run_case7
from .pool3d_case import run_case as run_case8
import tensorflow as tf
import tempfile as tmp
import numpy as np
import unittest
import shutil
import uuid
import os

class TestNN(unittest.TestCase):

	def setUp(self):
		self.net = DAALNet()
		self.checkpoint_dir = os.path.join(tmp.gettempdir(), str(uuid.uuid4()))
		os.makedirs(self.checkpoint_dir)
		np.random.seed(0)
		
	def tearDown(self):
		if os.path.exists(self.checkpoint_dir):
			shutil.rmtree(self.checkpoint_dir)

	def test_conv2d_case(self):
		test_data = np.random.randn(100, 3, 28, 28)/100
		tf_predictions = run_case1(test_data, self.checkpoint_dir)

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_array_equal(np.argmax(tf_predictions,1), np.argmax(predictions,1))
			assert_allclose(tf_predictions, predictions, rtol=1e-03)

	def test_conv2d_batch_case(self):
		test_data = np.random.randn(100, 3, 28, 28)/100
		tf_predictions = run_case1(test_data, self.checkpoint_dir)

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data, batch_size=10) as predictions:
			assert_array_equal(np.argmax(tf_predictions,1), np.argmax(predictions,1))
			assert_allclose(tf_predictions, predictions, rtol=1e-02)

	def test_pool_case(self):
		test_data = np.random.randn(100, 1, 100)
		tf_predictions = run_case2(test_data, self.checkpoint_dir)
		tf_predictions = np.transpose(tf_predictions, (0, 2, 1))

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_array_equal(tf_predictions, predictions)

	def test_activation_case(self):
		test_data = np.random.rand(100, 200)
		tf_predictions = run_case3(test_data, self.checkpoint_dir)

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_allclose(tf_predictions, predictions, atol=1e-05)

	def test_mlp_case(self):
		test_data = np.random.randn(1000, 2000)
		tf_predictions = run_case4(test_data, self.checkpoint_dir)

		self.net.build(self.checkpoint_dir)
		for _ in range(10):
			with self.net.predict(test_data) as predictions:
				assert_array_equal(np.argmax(tf_predictions,1), np.argmax(predictions,1))
				assert_allclose(tf_predictions, predictions, atol=1e-03)
			
	def test_conv2d_transpose_case(self):
		test_data = np.random.randn(100, 3, 6, 6)
		tf_predictions = run_case5(test_data, self.checkpoint_dir)
		tf_predictions = np.transpose(tf_predictions, (0, 3, 1, 2))

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_allclose(tf_predictions, predictions, atol=1e-03)

	def test_conv2d_concat_case(self):
		test_data = np.random.randn(100, 3, 28, 28)/100

		# with a sigmoid gateway node
		tf_predictions = run_case6(test_data, self.checkpoint_dir, use_gateway=True)
		tf_predictions = np.transpose(tf_predictions, (0, 3, 1, 2))

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_allclose(tf_predictions, predictions, atol=1e-04)

		# with an embedded/inferred identity gateway node
		tf_predictions = run_case6(test_data, self.checkpoint_dir, use_gateway=False)
		tf_predictions = np.transpose(tf_predictions, (0, 3, 1, 2))

		self.net = DAALNet()
		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_allclose(tf_predictions, predictions, atol=1e-04)

	def test_lrn_case(self):
		test_data = np.random.randn(100, 30, 16, 16)
		tf_predictions = run_case7(test_data, self.checkpoint_dir)
		tf_predictions = np.transpose(tf_predictions, (0, 3, 1, 2))

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_allclose(tf_predictions, predictions, atol=1e-05)

	def test_pool3d_case(self):
		test_data = np.random.randn(100, 1, 21, 21, 21)
		tf_predictions = run_case8(test_data, self.checkpoint_dir)
		tf_predictions = np.transpose(tf_predictions, (0, 4, 1, 2, 3))

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_array_equal(tf_predictions, predictions)

	def test_reshape_case(self):
		test_data = np.random.randn(100, 64, 7, 7)
		reshape = tf.reshape(tf.constant(test_data), [-1, 7*7*64])
		with tf.Session() as sess:
			tf_predictions = sess.run(reshape)

		# Transform to the Intel DAAL model
		model = pydaal.transform_all()
		pydaal.dump_model(model, self.checkpoint_dir)

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_allclose(tf_predictions, predictions, atol=1e-10)

	def test_reshape_case2(self):
		test_data = np.random.randn(100, 64, 7, 7)
		reshape = tf.reshape(tf.constant(test_data), [-1, 7, 7, 64])
		with tf.Session() as sess:
			tf_predictions = sess.run(reshape)

		# Transform to the Intel DAAL model
		model = pydaal.transform_all()
		pydaal.dump_model(model, self.checkpoint_dir)

		self.net.build(self.checkpoint_dir)

		with self.net.predict(test_data) as predictions:
			assert_allclose(tf_predictions, predictions, atol=1e-10)
			assert_allclose(test_data.reshape([-1, 7, 7, 64]), predictions, atol=1e-10)