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

from os import environ as env
from .hook import report_hook
from .preprocessing import crop_center
from future.moves.urllib.request import urlretrieve
from numpy.testing import assert_allclose, assert_array_equal 
from pydaalcontrib.model.nn import SoftmaxCrossEntropy, Softmax
from pydaalcaffe.kaffe import CaffeResolver
import pydaalcaffe as pydaal
import tempfile as tmp
import numpy as np
import unittest
import shutil
import uuid
import sys
import os


class TestAlexNet(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.def_file = os.path.join(os.path.dirname(__file__), '../proto', 'alexnet.prototxt')
		cls.npz_file = os.path.join(os.path.dirname(__file__), '../data', 'tiny_imagenet.npz')
		cls.resolver = CaffeResolver()

		if 'PYDAAL_CAFFE_MODEL_DIR' in env:
			cls.data_file = os.path.join(env['PYDAAL_CAFFE_MODEL_DIR'], 'alexnet.caffemodel')
		elif cls.resolver.has_pycaffe():
			cls.data_path = os.path.join(tmp.gettempdir(), str(uuid.uuid4()))
			os.makedirs(cls.data_path)
			
			cls.data_file = os.path.join(cls.data_path, 'alexnet.caffemodel')
			print('Downloading AlexNet model from http://dl.caffe.berkeleyvision.org...')
			urlretrieve('http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel', cls.data_file, report_hook)
			print('\n')

	@classmethod
	def tearDownClass(cls):
		if 'data_path' in  cls.__dict__ and os.path.exists(cls.data_path):
			shutil.rmtree(cls.data_path)
		
	def setUp(self):
		np.random.seed(0)

	def tearDown(self):
		pass

	def test_alexnet_model_buildup_from_proto(self):
		daal_model = pydaal.transform_proto(self.__class__.def_file)

		assert len(daal_model.outputs) == 1
		assert type(daal_model.outputs[0]) == Softmax
		
		assert 'conv1' in daal_model.nodes
		assert 'conv2' in daal_model.nodes
		assert 'conv3' in daal_model.nodes
		assert 'conv4' in daal_model.nodes
		assert 'conv5' in daal_model.nodes

		assert daal_model.nodes['conv1'].depth == 0
		assert daal_model.nodes['conv2'].depth == 4
		assert daal_model.nodes['conv3'].depth == 8
		assert daal_model.nodes['conv4'].depth == 10
		assert daal_model.nodes['conv5'].depth == 12

	def test_alexnet_model_buildup_from_model(self):
		if self.__class__.resolver.has_pycaffe():
			daal_model = pydaal.transform_model(self.__class__.data_file)

			assert len(daal_model.outputs) == 1
			assert type(daal_model.outputs[0]) == SoftmaxCrossEntropy
			
			assert 'conv1' in daal_model.nodes
			assert 'conv2' in daal_model.nodes
			assert 'conv3' in daal_model.nodes
			assert 'conv4' in daal_model.nodes
			assert 'conv5' in daal_model.nodes

			assert daal_model.nodes['conv1'].depth == 0
			assert daal_model.nodes['conv2'].depth == 4
			assert daal_model.nodes['conv3'].depth == 8
			assert daal_model.nodes['conv4'].depth == 10
			assert daal_model.nodes['conv5'].depth == 12

	def test_alexnet_model_inference(self):
		if self.__class__.resolver.has_pycaffe():
			caffe = self.__class__.resolver.caffe

			images = np.load(self.__class__.npz_file)
			images = [crop_center(images[im], 227, 227) for im in images]

			# transformation is based on https://tinyurl.com/y7emcayd
			mean = np.array([104, 117, 123]).reshape((1, 3, 1, 1))
			data = np.array(images, dtype=np.float32) - mean

			data_file = self.__class__.data_file
			def_file = self.__class__.def_file
			
			net = caffe.Net(def_file, data_file, caffe.TEST)
			caffe_probs = net.forward_all(data=data)['prob']

			model = pydaal.transform_model(data_file)
			net = pydaal.DAALNet().build(model, trainable=False)

			with net.predict(data) as daal_probs:
				assert_allclose(daal_probs, caffe_probs, rtol=1e-02)
				assert_array_equal(np.argmax(daal_probs,1), np.argmax(caffe_probs,1))

			model = pydaal.transform_proto(def_file)
			net = pydaal.DAALNet().build(model, trainable=False, caffe_model_path=data_file)

			with net.predict(data) as daal_probs:
				assert_allclose(daal_probs, caffe_probs, rtol=1e-02)
				assert_array_equal(np.argmax(daal_probs,1), np.argmax(caffe_probs,1))