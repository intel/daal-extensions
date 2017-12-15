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

from ..model.nn import Model
from ..builder import build_topology
from ..helpers import load_model, issubdtype, get_learning_rate
from daal.algorithms.neural_networks import prediction, training
from daal.algorithms.optimization_solver import sgd
from daal.data_management import Tensor, HomogenTensor, SubtensorDescriptor, readOnly
from multipledispatch import dispatch
import numpy as np

try:
	basestring = basestring
except NameError:
	# 'basestring' is undefined, must be Python 3
	basestring = (str,bytes)

class DAALNet:
	"""Wrapper class for working with :obj:`daal.algorithms.neural_networks` package.

	Notes
		Default working regime is training, see :obj:`daal.algorithms.neural_networks.training.Batch()`.
		Default solver used for training is SGD, see :obj:`daal.algorithms.optimization_solver.sgd.Batch()`.
	"""
	_daal_net_namespace = dict()

	def __init__(self):
		#TODO: set do_rebuild=False once memory allocation is fixed on prediction in Intel DAAL 2018
		self.do_rebuild = True
		self.initializer = None
		self.solver = sgd.Batch()
		self.net = training.Batch(self.solver)

	def with_solver(self, solver):
		"""Provides a specific solver for the Intel DAAL net/graph.

		Args:
			solver (from :obj:`daal.algorithms.optimization_solver` module): Intel DAAL solver.

		Returns:
			:py:class:`pydaalcontrib.nn.DAALNet`: Intel DAAL network with the provided solver.
		"""
		self.solver = solver
		self.net = training.Batch(self.solver)

		return self

	def with_initializer(self, initializer):
		self.initializer = initializer
		return self

	def train(self, data, labels, **kw_args):
		"""Trains a specific Intel DAAL net/graph based on the provided data and labels.

		Args:
			data (:obj:`daal.data_management.Tensor` or :obj:`numpy.ndarray`): Training data.
			labels (:obj:`daal.data_management.Tensor` or :obj:`numpy.ndarray`): Training labels.
			**kwargs: Arbitrary keyword arguments (``batch_size`` and ``learning_rate``).

		Returns:
			:py:class:`pydaalcontrib.nn.DAALNet`: Trained DAAL network.

		Raises:
			 ValueError: If the provided ``data`` or ``labels`` are of the wrong type or the topology is not set.
		"""
		if 'topology' not in self.__dict__:
			raise ValueError('Topology is not intialized!')
		if 'batch_size' in kw_args and 'result' not in self.__dict__:
			self.solver.parameter.batchSize = kw_args['batch_size']
		if 'learning_rate' in kw_args and 'learningRate' in self.solver.parameter.__swig_getmethods__:
			self.solver.parameter.learningRate = get_learning_rate(kw_args['learning_rate'])
		if 'learning_rate' in kw_args and 'learningRateSequence' in self.solver.parameter.__swig_getmethods__:
			self.solver.parameter.learningRateSequence = get_learning_rate(kw_args['learning_rate'])

		if isinstance(data, Tensor):
			self.data = data
		elif isinstance(data, np.ndarray) and data.base is not None:
			self.data = HomogenTensor(data.copy(), ntype=data.dtype)
		elif isinstance(data, np.ndarray) and data.base is None:
			self.data = HomogenTensor(data, ntype=data.dtype)
		else:
			raise ValueError('Data is not of numpy.ndarray or Tensor type!')

		if isinstance(labels, Tensor):
			self.labels = labels
		elif isinstance(labels, np.ndarray):
			if len(labels.shape) == 1:
				labels = labels.reshape([-1, 1])
			if issubdtype(labels, np.int):
				labels = labels.astype(np.intc)
			elif not issubdtype(labels, np.float):
				labels = labels.astype(np.float)

			self.labels = HomogenTensor(labels.copy(), ntype=labels.dtype)
		else:
			raise ValueError('Labels are not of numpy.ndarray or Tensor type!')

		if  'train_result' not in self.__dict__ or self.train_result is None:
			dims = self.data.getDimensions()[1:]
			dims.insert(0, self.solver.parameter.batchSize)
			self.net.initialize(dims, self.topology)

			# heuristically define the number of iterations for ``self.solver``
			batch_size = np.float(self.solver.parameter.batchSize) 
			n_iter = np.ceil(self.data.getDimensionSize(0)/batch_size)
			self.solver.parameter.nIterations = np.int(n_iter)

		# Pass a solver, training data and lables to the algorithm
		self.net.parameter.optimizationSolver = self.solver
		self.net.input.setInput(training.data, self.data)
		self.net.input.setInput(training.groundTruth, self.labels)

		# Do an actual compute and store the result
		self.train_result = self.net.compute()
		self.do_rebuild = False

		return self

	#TODO: refactor set rebuild=False once memory allocation is fixed on prediction in Intel DAAL 2018
	def predict(self, data, batch_size=None, rebuild=True):
		"""Predicts labels based on a prediction model.

		Supported notation is ``with net.predict(...) as predictions:``

		Args:
			data (:obj:`daal.data_management.Tensor` or :obj:`numpy.ndarray`): Prediction data.
			batch_size (:obj:`int`): Batch size for processing prediction data.
			rebuild (:obj:`bool`): Control parameter to force rebuild of the model.

		Returns:
			:py:class:`pydaalcontrib.nn.DAALNet`: DAAL network with the evaluated predictions.   

		Raises:
			 ValueError: If the provided ``data`` are of the wrong type.
		"""
		if isinstance(data, np.ndarray):
			_data = HomogenTensor(data.copy(), ntype=data.dtype)
		elif not isinstance(data, Tensor):
			raise ValueError('Data is not of numpy.ndarray or Tensor type!')

		if not batch_size or batch_size > _data.getDimensionSize(0):
			batch_size = _data.getDimensionSize(0)

		if rebuild and self.do_rebuild:
			#TODO: refactor set rebuild=False once memory allocation is fixed on prediction in Intel DAAL 2018
			parameter = prediction.Parameter(); parameter.batchSize = batch_size; self.do_rebuild = False
			rebuild_args = {'data_dims': [batch_size] + _data.getDimensions()[1:], 'parameter': parameter}
			self.model = self.build_model(self.descriptor, False, rebuild=rebuild_args, **self.build_args)
		elif 'train_result' in self.__dict__:
			self.model = self.train_result.get(training.model).getPredictionModel_Float32()

		net = prediction.Batch()
		net.parameter.batchSize = batch_size
		net.input.setModelInput(prediction.model, self.model)
		net.input.setTensorInput(prediction.data, _data)

		self.predictions = SubtensorDescriptor(ntype=data.dtype)
		self.predict_result = net.compute().getResult(prediction.prediction)
		self.predict_result.getSubtensor([], 0, self.predict_result.getDimensionSize(0), readOnly, self.predictions)

		return self

	def get_predictions(self):
		"""Gets the latest predictions after :py:meth:`predict` was called.

		Returns:
			:py:obj:`numpy.ndarray`: Evaluated predictions.
		"""
		if 'predictions' in self.__dict__:
			predictions_numpy = self.predictions.getArray()
			self.predict_result.releaseSubtensor(self.predictions)
			return predictions_numpy
		else:
			return None

	def __enter__(self):
		return self.predictions.getArray()

	def __exit__(self, type, value, traceback):
		self.predict_result.releaseSubtensor(self.predictions)

	def allocate_model(self, model, args):
		"""Allocates a contiguous memory for the model if 'rebuild' option is specified.

		Args:
			model (:obj:`daal.algorithms.neural_networks.prediction.Model`): instantiated model.
			args (:obj:`dict`): Different args which are passed from :py:func:`build_model`.

		Returns:
			:obj:`daal.algorithms.neural_networks.prediction.Model`
		"""
		if 'rebuild' in args:
			parameter = args['rebuild']['parameter']
			data_dims = args['rebuild']['data_dims']
			model.allocate_Float32(data_dims, parameter)

		return model

	def build_model(self, model, trainable, **kw_args):
		"""(re)Builds a specific Intel DAAL model based on the provided descriptor.

		Args:
			model (:py:class:`pydaalcontrib.model.ModelBase` or :obj:`str`): Instance of a model or a path to the folder/file containing the model (*pydaal.model*) file.
			trainable (:obj:`bool`): Flag indicating whether `training` or `prediction` topology to be built.
			kw_args (:obj:`dict`): Different keyword args which might be of use in sub-classes.

		Returns:
			:obj:`daal.algorithms.neural_networks.prediction.Model` or ``None``
		"""
		if 'model' in self.__dict__:
			return self.allocate_model(self.model, kw_args)

		if isinstance(model, basestring):
			self.descriptor = load_model(model)
		else:
			self.descriptor = model

		self.topology = build_topology(self.descriptor, trainable, initializer=self.initializer)
		#TODO: replace with training.Model(topology) once fixed
		return None if trainable else self.allocate_model(prediction.Model(self.topology), kw_args)

	@dispatch(basestring, namespace=_daal_net_namespace)
	def build(self, model_path, trainable=False, **kw_args):
		self.model = self.build_model(model_path, trainable, **kw_args)
		self.build_args = {'model_path': model_path} 
		self.build_args.update(kw_args)

		return self

	@dispatch(Model, namespace=_daal_net_namespace)
	def build(self, model, trainable=True, **kw_args):
		self.model = self.build_model(model, trainable, **kw_args)
		self.build_args = kw_args

		return self