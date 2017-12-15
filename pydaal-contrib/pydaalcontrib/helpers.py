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

import os
import numpy as np
import jsonpickle as jp
import jsonpickle.ext.numpy as jsonpickle_numpy
from multipledispatch import dispatch
from .constants import *

from daal.algorithms.neural_networks.layers import forward
from daal.data_management import (
	Tensor, HomogenTensor, SubtensorDescriptor, readOnly,
	HomogenNumericTable, NumericTableIface
)

try:
	basestring = basestring
except NameError:
	# 'basestring' is undefined, must be Python 3
	basestring = (str, bytes)

jsonpickle_numpy.register_handlers()

def get_model_path(path):
	"""Gets a path to the default Intel DAAL model file.

	Args:
		path (:obj:`str`): Directory or exact path where model is supposed to be found.

	Returns:
		:obj:`str`: Full model path.
	"""
	return path if PYDAAL_MODEL in path else os.path.join(path, PYDAAL_MODEL)

def dump_model(model, path):
	"""Dumps the Intel DAAL model into the specified directory with the hardcoded **pydaal.model** name and extension.

	Args:
		path (:obj:`str`): Directory name to dump the model.
	"""
	path_to_model = get_model_path(path)
	if not os.path.exists(path):
		os.makedirs(path)

	with open(path_to_model, "w") as file:
		file.write(jp.encode(model))

def load_model(path_to_model):
	"""Loads the Intel DAAL model from the specified directory.

	Args:
		path_to_model (:obj:`str`): File or Directory name to load the model from.

	Returns:
		:py:class:`pydaalcontrib.model.ModelBase`: Loaded model.

	Raises:
		 ValueError: If the path to file or directory does not exist.
	"""
	if not os.path.exists(path_to_model):
		raise ValueError(PYDAAL_MODEL_WRONG_PATH % path_to_model)
	elif os.path.isdir(path_to_model):
		path_to_model = get_model_path(path_to_model)

	with open(path_to_model, "r") as file:
		return jp.decode(file.read())

def get_learning_rate(learning_rate):
	"""Gets a learning rate which is properly wrapped for usage in Intel DAAL solvers.

	Args:
		learning_rate (:obj:`float`): Learning rate.

	Returns:
		:py:class:`daal.data_management.HomogenNumericTable`: Wrapped learning rate.
	"""
	return HomogenNumericTable(1, 1, NumericTableIface.doAllocate, learning_rate)

@dispatch(dict, dict, list)
def merge_kwargs(outer_kwargs, inner_kwargs, op_args):
	out_kwargs = inner_kwargs.copy()
	out_keys = out_kwargs.keys()
	for key in outer_kwargs.keys():
		if key not in out_keys and key in op_args:
			out_kwargs[key] = outer_kwargs[key]
		
	return out_kwargs

@dispatch(dict, dict)
def merge_kwargs(inner_kwargs, outer_kwargs):
	out_kwargs = inner_kwargs.copy()
	out_keys = out_kwargs.keys()
	for key in outer_kwargs.keys():
		if outer_kwargs[key] is not None:
			out_kwargs[key] = outer_kwargs[key]
		
	return out_kwargs

def initialize_weights(model, layer_ind, weights, biases, trainable):
	"""Helper function to initialize layer's ``weights`` and ``biases``.

	Args:
		model (:obj:`prediction.Model` or :obj:`training.Model`): The provided model [#]_.
		layer_ind (:obj:`int`): Layer's index in the model.
		weights (:obj:`numpy.ndarray`): Numpy array of ``weights``.
		baises (:obj:`numpy.ndarray`): Numpy array of ``biases``.
		trainable (:obj:`bool`): Indicator of being either :obj:`prediction.Model` or :obj:`training.Model`.

	.. [#] ``prediction.Model`` or ``training.Model`` is part of :obj:`daal.algorithms.neural_networks` package.
	"""
	if trainable:
		parameter = model.getForwardLayer(layer_ind).getLayerParameter()
	else:
		parameter = model.getLayer(layer_ind).getLayerParameter()

	# Set weights and biases of the layer
	initialize_input(model, layer_ind, weights, forward.weights, trainable)
	initialize_input(model, layer_ind, biases, forward.biases, trainable)

	# Set flag that specifies that weights and biases are initialized
	parameter.weightsAndBiasesInitialized = True

def initialize_input(model, layer_ind, input, input_id, trainable):
	"""Helper function to initialize layer's any input or variable."""
	if trainable:
		inputs = model.getForwardLayer(layer_ind).getLayerInput()
	else:
		inputs = model.getLayer(layer_ind).getLayerInput()

	initialize(inputs, input, input_id)

def initialize(inputs, input, input_id):
	if isinstance(input, np.ndarray) and len(input) > 0:
		inputs.setInput(input_id, to_tensor(input))
	elif isinstance(input, list) and len(input) > 0:
		inputs.setInput(input_id, to_tensor(np.array(input)))

def to_tensor(array):
	"""Helper function to convert from obj:`numpy.ndarray` to obj:`daal.data_management.HomogenTensor` with or w/o copying the underlying array.

	Args:
		array (:obj:`numpy.ndarray`): Numpy array.

	Returns:
		obj:`daal.data_management.HomogenTensor`: Intel DAAL tensor.
	"""
	if array.base is not None:
		# COPY of the `array` is essential to return a valid tensor (NOT A VIEW)
		return HomogenTensor(array.copy(), ntype=array.dtype)
	else:
		return HomogenTensor(array, ntype=array.dtype)

def issubdtype(array, type):
	"""Helper/shorthand function to assess the ``dtype`` of a numpy array.

	Args:
		array (:obj:`numpy.ndarray`): Numpy array.
		type (:obj:`str`): Numpy type.

	Returns:
		True if ``array`` is subtype of the provided ``type``.
	"""
	return np.issubdtype(array.dtype, type)

def in_all_types(op, lookup_types):
	return np.all([isinstance(op, type) for type in lookup_types])

def any_in_type(ops, lookup_type):
	return np.any([isinstance(op, lookup_type) for op in ops])

def find_previous_ops(current, types):
	previous_ops = [op for op in current.inputs if in_all_types(op, types)]

	if len(previous_ops) > 0:
		return np.array(previous_ops)
	else:
		previous_ops = [find_previous_ops(op, types) for op in current.inputs]
		return np.concatenate(previous_ops).flatten() if len(previous_ops) > 0 else []

class DataReader:
	"""Wrapper class for reading Intel DAAL tensors.
	
	Supported notation is ``with DataReader(...) as result:``.

	Args:
		tensor (:obj:`daal.data_management.Tensor`): Provided tensor.
		read_type (:obj:`numpy.dtype`, optional): Numpy type for the result tensor.   

	Raises:
		 ValueError: If provided argument is not a :obj:`daal.data_management.Tensor`.
	"""
	def __init__(self, tensor, read_type=np.float32):
		if not isinstance(tensor, Tensor):
			raise ValueError(PYDAAL_NOT_A_TENSOR % type(tensor))

		self.tensor = tensor
		self.block = SubtensorDescriptor(ntype=read_type)
		self.tensor.getSubtensor([], 0, tensor.getDimensionSize(0), readOnly, self.block)

	def __enter__(self):
		return self.block.getArray()

	def __exit__(self, type, value, traceback):
		self.tensor.releaseSubtensor(self.block)