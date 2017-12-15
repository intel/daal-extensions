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

import re
import inspect
import numpy as np
import pydaalcontrib.nn.ops as ops
from pydaalcontrib.model.nn import *
from pydaalcontrib.constants import *
from pydaalcontrib.helpers import merge_kwargs
from .constants import *
# do not move this import to the top: libstd++ resolution error occurs
import tensorflow as tf

#########################################################################
######################## PUBLIC API GOES HERE ###########################
#########################################################################

def transform_all():
	"""Convert the Tensorflow model assuming the last op in `tf.get_default_graph().get_operations()` is creating an output.

	Returns:
		:py:class:`pydaalcontrib.model.ModelBase`: Parametrized (w/o initilized weights) model with coverted ops.
	""" 
	model = Model()
	graph = tf.get_default_graph()
	ops = graph.get_operations()
	include_op(model, None, ops[-1])
	include_identity_op(model)

	return model

def transform(tensor):
	"""Convert the Tensorflow model where `tensor` is an output.

	Args:
		tensor (:obj:`tf.Tensor`): TensorFlow model's output.

	Returns:
		:py:class:`pydaalcontrib.model.ModelBase`: Parametrized (w/o initilized weights) model with coverted ops.
	""" 
	model = Model()
	include_op(model, None, tensor.op)
	include_identity_op(model)

	return model

#########################################################################
################ TENSORFLOW SPECIFIC KNOWLEDGE GOES HERE ################
#########################################################################
tensorflow_ops = dict()

def register_op(type, func):
	op_description = dict()
	op_description['func'] = func
	op_description['args'] = inspect.getargspec(func)[0]

	if 'model' in op_description['args']:
		op_description['args'].remove('model')
	if 'last_op' in op_description['args']:
		op_description['args'].remove('last_op')
	if 'op_id' in op_description['args']:
		op_description['args'].remove('op_id')
	if 'inputs' in op_description['args']:
		op_description['args'].remove('inputs')
	if 'constants' in op_description['args']:
		op_description['args'].remove('constants')
	if 'variables' in op_description['args']:
		op_description['args'].remove('variables')

	tensorflow_ops[type] = op_description

def escape_version(var):
	return re.sub(':\d+', '', var.name)

def escape_version2(var):
	return re.sub('_\d+', '', var.name)

def escape_read(var):
	return re.sub('/read:\d+', '', var.name)

def input_type(op):
	return tuple([i.op.type for i in op.inputs])

def named_type(op):
	return '%s:name' % escape_version2(op)

def is_tensor_const(tensor):
	return is_const(tensor.op)

def is_const(op):
	return op.type == 'Const'

def is_variable(tensor):
	var_names = [escape_version(var) for var in tf.global_variables()]
	return escape_version(tensor) in var_names or escape_read(tensor) in var_names 

def get_ntype(op):
	ntype = named_type(op); keys = tensorflow_ops.keys()
	found_ntype = [t for t in keys if type(t) != tuple and t in ntype]
	return found_ntype[-1] if len(found_ntype) > 0 else None

def get_attr(op, arg):
	try:
		return op.get_attr(arg)
	except ValueError:
		return None

def add_variables(args, tensor):
	if 'variables' not in args:
		args['variables'] = []
	if is_variable(tensor):
		args['variables'].append(tensor)
	else:
		for input in tensor.op.inputs:
			if is_variable(input):
				args['variables'].append(input)

def add_constants(args, op):
	for input in op.inputs:
		if is_tensor_const(input):
			if 'constants' in args:
				args['constants'].append(input)
			else:
				args['constants'] = [input]

def add_inputs(args, op):
	for input in op.inputs:
		if not (is_tensor_const(input) or is_variable(input)):
			if 'inputs' in args:
				args['inputs'].append(input)
			else:
				args['inputs'] = [input]

def connected(model, last_op, op): 
	return op._id in model.nodes and last_op in model.nodes[op._id].outputs

def unconnected_inputs(model, last_op, op):
	return [i for i in op.inputs if not connected(model, last_op, i.op)]

def bind_args(op, type):
	op_args = tensorflow_ops[type]['args']
	return {arg: get_attr(op, arg) for arg in op_args} 

def execute_op(model, last_op, op, op_type, args):
	func = tensorflow_ops[op_type]['func']
	return func(model, last_op, op._id, **args)

def include_input_ops(model, last_op, op):
	for input in unconnected_inputs(model, last_op, op):
		include_op(model, last_op, input.op)

def include_identity_op(model):
	nodes = model.nodes.values()
	inputs = [node for node in nodes if node.depth == 0]

	if len(inputs) > 1:
		identity = Identity()
		for input in inputs:
			model._add(identity, input)

def include_op(model, last_op, op):
	type1 = op.type
	type2 = input_type(op)
	ntype = get_ntype(op)

	if ntype is not None:
		args = bind_args(op, ntype)
		add_constants(args, op)
		add_inputs(args, op)
		model, last_op = execute_op(model, last_op, op, ntype, args)
		include_input_ops(model, last_op, op)
	elif type1 in tensorflow_ops:
		args = bind_args(op, type1)
		add_constants(args, op)
		add_inputs(args, op)
		model, last_op = execute_op(model, last_op, op, type1, args)
		include_input_ops(model, last_op, op)
	elif type2 in tensorflow_ops:
		args = dict()
		for input in op.inputs:
			op_args = bind_args(input.op, type2)
			args = merge_kwargs(args, op_args)
			add_constants(args, input.op)
			add_variables(args, input)
			add_inputs(args, input.op)

		model, last_op = execute_op(model, last_op, op, type2, args)
		include_input_ops(model, last_op, op)
	else:
		for input in unconnected_inputs(model, last_op, op):
			type3 = (input.op.type, op.type) 
			
			if type3 in tensorflow_ops:
				args = dict()
				op_args1 = bind_args(input.op, type3)
				op_args2 = bind_args(op, type3)
				args = merge_kwargs(args, op_args1)
				args = merge_kwargs(args, op_args2)
				add_constants(args, input.op)
				add_variables(args, input)
				add_inputs(args, input.op)

				for i in op.inputs:
					if i != input:
						add_constants(args, i.op)
						add_variables(args, i)
						add_inputs(args, i.op)

				model, last_op = execute_op(model, last_op, op, type3, args)

			include_op(model, last_op, input.op)

def calculate_padding(inputs, kernel, strides, data_format, kernel_format, dim, window_index):
	data_index = data_format.index(dim)
	kernel_index = kernel_format.index(dim)
	stride_size = strides[window_index] 
	kernel_size = kernel[kernel_index]
	data_size = inputs[data_index]

	output_size = np.ceil(float(data_size) / float(stride_size))
	padding_size = (output_size - 1) * stride_size + kernel_size - data_size
	
	return int(padding_size/2)

def calculate_paddings(inputs, kernel, strides, data_format, kernel_format):
	# www.tensorflow.org/api_docs/python/nn/convolution#convolution
	paddings = list()
	window_index = 1

	if 'D' in data_format and 'D' in kernel_format:
		paddings.append(calculate_padding(inputs, kernel, strides, data_format, kernel_format, 'D', window_index))
		window_index += 1

	if 'H' in data_format and 'H' in kernel_format:
		paddings.append(calculate_padding(inputs, kernel, strides, data_format, kernel_format, 'H', window_index))
		window_index += 1

	if 'W' in data_format and 'W' in kernel_format:
		paddings.append(calculate_padding(inputs, kernel, strides, data_format, kernel_format, 'W', window_index))

	return paddings

def get_paddings(padding, inputs, kernel, strides, data_format, kernel_format):
	if padding.decode('ascii') == 'VALID':
		return [0] * (len(kernel_format) - 2)
	else:
	 	return calculate_paddings(inputs, kernel, strides, data_format, kernel_format)
	
def default_args(data_format):
	if data_format is None:
		# indicator of 3D conv/pool op
		# set a default one from C++ defs
		# in TF: https://tinyurl.com/lkrjcsx
		data_format = 'NDHWC'
	else:
		data_format = data_format.decode('ascii')  
	
	args = dict()
	args['input_format'] = data_format 

	if len(data_format) > 4:
		args['paddings_format'] = 'DHW'
		args['strides_format'] = 'IODHW' if 'NC' in data_format else 'IDHWO'
	else: 
		args['paddings_format'] = 'HW'
		args['strides_format'] = 'IOHW' if 'NC' in data_format else 'IHWO'

	return args, data_format

def batch_norm_op(model, last_op, op_id, inputs=None, variables=None, epsilon=None):
	kw_args = dict()
	kw_args['scale_name'] = escape_read(variables[0])
	kw_args['offset_name'] = escape_read(variables[1])
	scale = variables[0].get_shape().as_list()
	offset = variables[1].get_shape().as_list()

	op = ops._batch_norm(scale, offset, **kw_args)
	op = op.with_epsilon(epsilon) if epsilon else op

	return model._add(op.with_id(op_id), last_op), op

def fc_op(model, last_op, op_id, inputs=None, variables=None):
	kw_args = dict()
	kw_args['weights_name'] = escape_read(variables[0])
	kw_args['biases_name'] = escape_read(variables[1])
	weights = variables[0].get_shape().as_list()
	biases = variables[1].get_shape().as_list()

	op = ops._fc(weights, biases, **kw_args)
	return model._add(op.with_id(op_id), last_op), op

def conv2d_op(model, last_op, op_id, inputs=None, variables=None, strides=None, padding='VALID', data_format=None):
	kernel = variables[0].get_shape().as_list()
	biases = variables[1].get_shape().as_list()
	inputs = inputs[0].get_shape().as_list()

	kw_args, data_format = default_args(data_format)
	kw_args['kernel_name'] = escape_read(variables[0])
	kw_args['biases_name'] = escape_read(variables[1])
	kw_args['kernel_format'] = 'DHWIO' if len(data_format) > 4 else 'HWIO'

	paddings = get_paddings(padding, inputs, kernel, strides, data_format, kw_args['kernel_format'])

	op = ops._conv2d(kernel, biases, strides, paddings, **kw_args)
	return model._add(op.with_id(op_id), last_op), op

def transposed_conv2d_op(model, last_op, op_id, inputs=None, variables=None, constants=None, strides=None, padding='VALID', data_format=None):
	output_shape = tf.contrib.util.constant_value(constants[0]).tolist()
	kernel = variables[0].get_shape().as_list()
	biases = variables[1].get_shape().as_list()
	inputs = inputs[0].get_shape().as_list()

	kw_args, data_format = default_args(data_format)
	kw_args['kernel_name'] = escape_read(variables[0])
	kw_args['biases_name'] = escape_read(variables[1])
	kw_args['kernel_format'] = 'DHWOI' if len(data_format) > 4 else 'HWOI'
	kw_args['output_format'] = 'IHWO'

	paddings = get_paddings(padding, inputs, kernel, strides, data_format, kw_args['kernel_format'])

	op = ops._transposed_conv2d(kernel, biases, strides, paddings, output_shape, **kw_args)
	return model._add(op.with_id(op_id), last_op), op

def max_pool_op(model, last_op, op_id, inputs=None, ksize=None, strides=None, padding='VALID', data_format=None):
	inputs = inputs[0].get_shape().as_list()
	kw_args, data_format = default_args(data_format)
	
	kw_args['strides'] = strides
	kw_args['kernel_format'] = 'IDHWO' if len(data_format) > 4 else 'IHWO'
	kw_args['paddings'] = get_paddings(padding, inputs, ksize, strides, data_format, kw_args['kernel_format'])

	if np.sum([x > 1 for x in ksize]) > 2:
		op = ops.max_pool3d(ksize, **kw_args)
	elif np.sum([x > 1 for x in ksize]) == 2:
		op = ops.max_pool2d(ksize, **kw_args)
	else:
		kw_args['input_format'] = kw_args['input_format'].replace('H', '')
		op = ops.max_pool1d(ksize, **kw_args)
	
	return model._add(op.with_id(op_id), last_op), op

def avg_pool_op(model, last_op, op_id, inputs=None, ksize=None, strides=None, padding='VALID', data_format=None):
	inputs = inputs[0].get_shape().as_list()
	kw_args, data_format = default_args(data_format)

	kw_args['strides'] = strides
	kw_args['kernel_format'] = 'IDHWO' if len(data_format) > 4 else 'IHWO'
	kw_args['paddings'] = get_paddings(padding, inputs, ksize, strides, data_format, kw_args['kernel_format'])

	if np.sum([x > 1 for x in ksize]) > 2:
		op = ops.avg_pool3d(ksize, **kw_args)
	elif np.sum([x > 1 for x in ksize]) == 2:
		op = ops.avg_pool2d(ksize, **kw_args)
	else:
		kw_args['input_format'] = kw_args['input_format'].replace('H', '')
		op = ops.avg_pool1d(ksize, **kw_args)
	
	return model._add(op.with_id(op_id), last_op), op

def lrn_op(model, last_op, op_id, inputs=None, depth_radius=5, bias=1., alpha=1., beta=.5):
	op = ops.lrn(depth=depth_radius*2+1, bias=bias, alpha=alpha, beta=beta)
	return model._add(op.with_id(op_id), last_op), op

def relu_op(model, last_op, op_id, inputs=None):
	op = Relu().with_id(op_id)
	return model._add(op, last_op), op

def elu_op(model, last_op, op_id, inputs=None):
	op = Elu().with_id(op_id)
	return model._add(op, last_op), op

def softmax_op(model, last_op, op_id, inputs=None, dim=None):
	op = Softmax().with_dimension(dim)
	return model._add(op.with_id(op_id), last_op), op

def softplus_op(model, last_op, op_id, inputs=None):
	op = SmoothRelu().with_id(op_id)
	return model._add(op, last_op), op

def sigmoid_op(model, last_op, op_id, inputs=None):
	op = Sigmoid().with_id(op_id)
	return model._add(op, last_op), op

def tanh_op(model, last_op, op_id, inputs=None):
	op = Tanh().with_id(op_id)
	return model._add(op, last_op), op

def dropout_op(model, last_op, op_id, inputs=None, value=None):
	if value is not None and len(value.float_val) > 0:
		op = ops.dropout(value.float_val[0])
	else:
		op = ops.dropout(.5)

	return model._add(op.with_id(op_id), last_op), op

def concat_op(model, last_op, op_id, inputs=None, constants=None, axis=None):
	if inputs is None or len(inputs) < 2:
		return model, last_op

	# TODO: fix here the concat dimension to `axis` as we can change formats in Intel DAAL 
	op = Concat().with_concat_dimension(1)
	return model._add(op.with_id(op_id), last_op), op

def addn_op(model, last_op, op_id, inputs=None):
	op = ElementwiseSum().with_id(op_id)
	return model._add(op, last_op), op

def reshape_op(model, last_op, op_id, inputs=None, constants=None):
	if constants is None or len(constants) < 1:
		return model, last_op

	input_shape = list() if inputs is None else inputs[0].get_shape().as_list()

	# TODO: interim fix for removing unnecessary Reshape between fc and conv2d 
	if len(input_shape) >= 4 and isinstance(last_op, FullyConnected):
		return model, last_op

	output_shape = tf.contrib.util.constant_value(constants[-1])
	op = Reshape().with_new_dimensions(output_shape.tolist())

	if np.any([x is None for x in output_shape]):
		return model, last_op

	return model._add(op.with_id(op_id), last_op), op

def softmax_cross_entropy_op(model, last_op, op_id, inputs=None):
	op = SoftmaxCrossEntropy().with_id(op_id)
	return model._add(op, last_op), op

###############################################################################
############### REGISTER TENSORFLOW TO INTEL DAAL CONVERTIBLE OPS #############
###############################################################################
register_op(('MatMul', 'BiasAdd'), fc_op)
register_op(('MatMul', 'Identity'), fc_op)
register_op(('Conv2D', 'BiasAdd'), conv2d_op)
register_op(('Conv2D', 'Identity'), conv2d_op)
register_op(('Conv2DBackpropInput', 'BiasAdd'), transposed_conv2d_op)
register_op(('Conv2DBackpropInput', 'Identity'), transposed_conv2d_op)
register_op('SoftmaxCrossEntropyWithLogits', softmax_cross_entropy_op)
register_op('SparseSoftmaxCrossEntropyWithLogits', softmax_cross_entropy_op)
register_op('dropout/div:name', dropout_op)
register_op('FusedBatchNorm', batch_norm_op)
register_op('ConcatV2', concat_op)
register_op('Reshape', reshape_op)
register_op('MaxPool', max_pool_op)
register_op('AvgPool', avg_pool_op)
register_op('MaxPool3D', max_pool_op)
register_op('AvgPool3D', avg_pool_op)
register_op('Softplus', softplus_op)
register_op('Softmax', softmax_op)
register_op('Sigmoid', sigmoid_op)
register_op('AddN', addn_op)
register_op('Relu', relu_op)
register_op('Elu', elu_op)
register_op('Tanh', tanh_op)
register_op('LRN', lrn_op)