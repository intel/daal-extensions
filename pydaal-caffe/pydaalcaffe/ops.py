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

import copy
import numpy as np
import pydaalcontrib.nn.ops as ops
import google.protobuf.text_format as txtf
from pydaalcontrib.model.nn import *
from pydaalcontrib.constants import *
from .kaffe import CaffeResolver
from .constants import *
from .ring import Ring

#########################################################################
######################## PUBLIC API GOES HERE ###########################
#########################################################################
caffe_namespace = dict()
caffe_resolver = CaffeResolver()

def transform_model(path_to_model):
	"""Convert the Caffe model at `path_to_model` into the Intel DAAL model.

	Args:
		path_to_model (:obj:`str`): Exact path where Caffe's model is supposed to be found.

	Returns:
		:py:class:`pydaalcontrib.model.ModelBase`: Parametrized (w/o initilized weights) model with coverted layers.
	""" 
	caffe_model = caffe_resolver.NetParameter()

	with open(path_to_model, 'rb') as file:
		caffe_model.MergeFromString(file.read())

	return transform(caffe_model, caffe_model_path=path_to_model)

def transform_proto(path_to_proto):
	"""Convert the Caffe model definition at `path_to_proto` into the Intel DAAL model.

	Args:
		path_to_proto (:obj:`str`): Exact path where Caffe's model definition is supposed to be found.

	Returns:
		:py:class:`pydaalcontrib.model.ModelBase`: Parametrized (w/o initilized weights) model with coverted layers.
	""" 
	caffe_model = caffe_resolver.NetParameter()

	with open(path_to_proto, 'r') as file:
		txtf.Merge(file.read(), caffe_model)

	return transform(caffe_model)

def transform(caffe_model, **kw_args):
	"""Convert Caffe's model given by `caffe_model` object into the Intel DAAL model.

	Args:
		caffe_model (:obj:`caffe.proto.caffe_pb2.NetParameter`): Caffe model loaded from proto file.
		kw_args (:obj:`dict`): Different keyword args which are passed on to the Intel DAAL.
		
	Returns:
		:py:class:`pydaalcontrib.model.ModelBase`: Parametrized (w/o initilized weights) model with coverted layers.
	""" 
	daal_model = Model()
	for key in kw_args:
		daal_model.__dict__[key] = kw_args[key]

	return process_ops(caffe_model, daal_model)

####################################################################
################ CAFFE SPECIFIC KNOWLEDGE GOES HERE ################
####################################################################
caffe_ops = dict()

def overwrites_bottom(layer): 
	return np.any([t in layer.bottom for t in layer.top])

def process_ops(caffe_model, daal_model):
	tops = dict()
	layers = dict()
	bottoms = set()
	# fetch proper layer definitions (LayerParameter vs. V1LayerParameter)
	caffe_layers = caffe_model.layer or caffe_model.layers

	for layer in caffe_layers:
		layers[layer.name] = layer
		bottom = list(layer.bottom)
		bottoms |= set(bottom)
	
	for layer in filter(overwrites_bottom, caffe_layers):
		for top in layer.top:
			if top in tops:
				tops[top].add(layer)
			else:
				tops[top] = Ring([layers[top], layer])

	# We start only with purely output layers and add ops in reverse order 
	is_output = lambda l: l.name not in bottoms and not overwrites_bottom(l)
	outputs = [layer for layer in caffe_layers if is_output(layer)]

	for output in outputs:
		include_op(daal_model, None, output, layers, tops)

	return daal_model

def include_op(model, last_op, layer_op, layers, tops):
	model, last_op = execute_op(model, last_op, layer_op, layer_op.type)
	connected = lambda id: id in model.nodes and last_op in model.nodes[id].outputs

	for bottom in [b for b in layer_op.bottom if b in layers]:
		if bottom not in tops and not connected(bottom):
			include_op(model, last_op, layers[bottom], layers, tops)
		elif bottom in tops and not connected(tops[bottom][-1].name): 
			# resolve top/bottom layer output overwriting issue  
			include_op(model, last_op, tops[bottom].last(), layers, tops)

def register_op(type, func):
	op_description = dict()
	op_description['func'] = func
	caffe_ops[type] = op_description

def execute_op(model, last_op, layer_op, op_type):
	if op_type in caffe_ops:
		func = caffe_ops[op_type]['func']
		return func(model, last_op, layer_op)
	else:
		return model, last_op
	
def default_args(param):
	kernel =  param.kernel_size
	args = dict()

	if type(kernel) != int and len(kernel) > 2:
		args['kernel_format'] = 'DHWO'
		args['paddings_format'] = 'DHW'
		args['strides_format'] = 'DHW'
		args['input_format'] = 'NCDHW'
	else:
		args['kernel_format'] = 'HWO'
		args['paddings_format'] = 'HW'
		args['strides_format'] = 'HW'
		args['input_format'] = 'NCHW'

	return args

def get_kernel(param, channels=None, **args):
	ch = [channels] if channels else []
	if type(param.kernel_size) == int:
		kernel = [param.kernel_size]
	else: 
		kernel = list(param.kernel_size)

	if len(kernel) == 0:
		# 2D case 
		return [param.kernel_h, param.kernel_w] + ch
	elif len(kernel) == 1:
		# ND generic case
		format = args['kernel_format']
		return kernel*(len(format)-1) + ch
	else:
		# ND specific case
		return kernel + ch

def get_paddings(param, **args):
	if type(param.pad) == int:
		paddings = [param.pad]
	else: 
		paddings = list(param.pad)

	if len(paddings) == 0:
		# 2D case 
		return [param.pad_h, param.pad_w]
	elif len(paddings) == 1:
		# ND generic case
		format = args['paddings_format']
		return paddings*len(format)
	else:
		# ND specific case
		return paddings

def get_strides(param, **args):
	if type(param.stride) == int:
		strides = [param.stride]
	else: 
		strides = list(param.stride)

	format = args['strides_format']

	if len(strides) == 0 and param.stride_h and param.stride_w:
		# 2D case 
		return [param.stride_h, param.stride_w]
	elif len(strides) == 0:
		return [1]*len(format)
	elif len(strides) == 1:
		# ND generic case
		return strides*len(format)
	else:
		# ND specific case
		return strides

# TODO: refactor when actual Scale layer is added to Intel DAAL 2018
def scale_op(model, last_op, layer_op):
	op = Scale().with_param(layer_op.scale_param)
	return model._add(op.with_id(layer_op.name), last_op), op

# TODO: refactor when actual Scale layer is added to Intel DAAL 2018
def batch_norm_op(model, last_op, layer_op):
	op = BatchNormalization()
	op = op.with_epsilon(layer_op.batch_norm_param.eps)
	op = op.with_alpha(1 - layer_op.batch_norm_param.moving_average_fraction)

	if isinstance(last_op, Scale):
		scale_param = last_op.get_param()
		last_op.with_param({'bias_term': scale_param.bias_term})
		weights_initializer = ops._initializer(scale_param.filler)
		op = op.with_weights(NodeDescriptor(PYDAAL_WEIGHTS_KERNEL, [None]))
		op = op.with_weights_initializer(weights_initializer)
		op = op.with_scale_op(last_op.id)
		
		if scale_param.bias_term:
			biases_initializer = ops._initializer(scale_param.bias_filler)
			op = op.with_biases(NodeDescriptor(PYDAAL_BIASES_KERNEL, [None]))
			op = op.with_biases_initializer(biases_initializer)
		else:
			op = op.with_biases(EmptyNodeDescriptor(PYDAAL_BIASES_KERNEL, [None]))

	return model._add(op.with_id(layer_op.name), last_op), op

def fc_op(model, last_op, layer_op):
	weights_initializer = ops._initializer(layer_op.inner_product_param.weight_filler)
	biases_initializer = ops._initializer(layer_op.inner_product_param.bias_filler)
	use_bias = layer_op.inner_product_param.bias_term

	op = ops.fc(layer_op.inner_product_param.num_output, use_bias)
	op = op.with_weights_initializer(weights_initializer)

	if use_bias:
		op = op.with_biases_initializer(biases_initializer)

	return model._add(op.with_id(layer_op.name), last_op), op

def conv2d_op(model, last_op, layer_op):
	kernel_initializer = ops._initializer(layer_op.convolution_param.weight_filler)
	biases_initializer = ops._initializer(layer_op.convolution_param.bias_filler) 
	kw_args = default_args(layer_op.convolution_param)
	channels = layer_op.convolution_param.num_output
	use_bias = layer_op.convolution_param.bias_term
	kw_args['use_bias'] = use_bias

	kernel = get_kernel(layer_op.convolution_param, channels, **kw_args)
	strides = get_strides(layer_op.convolution_param, **kw_args)
	paddings = get_paddings(layer_op.convolution_param, **kw_args)

	op = ops.conv2d(kernel, strides, paddings, **kw_args)
	op = op.with_weights_initializer(kernel_initializer)
	op = op.with_group(layer_op.convolution_param.group)

	if use_bias:
		op = op.with_biases_initializer(biases_initializer)

	return model._add(op.with_id(layer_op.name), last_op), op

def transposed_conv2d_op(model, last_op, layer_op):
	kernel_initializer = ops._initializer(layer_op.convolution_param.weight_filler)
	biases_initializer = ops._initializer(layer_op.convolution_param.bias_filler) 
	kw_args = default_args(layer_op.convolution_param)
	channels = layer_op.convolution_param.num_output
	use_bias = layer_op.convolution_param.bias_term
	kw_args['use_bias'] = use_bias

	kernel = get_kernel(layer_op.convolution_param, channels, **kw_args)
	strides = get_strides(layer_op.convolution_param, **kw_args)
	paddings = get_paddings(layer_op.convolution_param, **kw_args)
	kw_args['output_format'] = 'NCDHW' if len(kernel) > 2 else 'NCHW'

	### TODO: THIS OP WILL FAIL AS NO OUTPUT_SHAPE IS AVAILABLE AT DEFINITION
	op = ops.transposed_conv2d(kernel, [], strides, paddings, **kw_args)
	op = op.with_weights_initializer(kernel_initializer)
	op = op.with_group(layer_op.convolution_param.group)

	if use_bias:
		op = op.with_biases_initializer(biases_initializer)

	return model._add(op.with_id(layer_op.name), last_op), op

def pool_op(model, last_op, layer_op):
	kw_args = default_args(layer_op.pooling_param)	
	kernel = get_kernel(layer_op.pooling_param, None, **kw_args)
	strides = get_strides(layer_op.pooling_param, **kw_args)
	paddings = get_paddings(layer_op.pooling_param, **kw_args)

	if layer_op.pooling_param.pool == 0: #'MAX'
		op = ops.max_pool2d(kernel, strides, paddings, **kw_args)
	elif layer_op.pooling_param.pool == 1: #'AVE'
		op = ops.avg_pool2d(kernel, strides, paddings, **kw_args)
	elif layer_op.pooling_param.pool == 2: #'STOCHASTIC'
		op = ops.stochastic_pool2d(kernel, strides, paddings, **kw_args)
	else:
		raise ValueError(PYDAAL_CAFFE_UNKNOWN_POOLING_TYPE % layer_op.pool)
	
	return model._add(op.with_id(layer_op.name), last_op), op

def spp_op(model, last_op, layer_op):
	pyramid_height = layer_op.spp_param.pyramid_height

	if layer_op.pooling_param.pool == 0: #'MAX'
		op = MaxPyramidPooling2D().with_pyramid_height(pyramid_height)
	elif layer_op.pooling_param.pool == 1: #'AVE'
		op = AvgPyramidPooling2D().with_pyramid_height(pyramid_height)
	elif layer_op.pooling_param.pool == 2: #'STOCHASTIC'
		op = StochasticPyramidPooling2D().with_pyramid_height(pyramid_height)
	else:
		raise ValueError(PYDAAL_CAFFE_UNKNOWN_POOLING_TYPE % layer_op.pool)
	
	return model._add(op.with_id(layer_op.name), last_op), op

def eltwise_op(model, last_op, layer_op):
	if layer_op.eltwise_param.operation != 1: # non-sum layer is not supported yet 
		raise ValueError(PYDAAL_CAFFE_NON_SUM_ELTWISE_NOT_SUPPORTED)

	op = ops.elementwise_sum([c for c in layer_op.eltwise_param.coeff])
	return model._add(op.with_id(layer_op.name), last_op), op

def lrn_op(model, last_op, layer_op):
	depth = layer_op.lrn_param.local_size
	op = ops.lrn(depth=depth, 
				 bias=layer_op.lrn_param.k,
				 beta=layer_op.lrn_param.beta,
				 alpha=layer_op.lrn_param.alpha/depth)
	return model._add(op.with_id(layer_op.name), last_op), op

def split_op(model, last_op, layer_op):
	op = Identity().with_id(layer_op.name)
	return model._add(op, last_op), op

def relu_op(model, last_op, layer_op):
	if layer_op.relu_param.negative_slope != 0:
		raise ValueError(PYDAAL_CAFFE_LEAKY_RELU_NOT_IMPLEMENTED)

	op = Relu().with_id(layer_op.name)
	return model._add(op, last_op), op

def prelu_op(model, last_op, layer_op):
	op = ParametricRelu().with_id(layer_op.name)
	return model._add(op, last_op), op

def elu_op(model, last_op, layer_op):
	op = Elu().with_id(layer_op.name)
	return model._add(op, last_op), op

def softmax_op(model, last_op, layer_op):
	op = Softmax().with_dimension(layer_op.softmax_param.axis)
	return model._add(op.with_id(layer_op.name), last_op), op

def bnll_op(model, last_op, layer_op):
	op = SmoothRelu().with_id(layer_op.name)
	return model._add(op, last_op), op

def sigmoid_op(model, last_op, layer_op):
	op = Sigmoid().with_id(layer_op.name)
	return model._add(op, last_op), op

def abs_op(model, last_op, layer_op):
	op = Abs().with_id(layer_op.name)
	return model._add(op, last_op), op

def tanh_op(model, last_op, layer_op):
	op = Tanh().with_id(layer_op.name)
	return model._add(op, last_op), op

def dropout_op(model, last_op, layer_op):
	op = ops.dropout(1. - layer_op.dropout_param.dropout_ratio)
	return model._add(op.with_id(layer_op.name), last_op), op

def concat_op(model, last_op, layer_op):
	op = Concat().with_concat_dimension(layer_op.concat_param.axis)
	return model._add(op.with_id(layer_op.name), last_op), op

def reshape_op(model, last_op, layer_op):
	output_shape = list(layer_op.reshape_param.shape.dim)
	op = Reshape().with_new_dimensions(output_shape)

	return model._add(op.with_id(layer_op.name), last_op), op

def flatten_op(model, last_op, layer_op):
	op = Reshape().with_new_dimensions([0, -1])
	return model._add(op.with_id(layer_op.name), last_op), op

def softmax_cross_entropy_op(model, last_op, layer_op):
	op = SoftmaxCrossEntropy().with_class_dimension(layer_op.softmax_param.axis)
	return model._add(op.with_id(layer_op.name), last_op), op

def sigmoid_cross_entropy_op(model, last_op, layer_op):
	op = SigmoidCrossEntropy().with_id(layer_op.name)
	return model._add(op, last_op), op

##################################################################################
############## REGISTER CAFFE TO INTEL DAAL CONVERTIBLE OPS/LAYERS ###############
##################################################################################
register_op('InnerProduct', fc_op)
register_op('Convolution', conv2d_op)
register_op('Deconvolution', transposed_conv2d_op)
register_op('SoftmaxWithLoss', softmax_cross_entropy_op)
register_op('SigmoidCrossEntropyLoss', sigmoid_cross_entropy_op)
register_op('Dropout', dropout_op)
register_op('BatchNorm', batch_norm_op)
register_op('Concat', concat_op)
register_op('Reshape', reshape_op)
register_op('Flatten', flatten_op)
register_op('SPP', spp_op)
register_op('Pooling', pool_op)
register_op('Scale', scale_op)
register_op('Eltwise', eltwise_op)
register_op('Softmax', softmax_op)
register_op('Sigmoid', sigmoid_op)
register_op('Split', split_op)
register_op('PReLU', prelu_op)
register_op('ReLU', relu_op)
register_op('BNLL', bnll_op)
register_op('TanH', tanh_op)
register_op('ELU', elu_op)
register_op('Abs', abs_op)
register_op('LRN', lrn_op)
# Re-registering some of caffe.V1LayerParameter.LayerType layers
register_op(14, fc_op)
register_op(4, conv2d_op)
register_op(39, transposed_conv2d_op)
register_op(21, softmax_cross_entropy_op)
register_op(27, sigmoid_cross_entropy_op)
register_op(6, dropout_op)
register_op(3, concat_op)
register_op(8, flatten_op)
register_op(17, pool_op)
register_op(25, eltwise_op)
register_op(20, softmax_op)
register_op(19, sigmoid_op)
register_op(22, split_op)
register_op(18, relu_op)
register_op(2, bnll_op)
register_op(23, tanh_op)
register_op(35, abs_op)
register_op(15, lrn_op)