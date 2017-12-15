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

from ..model.nn import *
from ..helpers import to_tensor, initialize
from ..constants import PYDAAL_ELU_NOT_IMPLEMENTED
from daal.algorithms.neural_networks import prediction, training
from daal.algorithms.neural_networks.initializers import (
	uniform, xavier, gaussian, truncated_gaussian
)
from daal.algorithms.neural_networks.layers.loss import (
	softmax_cross, logistic_cross
)
from daal.algorithms.neural_networks.layers import ( 
    forward, pooling2d, pooling1d, pooling3d, fullyconnected, locallyconnected2d,
    maximum_pooling2d, average_pooling2d, stochastic_pooling2d, transposed_conv2d,
    spatial_average_pooling2d, spatial_maximum_pooling2d, spatial_stochastic_pooling2d,
    maximum_pooling1d, average_pooling1d, maximum_pooling3d, average_pooling3d, 
    relu, prelu, dropout, convolution2d, softmax, tanh, logistic, smoothrelu, 
    split, concat, abs, reshape, lrn, lcn, batch_normalization, eltwise_sum
)
from multipledispatch import dispatch
import sys

try: # experimental feature which is not included into distibution yet
	from daal.algorithms.neural_networks.layers import elu
except ImportError:
	sys.stderr.write(PYDAAL_ELU_NOT_IMPLEMENTED + '\n')

build_namespace = dict()

@dispatch(Model, bool, namespace=build_namespace)
def build_topology(model, trainable, initializer=None):
	# Embedd Split nodes to wire up multpile inputs/outputs
	build_namespace['initializer'] = initializer; build(model)
	topology = training.Topology() if trainable else prediction.Topology()
	# TODO: refactor when breadth-first adding to the topology is fixed in Intel DAAL 2018  
	nodes = {node : build(node, trainable) for node in model.nodes.values()}	
	wires = {node : topology.add(nodes[node]) for node in sorted(nodes)}

	# Create a wired topology
	for target in sorted(wires):
		target.daal_id = wires[target]

		for source in target.inputs:
			topology.get(wires[source]).addNext(wires[target])
			
	return topology

@dispatch(Model, namespace=build_namespace)
def build(model):
	for node in list(model.nodes.values()):
		outputs = node.outputs
		nsplits = len(outputs)

		if nsplits > 1:
			split = Split().with_nsplits(nsplits)
			split = node.clear_outputs()(split)
			for output in outputs:
				model._add(split, output)

@dispatch(Node, bool, object, namespace=build_namespace)
def build(node, trainable, layer):
	return layer if trainable else layer.forwardLayer.clone()

@dispatch(Intializable, object, namespace=build_namespace)
def add_initializer(initializable, layer):
	if build_namespace['initializer'] is not None:
		layer.parameter.weightsInitializer = build(build_namespace['initializer'])
		layer.parameter.biasesInitializer = build(build_namespace['initializer'])
	else:		
		if initializable.weights().has_initializer():
			layer.parameter.weightsInitializer = build(initializable.weights().initializer)

		if initializable.biases().has_initializer():
			layer.parameter.biasesInitializer = build(initializable.biases().initializer)

	return layer

@dispatch(Conv2D, object, namespace=build_namespace)
def add_group(node, layer):
	if node.get_group():
		layer.parameter.nGroups = node.get_group()

	return layer

@dispatch(Uniform, namespace=build_namespace)
def build(node):
	return uniform.Batch(node.left_bound, node.right_bound)

@dispatch(Gaussian, namespace=build_namespace)
def build(node):
	return gaussian.Batch(node.mean, node.std)

@dispatch(TruncatedGaussian, namespace=build_namespace)
def build(node):
	_initializer = truncated_gaussian.Batch(node.mean, node.std)
	if node.has_bounds():
		_initializer.parameter.a = node.left_bound
		_initializer.parameter.b = node.right_bound

	return _initializer

@dispatch(Xavier, namespace=build_namespace)
def build(node):
	return xavier.Batch()

@dispatch(Conv2D, bool, namespace=build_namespace)
def build(node, trainable):
	conv_layer = convolution2d.Batch()
	conv_layer.parameter.kernelSizes = convolution2d.KernelSizes(node.kernel_height(), node.kernel_width())
	conv_layer.parameter.paddings = convolution2d.Paddings(node.padding_height(), node.padding_width())
	conv_layer.parameter.strides = convolution2d.Strides(node.stride_height(), node.stride_width())
	# conv_layer.parameter.indices = convolution2d.Indices(node.input_height(), node.input_width())
	# conv_layer.parameter.groupDimension = node.input_channels()
	conv_layer.parameter.nKernels = node.kernel_output()
	conv_layer = add_initializer(node, conv_layer)
	conv_layer = add_group(node, conv_layer)

	return build(node, trainable, conv_layer)

@dispatch(LocallyConnected2D, bool, namespace=build_namespace)
def build(node, trainable):
	lc_layer = locallyconnected2d.Batch()
	lc_layer.parameter.kernelSizes = locallyconnected2d.KernelSizes(node.kernel_height(), node.kernel_width())
	lc_layer.parameter.paddings = locallyconnected2d.Paddings(node.padding_height(), node.padding_width())
	lc_layer.parameter.strides = locallyconnected2d.Strides(node.stride_height(), node.stride_width())
	# lc_layer.parameter.indices = locallyconnected2d.Indices(node.input_height(), node.input_width())
	# lc_layer.parameter.groupDimension = node.input_channels()
	lc_layer.parameter.nKernels = node.kernel_output()
	lc_layer = add_initializer(node, lc_layer)
	lc_layer = add_group(node, lc_layer)

	return build(node, trainable, lc_layer)

@dispatch(TransposedConv2D, bool, namespace=build_namespace)
def build(node, trainable):
	transposed_conv_layer = transposed_conv2d.Batch()
	transposed_conv_layer.parameter.kernelSizes = transposed_conv2d.KernelSizes(node.kernel_height(), node.kernel_width())
	transposed_conv_layer.parameter.valueSizes = transposed_conv2d.ValueSizes(node.output_height(), node.output_width())
	transposed_conv_layer.parameter.paddings = transposed_conv2d.Paddings(node.padding_height(), node.padding_width())
	transposed_conv_layer.parameter.strides = transposed_conv2d.Strides(node.stride_height(), node.stride_width())
	# transposed_conv_layer.parameter.indices = transposed_conv2d.Indices(node.input_height(), node.input_width())
	# transposed_conv_layer.parameter.groupDimension = node.input_channels()
	transposed_conv_layer.parameter.nKernels = node.kernel_output()
	transposed_conv_layer = add_initializer(node, transposed_conv_layer)
	transposed_conv_layer = add_group(node, transposed_conv_layer)

	return build(node, trainable, transposed_conv_layer)

@dispatch(MaxPooling2D, bool, namespace=build_namespace)
def build(node, trainable):
	return build_pooling(node, maximum_pooling2d.Batch(node.num_input_dimensions()), trainable) 

@dispatch(AvgPooling2D, bool, namespace=build_namespace)
def build(node, trainable):
	return build_pooling(node, average_pooling2d.Batch(node.num_input_dimensions()), trainable)

@dispatch(StochasticPooling2D, bool, namespace=build_namespace)
def build(node, trainable):
	return build_pooling(node, stochastic_pooling2d.Batch(node.num_input_dimensions()), trainable) 

@dispatch(MaxPyramidPooling2D, bool, namespace=build_namespace)
def build(node, trainable):
	return build(node, trainable, spatial_maximum_pooling2d.Batch(node.get_pyramid_height(), node.num_input_dimensions())) 

@dispatch(AvgPyramidPooling2D, bool, namespace=build_namespace)
def build(node, trainable):
	return build(node, trainable, spatial_average_pooling2d.Batch(node.get_pyramid_height(), node.num_input_dimensions()))

@dispatch(StochasticPyramidPooling2D, bool, namespace=build_namespace)
def build(node, trainable):
	return build(node, trainable, spatial_stochastic_pooling2d.Batch(node.get_pyramid_height(), node.num_input_dimensions())) 

@dispatch(Pooling2D, object, bool, namespace=build_namespace)
def build_pooling(node, pool_layer, trainable):
	pool_layer.parameter.kernelSizes = pooling2d.KernelSizes(node.kernel_height(), node.kernel_width())
	pool_layer.parameter.paddings = pooling2d.Paddings(node.padding_height(), node.padding_width())
	pool_layer.parameter.strides = pooling2d.Strides(node.stride_height(), node.stride_width())
	# pool_layer.parameter.indices = pooling2d.Indices(node.input_height(), node.input_width())

	return build(node, trainable, pool_layer)

@dispatch(MaxPooling3D, bool, namespace=build_namespace)
def build(node, trainable):
	return build_pooling(node, maximum_pooling3d.Batch(node.num_input_dimensions()), trainable) 

@dispatch(AvgPooling3D, bool, namespace=build_namespace)
def build(node, trainable):
	return build_pooling(node, average_pooling3d.Batch(node.num_input_dimensions()), trainable) 

@dispatch(Pooling3D, object, bool, namespace=build_namespace)
def build_pooling(node, pool_layer, trainable):
	pool_layer.parameter.kernelSizes = pooling3d.KernelSizes(node.kernel_depth(), node.kernel_height(), node.kernel_width())
	pool_layer.parameter.paddings = pooling3d.Paddings(node.padding_depth(), node.padding_height(), node.padding_width())
	pool_layer.parameter.strides = pooling3d.Strides(node.stride_depth(), node.stride_height(), node.stride_width())
	# pool_layer.parameter.indices = pooling2d.Indices(node.input_depth(), node.input_height(), node.input_width())

	return build(node, trainable, pool_layer)

@dispatch(MaxPooling1D, bool, namespace=build_namespace)
def build(node, trainable):
	return build_pooling(node, maximum_pooling1d.Batch(node.num_input_dimensions()), trainable) 

@dispatch(AvgPooling1D, bool, namespace=build_namespace)
def build(node, trainable):
	return build_pooling(node, average_pooling1d.Batch(node.num_input_dimensions()), trainable) 

@dispatch(Pooling1D, object, bool, namespace=build_namespace)
def build_pooling(node, pool_layer, trainable):
	pool_layer.parameter.kernelSize = pooling1d.KernelSize(node.kernel())
	pool_layer.parameter.padding = pooling1d.Padding(node.padding())
	pool_layer.parameter.stride = pooling1d.Stride(node.stride())
	# pool_layer.parameter.index = pooling1d.Index(node.width())

	return build(node, trainable, pool_layer)

@dispatch(FullyConnected, bool, namespace=build_namespace)
def build(node, trainable):
	fc_layer = fullyconnected.Batch(node.num_biases())
	fc_layer = add_initializer(node, fc_layer)

	return build(node, trainable, fc_layer)
	
@dispatch(SoftmaxCrossEntropy, bool, namespace=build_namespace)
def build(node, trainable):
	if not trainable:
		return build(node.get_softmax_part(), trainable)

	sce_layer = softmax_cross.Batch()
	if node.get_class_dimension():
		sce_layer.parameter.dimension = node.get_class_dimension() 

	return build(node, trainable, sce_layer)

@dispatch(SigmoidCrossEntropy, bool, namespace=build_namespace)
def build(node, trainable):
	if not trainable:
		raise build(Sigmoid(), trainable)

	return logistic_cross.Batch()

@dispatch(Relu, bool, namespace=build_namespace)
def build(node, trainable):
	return relu.Batch() if trainable else relu.forward.Batch()

@dispatch(Elu, bool, namespace=build_namespace)
def build(node, trainable):
	return elu.Batch() if trainable else elu.forward.Batch()

@dispatch(SmoothRelu, bool, namespace=build_namespace)
def build(node, trainable):
	return smoothrelu.Batch() if trainable else smoothrelu.forward.Batch()

@dispatch(ParametricRelu, bool, namespace=build_namespace)
def build(node, trainable):
	return prelu.Batch() if trainable else prelu.forward.Batch()

@dispatch(Softmax, bool, namespace=build_namespace)
def build(node, trainable):
	softmax_layer = softmax.Batch()
	if node.get_dimension() and node.get_dimension() != -1:
		softmax_layer.parameter.dimension = node.get_dimension()

	return build(node, trainable, softmax_layer)

@dispatch(Sigmoid, bool, namespace=build_namespace)
def build(node, trainable):
	return logistic.Batch() if trainable else logistic.forward.Batch()

@dispatch(Tanh, bool, namespace=build_namespace)
def build(node, trainable):
	return tanh.Batch() if trainable else tanh.forward.Batch()

@dispatch(Abs, bool, namespace=build_namespace)
def build(node, trainable):
	return abs.Batch() if trainable else abs.forward.Batch()

@dispatch(Reshape, bool, namespace=build_namespace)
def build(node, trainable):
	return build(node, trainable, reshape.Batch(node.get_new_dimensions()))

@dispatch(Concat, bool, namespace=build_namespace)
def build(node, trainable):
	#TODO: interim workaround to fix segfault with clone() on forwardLayer
	return build(node, trainable, concat.Batch(node.get_concat_dimension()))

@dispatch(Split, bool, namespace=build_namespace)
def build(node, trainable):
	return build(node, trainable, split.Batch(node.get_nsplits(), node.get_nsplits()))

@dispatch(ElementwiseSum, bool, namespace=build_namespace)
def build(node, trainable):
	eltwise_layer = eltwise_sum.Batch()
	if node.get_coefficients() is not None:
		input_id = eltwise_sum.forward.coefficients
		inputs = eltwise_layer.forwardLayer.getLayerInput()
		initialize(inputs, node.get_coefficients(), input_id)

	return build(node, trainable, eltwise_layer)

@dispatch(Dropout, bool, namespace=build_namespace)
def build(node, trainable):
	dropout_layer = dropout.Batch()	
	if node.get_probability():
		dropout_layer.parameter.retainRatio = node.get_probability() 
	
	return build(node, trainable, dropout_layer)

@dispatch(LocalResponseNormalization, bool, namespace=build_namespace)
def build(node, trainable):
	lrn_layer = lrn.Batch()
	if node.get_depth():
		lrn_layer.parameter.nAdjust = node.get_depth()
	if node.get_bias():
		lrn_layer.parameter.kappa = node.get_bias()
	if node.get_alpha():
		lrn_layer.parameter.alpha = node.get_alpha()
	if node.get_beta():
		lrn_layer.parameter.beta = node.get_beta()

	return build(node, trainable, lrn_layer)

@dispatch(LocalContrastNormalization, bool, namespace=build_namespace)
def build(node, trainable):
	lcn_layer = lcn.Batch() 
	if node.kernel_data() is not None:
		kernel = to_tensor(node.kernel_data())
		lcn_layer.parameter.kernel = kernel

	return build(node, trainable, lcn_layer)

@dispatch(BatchNormalization, bool, namespace=build_namespace)
def build(node, trainable):
	bn_layer = batch_normalization.Batch()	
	if node.get_epsilon():
		bn_layer.parameter.epsilon = node.get_epsilon()
	if node.get_alpha():
		bn_layer.parameter.alpha = node.get_alpha()

	return build(node, trainable, bn_layer)