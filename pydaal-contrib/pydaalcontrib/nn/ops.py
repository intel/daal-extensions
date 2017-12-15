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
PyDAAL contrib's submodule alleviating ops creation for Neural Networks
-----------------------------------------------------------------------

Provides
	1. Wrappers for creating NN ops, *e.g.* :py:func:`conv2d` (2D Convolution node) *etc.*

Note
	**HWIO** tensor format is deciphered as :math:`height \\times width \\times input \\times output` dimensions.
"""

from ..model.nn import *
from ..constants import *
from multipledispatch import dispatch
import numpy as np

def _check_strides(strides):
	if len(strides) == 0 and np.sum(strides) > 0:
		raise ValueError(PYDAAL_TF_EMPTY_STRIDES)

def _check_paddings(paddings):
	if len(paddings) == 0:
		raise ValueError(PYDAAL_TF_EMPTY_PADDINGS)

######################################################################
################ INTEL DAAL EXT OPS ARE DEFINED BELOW ################
######################################################################
def fc(num_units, use_bias=True):
	"""Creates Fully Connected node with the provided number of hidden units.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/hl2aed2>`__.

	Args:
		num_units (:obj:`int`): Number of hidden units.
		use_bias (:obj:`bool`): Flag for appling bias or not.

	Returns:
		:py:class:`pydaalcontrib.model.nn.FullyConnected`: Fully Connected node.
	"""
	op = FullyConnected()
	op = op.with_weights(NodeDescriptor(PYDAAL_WEIGHTS_KERNEL, [num_units, None]))

	if use_bias:
		op = op.with_biases(BiasNodeDescriptor(PYDAAL_BIASES_KERNEL, num_units))
	else:
		# overcoming Intel DAAL initialization issues where biases should be compalsory set to zeros 
		op = op.with_biases(EmptyNodeDescriptor(PYDAAL_BIASES_KERNEL, num_units))
		op = op.with_biases_initializer(Uniform(0, 0))

	return op

def conv2d(kernel, strides=[1, 1], paddings=[0, 0], 
	       input_format='NCHW', kernel_format='HWO', 
	       strides_format='HW', paddings_format='HW',
	       use_bias=True):
	"""Creates 2D Convolution node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/z4b54ra>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides (symmetric).
		paddings (:obj:`list`, optional): Sizes of paddings (symmetric).
		kernel_format (:obj:`str`, optional): Kernel format, *e.g.* **HWO**.
		strides_format (:obj:`str`, optional): Strides format, *e.g.* **HW**.
		paddings_format (:obj:`str`, optional): Paddings format, *e.g.* **HW**.
		use_bias (:obj:`bool`): Flag for appling bias or not.

	Returns:
		:py:class:`pydaalcontrib.model.nn.Conv2D`: 2D Convolution node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)
	
	return _fill_conv_op(Conv2D(), kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format, use_bias=use_bias)

def lc2d(kernel, strides=[1, 1], paddings=[0, 0], 
	     input_format='NCHW', kernel_format='HWO', 
	     strides_format='HW', paddings_format='HW',
	     use_bias=True):
	"""Creates 2D Locally Connected node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/lhapxou>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides (symmetric).
		paddings (:obj:`list`, optional): Sizes of paddings (symmetric).
		kernel_format (:obj:`str`, optional): Kernel format, *e.g.* **HWO**.
		strides_format (:obj:`str`, optional): Strides format, *e.g.* **HW**.
		paddings_format (:obj:`str`, optional): Paddings format, *e.g.* **HW**.
		use_bias (:obj:`bool`): Flag for appling bias or not.

	Returns:
		:py:class:`pydaalcontrib.model.nn.LocallyConnected2D`: 2D Locally Connected node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)
	
	return _fill_conv_op(LocallyConnected2D(), kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format, use_bias)

def transposed_conv2d(kernel, output_shape, strides=[1, 1], paddings=[0, 0], 
					  input_format='NCHW', kernel_format='HWO', output_format='HW', 
					  strides_format='HW', paddings_format='HW', use_bias=True):
	"""Creates Transposed 2D Convolution node with the provided kernel, output shape, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/zzzmeh4>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		output_shape (:obj:`list`): Sizes of an output tensor (only spatial dimensions).
		strides (:obj:`list`, optional): Sizes of strides (symmetric).
		paddings (:obj:`list`, optional): Sizes of paddings (symmetric).
		kernel_format (:obj:`str`, optional): Kernel format, *e.g.* **HWO**.
		output_format (:obj:`str`, optional): Output shape format, *e.g.* **HW**.
		strides_format (:obj:`str`, optional): Strides format, *e.g.* **HW**.
		paddings_format (:obj:`str`, optional): Paddings format, *e.g.* **HW**.
		use_bias (:obj:`bool`): Flag for appling bias or not.

	Returns:
		:py:class:`pydaalcontrib.model.nn.TransposedConv2D`: Transposed 2D Convolution node.
	"""

	_check_strides(strides)
	_check_paddings(paddings)
	
	op = TransposedConv2D()
	op = op.with_output(Output(output_shape), output_format)
	
	return _fill_conv_op(op, kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format, use_bias)

def max_pool2d(kernel, strides=[2, 2], paddings=[0, 0], 
			   input_format='NCHW', kernel_format='HW', 
			   strides_format='HW', paddings_format='HW',
			   pyramid_height=None):
	"""Creates 2D Maximum Pooling node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/jf8z7sg>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides (symmetric).
		paddings (:obj:`list`, optional): Sizes of paddings (symmetric).
		kernel_format (:obj:`str`, optional): Kernel format, *e.g.* **HWO**.
		strides_format (:obj:`str`, optional): Strides format, *e.g.* **HW**.
		paddings_format (:obj:`str`, optional): Paddings format, *e.g.* **HW**.

	Returns:
		:py:class:`pydaalcontrib.model.nn.MaxPooling2D`: 2D Maximum Pooling node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)

	return _fill_pool_op(MaxPooling2D(), PYDAAL_MAX_POOL_KERNEL, kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format)

def max_pool1d(kernel, strides=[2], paddings=[0], 
			   input_format='NCW', kernel_format='W', 
			   strides_format='W', paddings_format='W'):
	"""Creates 1D Maximum Pooling node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/jekrcc7>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides.
		paddings (:obj:`list`, optional): Sizes of paddings.
		kernel_format (:obj:`str`, optional): Kernel format.
		strides_format (:obj:`str`, optional): Strides format.
		paddings_format (:obj:`str`, optional): Paddings format.
		group_dimension (:obj:`int`, optional): The last pooling dimension.
		name (:obj:`str`, optional): node's name.

	Returns:
		:py:class:`pydaalcontrib.model.nn.MaxPooling1D`: 1D Maximum Pooling node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)

	return _fill_pool_op(MaxPooling1D(), PYDAAL_MAX_POOL_KERNEL, kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format)

def max_pool3d(kernel, strides=[2, 2, 2], paddings=[0, 0, 0], 
			   input_format='NCDHW', kernel_format='DHW', 
			   strides_format='DHW', paddings_format='DHW'):
	"""Creates 3D Maximum Pooling node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/kxarx42>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides.
		paddings (:obj:`list`, optional): Sizes of paddings.
		kernel_format (:obj:`str`, optional): Kernel format.
		strides_format (:obj:`str`, optional): Strides format.
		paddings_format (:obj:`str`, optional): Paddings format.
		group_dimension (:obj:`int`, optional): The last pooling dimension.
		name (:obj:`str`, optional): node's name.

	Returns:
		:py:class:`pydaalcontrib.model.nn.MaxPooling3D`: 3D Maximum Pooling node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)

	return _fill_pool_op(MaxPooling3D(), PYDAAL_MAX_POOL_KERNEL, kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format)

def avg_pool2d(kernel, strides=[2, 2], paddings=[0, 0], 
			   input_format='NCHW', kernel_format='HW', 
			   strides_format='HW', paddings_format='HW',
			   pyramid_height=None):
	"""Creates 2D Average Pooling node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/zskely9>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides (symmetric).
		paddings (:obj:`list`, optional): Sizes of paddings (symmetric).
		kernel_format (:obj:`str`, optional): Kernel format, *e.g.* **HWO**.
		strides_format (:obj:`str`, optional): Strides format, *e.g.* **HW**.
		paddings_format (:obj:`str`, optional): Paddings format, *e.g.* **HW**.

	Returns:
		:py:class:`pydaalcontrib.model.nn.AvgPooling2D`: 2D Average Pooling node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)

	return _fill_pool_op(AvgPooling2D(), PYDAAL_AVG_POOL_KERNEL, kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format)


def avg_pool1d(kernel, strides=[2], paddings=[0], 
			   input_format='NCW', kernel_format='W', 
			   strides_format='W', paddings_format='W'):
	"""Creates 1D Average Pooling node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/js4xj9w>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides.
		paddings (:obj:`list`, optional): Sizes of paddings.
		kernel_format (:obj:`str`, optional): Kernel format.
		strides_format (:obj:`str`, optional): Strides format.
		paddings_format (:obj:`str`, optional): Paddings format.

	Returns:
		:py:class:`pydaalcontrib.model.nn.AvgPooling1D`: 1D Average Pooling node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)

	return _fill_pool_op(AvgPooling1D(), PYDAAL_AVG_POOL_KERNEL, kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format)

def avg_pool3d(kernel, strides=[2, 2, 2], paddings=[0, 0, 0], 
			   input_format='NCDHW', kernel_format='DHW', 
			   strides_format='DHW', paddings_format='DHW'):
	"""Creates 3D Average Pooling node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/kxjlpuc>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides.
		paddings (:obj:`list`, optional): Sizes of paddings.
		kernel_format (:obj:`str`, optional): Kernel format.
		strides_format (:obj:`str`, optional): Strides format.
		paddings_format (:obj:`str`, optional): Paddings format.
		name (:obj:`str`, optional): node's name.

	Returns:
		:py:class:`pydaalcontrib.model.nn.AvgPooling3D`: 3D Average Pooling node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)

	return _fill_pool_op(AvgPooling3D(), PYDAAL_MAX_POOL_KERNEL, kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format)

def stochastic_pool2d(kernel, strides=[2, 2], paddings=[0, 0], 
					  input_format='NCHW', kernel_format='HW', 
					  strides_format='HW', paddings_format='HW'):
	"""Creates 2D Stochastic Pooling node with the provided kernel, strides and paddings.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/lb6wgmj>`__.

	Args:
		kernel (:obj:`list`): Sizes of a kernel (filter).
		strides (:obj:`list`, optional): Sizes of strides (symmetric).
		paddings (:obj:`list`, optional): Sizes of paddings (symmetric).
		kernel_format (:obj:`str`, optional): Kernel format, *e.g.* **HWO**.
		strides_format (:obj:`str`, optional): Strides format, *e.g.* **HW**.
		paddings_format (:obj:`str`, optional): Paddings format, *e.g.* **HW**.

	Returns:
		:py:class:`pydaalcontrib.model.nn.StochasticPooling2D`: 2D Stochastic Pooling node.
	"""
	_check_strides(strides)
	_check_paddings(paddings)

	return _fill_pool_op(StochasticPooling2D(), PYDAAL_STOCH_POOL_KERNEL, kernel, strides, paddings, 
						 input_format, kernel_format, strides_format, paddings_format)


def reshape(new_dimensions):
	"""Creates Reshape node with the provided new dimensions.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/j3d7kc5>`__.

	Args:
		new_dimensions (:obj:`list`): Sizes of an output (reshaped) tensor.

	Returns:
		:py:class:`pydaalcontrib.model.nn.Reshape`: Reshape node.
	"""
	return Reshape().with_new_dimensions(new_dimensions)

def concatenate(nodes, axis=1):
	"""Creates Concat node with the provided axis/dimensions across which concatenation happens.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/l58acvo>`__.

	Args:
		nodes (list): List of nodes to be concatenated.
		axis (:obj:`int`): Axis/dimension across which concatenation happens.

	Returns:
		:py:class:`pydaalcontrib.model.nn.Concat`: Concat node.
	"""
	concat = Concat().with_concat_dimension(axis)
	
	for node in nodes:
		node(concat)

	return concat 

def softmax_cross_entropy(class_dimension=1):
	"""Creates Softmax Cross-Entropy loss node with the specified dimension holding labels (classes).

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/hlpjdmq>`__.

	Args:
		class_dimension (:obj:`int`, optional): Index of the 1-sized dimension holding labels (class info per sample).

	Returns:
		:py:class:`pydaalcontrib.model.nn.SoftmaxCrossEntropy`: Softmax Cross-Entropy loss node.
	"""
	return SoftmaxCrossEntropy().with_class_dimension(class_dimension)

def dropout(keep_prob=.5):
	"""Creates Dropout node with the provided retention probability.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/z6bnv4z>`__.

	Args:
		keep_prob (:obj:`float`): Retention (keep) probability.

	Returns:
		:py:class:`pydaalcontrib.model.nn.Dropout`: Dropout node.

	Raises:
		ValueError: If the retetntion probability is out of range, where :math:`p \\in [0 \\dots 1]`.
	"""
	if not 0 < keep_prob <= 1:
		raise ValueError(PYDAAL_TF_WRONG_KEEP_PROBABILITY)

	return Dropout().with_probability(keep_prob)

def lrn(depth=5, bias=2., alpha=1e-4, beta=.75):
	"""Creates Local-Response Normalization node with the provided parameters.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/h2utgtk>`__.

	Args:
		depth (:obj:`int`, optional): Full-width of the 1-D normalization window.
		bias (:obj:`float`, optional): A bias :math:`\\kappa` in the denominator (to avoid dividing by 0).
		alpha (:obj:`float`, optional):  A scale factor :math:`\\alpha` (typically positive).
		beta (:obj:`float`, optional):  An exponent :math:`\\beta`.

	Returns:
		:py:class:`pydaalcontrib.model.nn.LocalResponseNormalization`: Local-Response Normalization node.
	"""
	op = LocalResponseNormalization()
	op = op.with_depth(depth)
	op = op.with_alpha(alpha)
	op = op.with_beta(beta)
	op = op.with_bias(bias)

	return op

def lcn(kernel=None, kernel_format='HW', name=None):
	"""Creates Local-Contrast Normalization node with the provided kernel.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/jgfckja>`__.

	Args:
		kernel (:obj:`list`, optional): Sizes of a kernel (filter).
		kernel_format (:obj:`str`, optional): Kernel format, *e.g.* **HW**.
		name (:obj:`str`, optional): node's name.

	Returns:
		:py:class:`pydaalcontrib.model.nn.LocalContrastNormalization`: Local-Contrast Normalization node.
	"""
	op = LocalContrastNormalization()

	if kernel is not None:
		# normalize to sum 1
		kernel /= kernel.sum()
		name = _next_node_name(name)
		node = NodeDescriptor(':'.join([name, PYDAAL_LCN_KERNEL]), kernel)
		op.with_kernel(node, kernel_format)

	return op

def elementwise_sum(coefficients=None):
	"""Creates Elementwise Sum node with the provided coefficient.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/y79xelza>`__.

	Args:
		coefficients (:obj:`list` or :obj:`numpy.ndarray`, optional): Coefficients per input.

	Returns:
		:py:class:`pydaalcontrib.model.nn.ElementwiseSum`: Elementwise Sum node.
	"""
	op = ElementwiseSum()

	if coefficients is not None and type(coefficients) is not np.ndarray:
		op = op.with_coefficients(np.array(coefficients))
	elif coefficients is not None:
		op = op.with_coefficients(coefficients)

	return op

##########################################################################
############### INTERNAL OPS/FUNCTIONS ARE DEFINED BELOW #################
##########################################################################

def _initializer(initializer):
	if initializer.type.startswith('gaussian'):
		return Gaussian(initializer.mean, initializer.std)
	elif initializer.type.startswith('constant'):
		return Uniform(initializer.value, initializer.value)
	elif initializer.type.startswith('uniform'):
		return Uniform(initializer.min, initializer.max)
	elif initializer.type.startswith('xavier'):
		return Xavier()
	else:
		raise ValueError(PYDAAL_NOT_A_INITIALIZER % initializer_name)

def _fill_pool_op(op, kernel_name, kernel, strides, paddings, 
				  input_format, kernel_format, strides_format, paddings_format):
	op = op.with_kernel(NodeDescriptor(kernel_name, kernel), kernel_format)
	op = op.with_paddings(Paddings(paddings), paddings_format)
	op = op.with_strides(Strides(strides), strides_format)
	op = op.with_input_format(input_format)

	return op

def _fill_conv_op(op, kernel, strides, paddings, 
				  input_format, kernel_format, 
				  strides_format, paddings_format,
				  use_bias=True):
	op = op.with_kernel(NodeDescriptor(PYDAAL_WEIGHTS_KERNEL, kernel), kernel_format)
	op = op.with_paddings(Paddings(paddings), paddings_format)
	op = op.with_strides(Strides(strides), strides_format)
	op = op.with_input_format(input_format)

	if use_bias:
		op = op.with_biases(BiasNodeDescriptor(PYDAAL_BIASES_KERNEL, kernel[kernel_format.index('O')]))
	else:
		# overcoming Intel DAAL initialization issues where biases should be compalsory set to zeros 
		op = op.with_biases(EmptyNodeDescriptor(PYDAAL_BIASES_KERNEL, kernel[kernel_format.index('O')]))
		op = op.with_biases_initializer(Uniform(0, 0))

	return op

def _batch_norm(scale, offset, **kw_args):
	op = BatchNormalization()
	op = op.with_kernel(NodeDescriptor(kw_args['scale_name'], scale))
	op = op.with_biases(BiasNodeDescriptor(kw_args['offset_name'], offset))
	
	return op

def _fc(weights, biases, **kw_args):
	op = FullyConnected()
	op = op.with_weights(NodeDescriptor(kw_args['weights_name'], weights))
	op = op.with_biases(BiasNodeDescriptor(kw_args['biases_name'], biases))

	return op

def _conv2d(kernel, biases, strides, paddings, **kw_args):
	_check_strides(strides)
	_check_paddings(paddings)
	
	op = Conv2D()
	op = op.with_kernel(NodeDescriptor(kw_args['kernel_name'], kernel), kw_args['kernel_format'])
	op = op.with_biases(BiasNodeDescriptor(kw_args['biases_name'], biases))
	op = op.with_paddings(Paddings(paddings), kw_args['paddings_format'])
	op = op.with_strides(Strides(strides), kw_args['strides_format'])
	op = op.with_input_format(kw_args['input_format'])
	
	return op

def _transposed_conv2d(kernel, biases, strides, paddings, output_shape, **kw_args):
	_check_strides(strides)
	_check_paddings(paddings)
	
	op = TransposedConv2D()
	op = op.with_kernel(NodeDescriptor(kw_args['kernel_name'], kernel), kw_args['kernel_format'])
	op = op.with_biases(BiasNodeDescriptor(kw_args['biases_name'], biases))
	op = op.with_output(Output(output_shape), kw_args['output_format'])
	op = op.with_paddings(Paddings(paddings), kw_args['paddings_format'])
	op = op.with_strides(Strides(strides), kw_args['strides_format'])
	op = op.with_input_format(kw_args['input_format'])
	
	return op