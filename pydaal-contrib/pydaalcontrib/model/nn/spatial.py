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

import numpy as np
from multipledispatch import dispatch
from .misc import Scale
from .base import *

try:
	basestring = basestring
except NameError:
	# 'basestring' is undefined, must be Python 3
	basestring = (str, bytes)

class Output(NodeDescriptor):
	"""Descriptor of the output in :py:class:`TransposedConv2D`."""
	def __init__(self, shape):
		self.name = "output"
		self.shape = list(shape)

class Strides(NodeDescriptor):
	"""Descriptor of srides in :py:class:`ComputeSpatial`."""
	def __init__(self, shape):
		self.name = "strides"
		self.shape = list(shape)

class Paddings(NodeDescriptor):
	"""Descriptor of paddings in :py:class:`ComputeSpatial`."""
	def __init__(self, shape):
		self.name = "paddings"
		self.shape = list(shape)

class ComputeSpatial(ConnectedNode):
	"""Base class for all spatial-compute specific nodes with the kernel, strides and paddings."""
	def with_input_format(self, input_format):
		self.variables['input_format'] = input_format
		return self

	def input_format(self):
		return self.variables['input_format']

	def num_input_dimensions(self):
		return len(self.variables['input_format'])

	@dispatch(NodeDescriptor, basestring)
	def with_strides(self, strides, strides_format):
		"""Sets the ``strides`` descriptor."""
		self.variables['strides'] = strides
		self.variables['strides_format'] = strides_format
		return self

	@dispatch(NodeDescriptor, basestring)
	def with_paddings(self, paddings, paddings_format):
		"""Sets the ``paddings`` descriptor."""
		self.variables['paddings'] = paddings
		self.variables['paddings_format'] = paddings_format
		return self

	@dispatch(NodeDescriptor, basestring)
	def with_kernel(self, kernel, kernel_format):
		"""Sets the ``kernel`` descriptor."""
		self.variables['kernel_format'] = kernel_format
		self.with_weights(kernel)
		return self

	def strides_shape_param(self, param):
		"""Returns the shape parameter of ``strides``."""
		index = self.variables['strides_format'].index(param)
		return self.variables['strides'].shape[index]

	def paddings_shape_param(self, param):
		"""Returns the shape parameter of ``paddings``."""
		index = self.variables['paddings_format'].index(param)
		return self.variables['paddings'].shape[index]

	def kernel(self):
		"""Returns ``kernel`` descriptor."""
		return self.weights()

	def kernel_shape(self):
		"""Returns the shape of the ``kernel``."""
		return self.weights_shape()

	def kernel_data(self):
		"""Returns the underlying data of the ``kernel``."""
		return self.weights_data()

	def kernel_shape_param(self, param):
		"""Returns the shape parameter of the ``kernel``."""
		index = self.variables['kernel_format'].index(param)
		return self.variables['weights'].shape[index]

class Compute1D(ComputeSpatial):
	"""Base class for all 1D spatial-compute specific nodes."""
	def width(self):
		return self.variables['input_format'].index('W')

	def stride(self):
		"""Returns the width dimension of ``strides``."""
		return self.strides_shape_param('W')

	def padding(self):
		"""Returns the width dimension of ``paddings``."""
		return self.paddings_shape_param('W')

	def kernel(self):
		"""Returns the width dimension of the ``kernel``."""
		return self.kernel_shape_param('W')

class Compute2D(ComputeSpatial):
	"""Base class for all 2D spatial-compute specific nodes."""
	def input_height(self):
		return self.variables['input_format'].index('H')

	def input_width(self):
		return self.variables['input_format'].index('W')

	def input_channels(self):
		return self.variables['input_format'].index('C')

	def stride_height(self):
		"""Returns the height dimension of ``strides``."""
		return self.strides_shape_param('H')

	def stride_width(self):
		"""Returns the width dimension of ``strides``."""
		return self.strides_shape_param('W')

	def padding_height(self):
		"""Returns the height dimension of ``paddings``."""
		return self.paddings_shape_param('H')

	def padding_width(self):
		"""Returns the width dimension of ``paddings``."""
		return self.paddings_shape_param('W')

	def kernel_height(self):
		"""Returns the height dimension of the ``kernel``."""
		return self.kernel_shape_param('H')

	def kernel_width(self):
		"""Returns the width dimension  of the ``kernel``."""
		return self.kernel_shape_param('W')

	def kernel_input(self):
		"""Returns the input dimension of the ``kernel``."""
		return self.kernel_shape_param('I')

	def kernel_output(self):
		"""Returns the output dimension of the ``kernel``."""
		return self.kernel_shape_param('O')

class Compute3D(Compute2D):
	"""Base class for all 3D spatial-compute specific nodes."""
	def input_depth(self):
		return self.variables['input_format'].index('D')

	def stride_depth(self):
		"""Returns the depth dimension of ``strides``."""
		return self.strides_shape_param('D')

	def padding_depth(self):
		"""Returns the depth dimension of ``paddings``."""
		return self.paddings_shape_param('D')

	def kernel_depth(self):
		"""Returns the depth dimension of the ``kernel``."""
		return self.kernel_shape_param('D')

class Conv2D(Compute2D, Intializable):
	"""2D Convolution node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/z4b54ra>`__.
	"""
	def with_group(self, group):
		"""Sets the 2D Convolution ``group`` parameter."""
		self.variables['group'] = group
		return self

	def get_group(self):
		"""Returns the 2D Convolution ``group`` parameter."""
		return self.variables.get('group')

class LocallyConnected2D(Conv2D, Intializable):
	"""2D Locally Connected node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/lhapxou>`__.
	"""
	pass

class TransposedConv2D(Conv2D, Intializable):
	"""Transposed 2D Convolution node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/zzzmeh4>`__.
	"""
	@dispatch(NodeDescriptor, basestring)
	def with_output(self, output, output_format):
		"""Sets the ``output`` descriptor."""
		self.variables['output'] = output
		self.variables['output_format'] = output_format
		return self

	def output_shape(self):
		"""Returns ``output`` descriptor."""
		return self.variables['output'].shape

	def output_shape_param(self, param):
		"""Returns the shape parameter of the ``output``."""
		index = self.variables['output_format'].index(param)
		return self.variables['output'].shape[index]

	def output_height(self):
		"""Returns the height dimension of the ``output``."""
		return self.output_shape_param('H')

	def output_width(self):
		"""Returns the width dimension of the ``output``."""
		return self.output_shape_param('W')

class BatchNormalization(Compute2D, Intializable):
	"""Batch Normalization node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/hsnz8hx>`__.
	"""
	@dispatch(NodeDescriptor)
	def with_population_mean(self, mean):
		"""Sets the population ``mean`` descriptor."""
		self.variables['population_mean'] = mean
		return self

	@dispatch(NodeDescriptor)
	def with_population_variance(self, variance):
		"""Sets the population ``variance`` descriptor."""
		self.variables['population_variance'] = variance
		return self

	def with_scale_op(self, scale):
		"""Sets ``id`` if the subsequent ``scale`` op."""
		self.variables['scale'] = scale
		return self

	def with_epsilon(self, epsilon):
		"""Sets the :math:`\\epsilon` parameter of the BN."""
		self.variables['epsilon'] = epsilon
		return self

	def with_alpha(self, epsilon):
		"""Sets the :math:`\\alpha` moving average parameter of the BN."""
		self.variables['alpha'] = epsilon
		return self

	def get_epsilon(self):
		"""Returns the :math:`\\epsilon` parameter of the BN."""
		return self.variables.get('epsilon')

	def get_alpha(self):
		"""Returns the :math:`\\alpha` moving average parameter of the BN."""
		return self.variables.get('alpha')

	def get_population_mean(self):
		"""Returns the population ``mean`` descriptor."""
		return self.variables.get('population_mean')

	def get_population_variance(self):
		"""Returns the population ``variance`` descriptor."""
		return self.variables.get('population_variance')

	def get_scale_op(self):
		"""Returns ``id`` if the subsequent ``scale`` op."""
		return self.variables.get('scale')

class LocalContrastNormalization(Compute2D, Intializable):
	"""Local Contrast Normalization node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/jgfckja>`__.
	"""
	pass

class Pooling2D(Compute2D):
	"""Base class for all 2D Pooling nodes/ops."""
	pass

class Pooling3D(Compute3D):
	"""Base class for all 3D Pooling nodes/ops."""
	pass

class PyramidPooling2D(Compute2D):
	"""Base class for all 2D Spatial Pyramid Pooling nodes/ops.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/lwxyy26>`__.
	"""
	def with_pyramid_height(self, height):
		"""Sets the pyramid ``height`` parameter."""
		self.variables['pyramid_height'] = height
		self.with_input_format('NCHW')
		return self

	def get_pyramid_height(self):
		"""Returns the pyramid ``height`` parameter."""
		return self.variables.get('pyramid_height')

class Pooling1D(Compute1D):
	"""Base class for all 1D Pooling nodes/ops."""
	pass

class MaxPooling2D(Pooling2D):
	"""Maximum 2D Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <http://tinyurl.com/jf8z7sg>`__.
	"""
	pass

class AvgPooling2D(Pooling2D):
	"""Average 2D Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/zskely9>`__.
	"""
	pass

class StochasticPooling2D(Pooling2D):
	"""Stochastic 2D Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/lb6wgmj>`__.
	"""
	pass

class MaxPyramidPooling2D(PyramidPooling2D):
	"""Maximum 2D Spatial Pyramid Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/lwxyy26>`__.
	"""
	pass

class AvgPyramidPooling2D(PyramidPooling2D):
	"""Average 2D Spatial Pyramid Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/lwxyy26>`__.
	"""
	pass

class StochasticPyramidPooling2D(PyramidPooling2D):
	"""Stochastic 2D Spatial Pyramid Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/lwxyy26>`__.
	"""
	pass

class MaxPooling1D(Pooling1D):
	"""Maximum 1D Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/jekrcc7>`__.
	"""
	pass

class AvgPooling1D(Pooling1D):
	"""Average 1D Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/js4xj9w>`__.
	"""
	pass

class MaxPooling3D(Pooling3D):
	"""Maximum 3D Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/kxarx42>`__.
	"""
	pass

class AvgPooling3D(Pooling3D):
	"""Average 3D Pooling node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/kxjlpuc>`__.
	"""
	pass