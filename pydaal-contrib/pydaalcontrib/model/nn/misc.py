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

from .initializers import *
from .spatial import *
from .base import *

class FullyConnected(ConnectedNode, Intializable):
	"""Fully Connected node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/hl2aed2>`__.
	"""
	pass

class Relu(Node):
	"""Rectified Linear Unit (ReLU) node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/zcsun4s>`__.
	"""
	pass

class Elu(Node):
	"""Exponential Linear Unit (ELU) node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/zcsun4s>`__.
	"""
	pass

class ParametricRelu(ConnectedNode):
	"""Parametric Rectified Linear Unit (pReLU) node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/jxgyz9g>`__.
	"""
	pass

class SmoothRelu(Node):
	"""Smooth Rectified Linear Unit (Smooth ReLU) node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/hmao2t7>`__.
	"""
	pass

class Softmax(Node):
	"""Softmax node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/hs5yb7a>`__.
	"""
	def with_dimension(self, dimension):
		"""Sets the dimension (in a tensor) across which Softmax is being computed."""
		self.variables['dimension'] = dimension
		return self

	def get_dimension(self):
		"""Gets the dimension (in a tensor) across which Softmax is being computed."""
		return self.variables.get('dimension')

class Sigmoid(Node):
	"""Sigmoid (Logistic) node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/htnam82>`__.
	"""
	pass

class Tanh(Node):
	"""Hyperbolic Tangent (Tanh) node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/h4um57q>`__.
	"""
	pass

class Abs(Node):
	"""Absolute (Abs) value node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/zrukhr4>`__.
	"""
	pass

class SigmoidCrossEntropy(LossNode):
	"""Sigmoid (Logistic) Cross-Entropy loss node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/jeuzd5c>`__.
	"""
	pass

class SoftmaxCrossEntropy(LossNode):
	"""Softmax Cross-Entropy loss node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/hlpjdmq>`__.
	"""
	def with_class_dimension(self, dimension):
		"""Sets the class dimension (in a tensor) across which Cross-Entropy is being computed."""
		self.variables['class_dimension'] = dimension
		return self

	def get_class_dimension(self):
		"""Gets the class dimension (in a tensor) across which Cross-Entropy is being computed."""
		return self.variables.get('class_dimension')

	def get_softmax_part(self):
		return Softmax().with_dimension(self.get_class_dimension())

class Dropout(Node):
	"""Dropout node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/z6bnv4z>`__.
	"""
	def with_probability(self, probability):
		"""Sets the retention (keep) ``probability``."""
		self.variables['probability'] = probability
		return self

	def get_probability(self):
		"""Gets the retention (keep) ``probability``."""
		return self.variables.get('probability')

class Identity(Split):
	"""Identity node returning a copy of an input."""
	def __init__(self):
		super(Identity, self).__init__()
		self.variables['nsplits'] = 1

class Scale(Identity):
	"""Dummy node which will be implemented in the future."""
	def with_param(self, param):
		"""Sets all parameters at once."""
		self.variables['param'] = param
		return self

	def get_param(self):
		"""Gets all parameters at once."""
		return self.variables.get('param')

class ElementwiseSum(Node):
	"""Elementwise sum node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/y79xelza>`__.
	"""
	def with_coefficients(self, coefficients):
		"""Sets new coefficients applied to the sum."""
		self.variables['coefficients'] = coefficients
		return self

	def get_coefficients(self):
		"""Gets new coefficients applied to the sum."""
		return self.variables.get('coefficients')

class Reshape(Node):
	"""Reshape node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/j3d7kc5>`__.
	"""
	def with_new_dimensions(self, dimensions):
		"""Sets new output dimensions of a tensor."""
		self.variables['new_dimensions'] = dimensions
		return self

	def get_new_dimensions(self):
		"""Gets new output dimensions of a tensor."""
		return self.variables.get('new_dimensions')

class LocalResponseNormalization(Node):
	"""Local Response Normalization node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/h2utgtk>`__.
	"""
	def with_depth(self, depth):
		"""Sets the normalization ``depth``."""
		self.variables['depth'] = depth
		return self

	def get_depth(self):
		"""Gets the normalization ``depth``."""
		return self.variables.get('depth')

	def with_alpha(self, alpha):
		"""Sets the normalization :math:`\\alpha` param."""
		self.variables['alpha'] = alpha
		return self

	def get_alpha(self):
		"""Gets the normalization :math:`\\alpha` param."""
		return self.variables.get('alpha')

	def with_beta(self, beta):
		"""Sets the normalization :math:`\\beta` param."""
		self.variables['beta'] = beta
		return self

	def get_beta(self):
		"""Gets the normalization :math:`\\beta` param."""
		return self.variables.get('beta')

	def with_bias(self, bias):
		"""Sets the normalization ``bias`` (:math:`\\kappa`) param."""
		self.variables['bias'] = bias
		return self

	def get_bias(self):
		"""Gets the normalization ``bias`` (:math:`\\kappa`) param."""
		return self.variables.get('bias')