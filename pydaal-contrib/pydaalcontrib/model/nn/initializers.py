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

class Initializer(object):
	"""Base class for all initializers."""
	pass

class Xavier(Initializer):
	"""Xavier initializer for ``weights`` or ``biases``.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/z8vugzm>`__.
	"""
	pass

class Uniform(Initializer):
	"""Uniform initializer for ``weights`` or ``biases``.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/zs2bhq9>`__.
	"""
	def __init__(self, left_bound, right_bound):
		self.left_bound = left_bound
		self.right_bound = right_bound

class Gaussian(Initializer):
	"""Gaussian initializer for ``weights`` or ``biases``.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/hh9jdbp>`__.
	"""
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

class TruncatedGaussian(Gaussian):
	"""Truncated Gaussian initializer for ``weights`` or ``biases``.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/jd3sy96>`__.
	"""
	def with_bounds(self, left_bound, right_bound):
		"""Sets the left and the right bounds for initializer.

		Args:
			left_bound (:obj:`float`): The left bound.
			right_bound (:obj:`float`): The right bound.
		"""
		self.left_bound = left_bound
		self.right_bound = right_bound
		return self

	def has_bounds(self):
		"""Assesses if bounds are set for initializer."""
		return 'left_bound' in self.__dict__ and 'right_bound' in self.__dict__