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

from pydaalcontrib.model import ModelBase
from multipledispatch import dispatch
from .initializers import Initializer
from past.builtins import cmp
import numpy as np
import uuid
import sys

# increasing recursion limit for very deep models
sys.setrecursionlimit(10000)

try:
	basestring = basestring
except NameError:
	# 'basestring' is undefined, must be Python 3
	basestring = (str, bytes)

class Intializable(object):
	"""Base class for all nodes which have intializable ``weights`` and ``biases``."""
	@dispatch(Initializer)
	def with_initializer(self, initializer):
		"""Sets the initializer for ``weights`` or ``biases``."""
		self.initializer = initializer
		return self

	def has_initializer(self):
		"""Assesses if initializer is set."""
		return 'initializer' in self.__dict__

class NodeDescriptor(Intializable):
	"""Base class for all node descriptors, e.g. :py:class:`Strides`."""
	@dispatch(basestring, int)
	def __init__(self, name, shape):
		self.name = name
		self.shape = [shape]

	@dispatch(basestring, list)
	def __init__(self, name, shape):
		self.name = name
		self.shape = list(shape)

	@dispatch(basestring, np.ndarray)
	def __init__(self, name, data):
		self.name = name
		self.data = data
		self.shape = list(data.shape)

	def has_data(self):
		"""Assesses if `data` was set when descriptor was initialized."""
		return 'data' in self.__dict__

	def shape_data(self, data):
		"""Shapes external data according to internal shape."""
		return np.squeeze(data) if None in self.shape else data

class EmptyNodeDescriptor(NodeDescriptor):
	def shape_data(self, data):
		return np.zeros_like(data) if None in self.shape else np.zeros(self.shape)

class BiasNodeDescriptor(NodeDescriptor):
	def shape_data(self, data):
		return np.reshape(data, self.shape)

class Node(object):
	"""Base class for all nodes in a model."""
	def __init__(self):
		self.depth = 0
		self.inputs = list()
		self.outputs = list()
		self.id = uuid.uuid4()
		self.with_variables(dict())

	def __call__(self, output):
		self.outputs.append(output)
		output.inputs.append(self)
		output.depth = self.depth + 1

		return output

	def __eq__(self, other):
		return self.id == other.id

	def __hash__(self):
		return hash(self.id)

	# TODO: refactor when breadth-first adding to the topology is fixed in Intel DAAL 2018
	######################################################################################
	def __lt__(self, other):
		return self.__cmp__(other) < 0

	def __cmp__(self, other):
		if self.outputs == other.outputs != list(): 
			return np.sum([self.__order_cmp__(other, o) for o in self.outputs])
		elif cmp(self.depth, other.depth) != 0:
			return cmp(self.depth, other.depth)
		else:
			return self.__id_cmp__(other)

	def __id_cmp__(self, other):
		if type(self.id) == uuid.UUID or type(other.id) == uuid.UUID:
			return 0
		else:
			return cmp(self.id, other.id)

	def __order_cmp__(self, other, output):
		ids = [i.id for i in output.inputs]
		return cmp(ids.index(self.id), ids.index(other.id))
	######################################################################################

	def clear_outputs(self):
		self.outputs = list()
		return self

	def with_id(self, id):
		self.id = id
		return self

	def with_variables(self, variables):
		"""Sets within a node some ``variables``."""
		self.variables = variables
		return self

	def has_variable(self, variable):
		"""Assesses if ``variable`` is set."""
		return variable in self.variables

class LossNode(Node):
	pass

class ConnectedNode(Node):
	"""Base class for all nodes with ``weights`` and ``biases``."""
	@dispatch(NodeDescriptor)
	def with_weights(self, weights):
		"""Sets the ``weights`` descriptor."""
		self.variables['weights'] = weights
		return self

	@dispatch(NodeDescriptor)
	def with_biases(self, biases):
		"""Sets the ``biases`` descriptor."""
		self.variables['biases'] = biases
		return self

	def with_weights_initializer(self, initializer):
		"""Sets the ``weights`` initializer.

		Args:
			initializer (:py:class:`Initializer`): initializer.
		"""
		if self.has_variable('weights') and not self.weights().has_data():
			self.weights().with_initializer(initializer)
		
		return self

	def with_biases_initializer(self, initializer):
		"""Sets the ``biases`` initializer.

		Args:
			initializer (:py:class:`Initializer`): initializer.
		"""
		if self.has_variable('biases') and not self.biases().has_data():
			self.biases().with_initializer(initializer)
		
		return self

	def weights(self):
		"""Returns ``weights`` descriptor."""
		return self.variables.get('weights')

	def weights_shape(self):
		"""Returns the shape of ``weights``."""
		return self.variables.get('weights').shape

	def weights_name(self):
		"""Returns the name of ``weights``."""
		return self.variables.get('weights').name

	def weights_data(self):
		"""Returns the underlying data of ``weights``."""
		if self.has_variable('weights') and self.weights().has_data():
			return self.weights().data
		else:
			return None

	def biases(self):
		"""Returns ``biases`` descriptor."""
		return self.variables.get('biases')

	def biases_shape(self):
		"""Returns the shape of ``biases``."""
		return self.variables.get('biases').shape

	def biases_name(self):
		"""Returns the name of ``biases``."""
		return self.variables.get('biases').name

	def biases_data(self):
		"""Returns the underlying data of ``biases``."""
		if self.has_variable('biases') and self.biases().has_data():
			return self.biases().data
		else:
			return None

	def num_biases(self):
		"""Returns the number of ``biases``."""
		return self.variables.get('biases').shape[0]

class Concat(Node):
	"""Concatenation node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/l58acvo>`__.
	"""
	def with_concat_dimension(self, dimension):
		"""Sets the dimension across which concatenation happens."""
		self.variables['concat_dimension'] = dimension
		return self

	def get_concat_dimension(self):
		"""Gets the dimension across which concatenation happens."""
		return self.variables.get('concat_dimension')

class Split(Node):
	"""Split node/op.

	Find more information on the official Intel DAAL `doc pages <https://tinyurl.com/kcepwkk>`__.
	"""
	def with_nsplits(self, nsplits):
		"""Sets the number of splits."""
		self.variables['nsplits'] = nsplits
		return self

	def get_nsplits(self):
		"""Gets the number of splits."""
		return self.variables.get('nsplits')

class Model(ModelBase):
	"""Base class for all NN models.

	Examples
	--------
	>>> from pydaalcontrib.model.nn import *
	>>> from pydaalcontrib.nn import *
	...
	>>> x = conv2d([3,3,32], strides=[1,1], paddings=[2,2])
	>>> x = x(max_pool2d([2,2])(Relu()))
	>>> x = x(dropout(keep_prob=.9)))
	>>> model = Model(x)
	"""
	@dispatch()
	def __init__(self):
		self.outputs = list()
		self.nodes = dict()

	@dispatch(list)
	def __init__(self, outputs):
		self.nodes = dict()
		self.outputs = outputs
		for node in outputs:
			self.nodes[node.id] = node
			self._traverse(node)

	@dispatch(Node)
	def __init__(self, output):
		self.nodes = {output.id : output}
		self.outputs = [output]
		self._traverse(output)

	@dispatch(Node)
	def _traverse(self, node):
		for input in node.inputs:
			self.nodes[input.id] = input
			self._traverse(input)

	@dispatch(Node, Node)
	def _add(self, node, next_node):
		# ensuring the same node
		if node.id not in self.nodes:
			self.nodes[node.id] = node
		else:
			node = self.nodes[node.id]
		# ensuring the same node
		if next_node.id not in self.nodes:
			self.nodes[next_node.id] = next_node
		else:
			next_node = self.nodes[next_node.id]

		next_node = node(next_node)
		self._increase_depth(next_node)

		# clean-up of the next node inputs and outputs
		for input in node.inputs:
			if input in next_node.inputs:
				next_node.inputs.remove(input)

		for output in node.outputs:
			if output in next_node.outputs: 
				next_node.outputs.remove(output)

		return self

	@dispatch(Node, type(None))
	def _add(self, node, last):
		self.nodes[node.id] = node
		if len(self.outputs) == 0:
			self.outputs.append(node)
		else:
			last = self.outputs[-1]
			self.outputs = [node]
			self._add(last, node)

		return self

	@dispatch(Node)
	def _increase_depth(self, node):
		for output in node.outputs:
			if output.depth <= node.depth:
				output.depth = node.depth + 1
				self._increase_depth(output)
