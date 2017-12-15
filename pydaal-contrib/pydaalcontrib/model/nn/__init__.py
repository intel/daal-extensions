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
PyDAAL-contrib's module for Neural Network models
-------------------------------------------------

Provides:
	1. Base/final classes for NN models, *e.g.* :py:class:`pydaalcontrib.model.nn.Model`.
	2. Base/final classes for nodes/ops, *e.g.* :py:class:`pydaalcontrib.model.nn.Conv2D`.
	3. Base/final classes for intializers, *e.g.* :py:class:`pydaalcontrib.model.nn.Xavier`.
	4. Base/final classes for node descriptors, *e.g.* :py:class:`pydaalcontrib.model.nn.Strides`.
"""

import sys, inspect
from .initializers import *
from .spatial import *
from .base import *
from .misc import *

__all__ = [name for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass) if name[0].isupper()]