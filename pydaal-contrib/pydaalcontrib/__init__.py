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
PyDAAL-contrib's main module
----------------------------

Designed as a contribution to Intel DAAL library :py:mod:`pydaalcontrib` module aims at providing 
reusable and easily understood APIs in a *pythonic* way for Data Science and Machine Learning 
communities. It alleviates usage of the Intel DAAL data structures and abstracts away some of the 
inner and tedious API details. The main focus of the module is interpretability, ease of use 
and immediate applicability of the Intel DAAL primitives and functionality in Python-centric technical 
computing workflows and environments, e.g. Anaconda distribution or Scikit-Learn library.

Provides
	1. Basic support functionality to alleviate usage of the Intel DAAL internal data structures
	2. Basic support functionality for loading and dumping Intel DAAL models (for instance Deep NNs)
"""

from .helpers import DataReader, dump_model, load_model

__author__ = "Vilen Jumutc"
__version__ = "2017.0"

__all__ = ['DataReader', 'dump_model', 'load_model']