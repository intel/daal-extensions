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

## Intel DAAL specific
PYDAAL_LAYER = 'Layer'
PYDAAL_VARIABLE = 'Variable'
PYDAAL_MODEL = 'pydaal.model'
PYDAAL_BIASES_KERNEL = "biases.kernel"
PYDAAL_WEIGHTS_KERNEL = "weights.kernel"
PYDAAL_AVG_POOL_KERNEL = "avg_pool.kernel"
PYDAAL_MAX_POOL_KERNEL = "max_pool.kernel"
PYDAAL_STOCH_POOL_KERNEL = "stochastic_pool.kernel"
PYDAAL_LCN_KERNEL = "local_contrast_normalization.kernel"
PYDAAL_NOT_A_TENSOR = "Couldn't extract data from %s. Expecting <daal.data_management.Tensor>"
PYDAAL_NOT_A_PREDICTION = "Couldn't extract data from %s. Expecting <daal.algorithms.Prediction> instance/subclass"
PYDAAL_MODEL_WRONG_PATH = "Could not find the path (%s) to the PyDAAL model!"
PYDAAL_WRONG_NODE_DESCRIPTOR = "Could not assign a descriptor to the PyDAAL model node. Wrong type!"
PYDAAL_NOT_A_PREDICTION_LAYER = "Couldn't add a loss layer (%s) to the preidction model!"
PYDAAL_NOT_A_INITIALIZER = "Couldn't find in Intel DAAL a corresponding (%s) initializer!"
PYDAAL_ELU_NOT_IMPLEMENTED = "Couldn't import ELU layers/functionality. Not yet implemented!"