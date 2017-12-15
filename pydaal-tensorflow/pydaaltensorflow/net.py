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

from pydaalcontrib.helpers import load_model, initialize_weights
from pydaalcontrib.model.nn import *
from pydaalcontrib.helpers import *
import pydaalcontrib.nn
import tensorflow as tf
import numpy as np
import sys
import os

class DAALNet(pydaalcontrib.nn.DAALNet):
    """Wrapper class for working with :obj:`daal.algorithms.neural_networks` package.

    Supports conversions of `TensorFlow <https://www.tensorflow.org>`__ ops and variables to Intel DAAL.
    """

    def build_model(self, checkpoint_path, trainable, **kw_args):
        """Build a specific TF-to-DAAL model based on the Intel DAAL model and TF checkpoint files dicoverable at `checkpoint_path`.

        Args:
            checkpoint_path (:obj:`str`): Path to the TF checkpoint directory.
            trainable (:obj:`bool`): Flag indicating whether `training` or `prediction` model to be built.
            kw_args (:obj:`dict`): Different keyword args which might be of use in the DAALNet class.

        Returns:
            :obj:`daal.algorithms.neural_networks.prediction.Model`
        """
        model = pydaalcontrib.nn.DAALNet.build_model(self, checkpoint_path, trainable, **kw_args)
        
        # do nothing if 'rebuild' option is specified or `daal_model` is not initialized
        if self.do_rebuild or model is None:
            return model

        # check that `checkpoint_path` / `model_path` parameter is a valid choice 
        if not isinstance(checkpoint_path, basestring) and not 'model_path' in kw_args:
            _show_fallback_msg()
            return model
        elif 'model_path' in kw_args:
            # overriding `model_path` parameter if explicitly given in `kw_args`
            checkpoint_path = kw_args['model_path']

        # double check that `checkpoint_path` parameter is a valid choice 
        if not isinstance(checkpoint_path, basestring) or not os.path.isdir(checkpoint_path):
            _show_fallback_msg()
            return model

        # Read the checkpoint of the Tensorflow model
        state = tf.train.get_checkpoint_state(checkpoint_path)

        if state:
            reader = tf.train.NewCheckpointReader(state.model_checkpoint_path)

            # Set weights and biases for Intializable nodes
            for node in self.descriptor.nodes.values():
                if isinstance(node, Intializable):
                    # Read all weights from the Tensorflow checkpoint file
                    biases = node.biases().shape_data(reader.get_tensor(node.biases_name())) if node.has_variable('biases') else None
                    weights = node.weights().shape_data(reader.get_tensor(node.weights_name())) if node.has_variable('weights') else None
                    weights = _shape_tensor_to_daal_format(weights, node)
                    # Initialize the layer with all weights and biases
                    initialize_weights(model, node.daal_id, weights, biases, trainable)

        return model

def _shape_tensor_to_daal_format(tensor, current):
    if tensor is None:
        return tensor
        
    if isinstance(current, Compute2D):
        # reshaping TF format <HWIO> to the Intel DAAL format <OIHW>
        tensor = np.transpose(tensor, (3, 2, 0, 1))

    if isinstance(current, FullyConnected):
        previous_2d = find_previous_ops(current, [Conv2D])
        previous_fc = find_previous_ops(current, [FullyConnected])
        previous_concat = find_previous_ops(current, [Concat])

        if len(previous_concat) > 0 and len(previous_fc) == 0:
            # reshaping to conform to the Intel DAAL format from 2d to FC layer
            # converting TF format <IO> for FC layer to the Intel DAAL format <OIHW> 
            # from all Conv2D availablenodes preceeding the Concat node 
            conv2d_ops = find_previous_ops(previous_concat[0], [Conv2D])
            if len(conv2d_ops) > 0:
                kernel_output = 0
                for op in conv2d_ops:
                    kernel_output += op.kernel_output()

                kernel = int(np.sqrt(tensor.shape[0] / kernel_output))  
                new_shape = [tensor.shape[1], kernel, kernel, kernel_output]
                tensor = np.transpose(tensor.T.reshape(new_shape), (0, 3, 1, 2))

        elif len(previous_2d) > 0 and len(previous_fc) == 0:
            # reshaping to conform to the Intel DAAL format from 2d to FC layer
            # converting TF format <IO> for FC layer to the Intel DAAL format <OIHW> 
            kernel_output = previous_2d[0].kernel_output()
            kernel = int(np.sqrt(tensor.shape[0] / kernel_output))  
            new_shape = [tensor.shape[1], kernel, kernel, kernel_output]
            tensor = np.transpose(tensor.T.reshape(new_shape), (0, 3, 1, 2))

        else:
            # converting TF format <IO> for FC layer to the Intel DAAL format <OI> 
            tensor = np.transpose(tensor, (1, 0))
        
    return tensor


def _show_fallback_msg():
    msg = '''
-----------------------------------------------------------------------------------
    WARNING: Tensorflow chekpoint directory is not provided or missing!
    Cannot fully initialize the Intel DAAL model from a non-existent checkpoint.
-----------------------------------------------------------------------------------

'''
    sys.stderr.write(msg)
