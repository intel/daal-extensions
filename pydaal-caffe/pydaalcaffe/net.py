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

from daal.algorithms.neural_networks.layers.batch_normalization.forward import populationMean, populationVariance
from pydaalcontrib.helpers import initialize_weights, initialize_input
from pydaalcontrib.model.nn import *
from pydaalcontrib.helpers import *
from .kaffe import CaffeResolver
import pydaalcontrib.nn
import numpy as np
import daal
import sys
import os

class DAALNet(pydaalcontrib.nn.DAALNet):
    """Wrapper class for working with :obj:`daal.algorithms.neural_networks` package.

    Supports conversions of `Caffe <http://caffe.berkeleyvision.org>`__ layers, blobs and variables to Intel DAAL.
    """

    def build_model(self, model, trainable, **kw_args):
        """Build a specific Caffe-to-DAAL model based on the provided DAAL model and `caffe_model_path` in `kw_args`.

        Args:
            model (:py:class:`pydaalcontrib.model.ModelBase` or :obj:`str`): Instance of a model or a path to the folder/file containing the model (*pydaal.model*) file.
            trainable (:obj:`bool`): Flag indicating whether `training` or `prediction` model to be built.
            kw_args (:obj:`dict`): Different keyword args which might be of use in the DAALNet class.

        Returns:
            :obj:`daal.algorithms.neural_networks.prediction.Model`
        """
        model_given = 'caffe_model_path' in kw_args and os.path.isfile(kw_args['caffe_model_path'])
        daal_model = pydaalcontrib.nn.DAALNet.build_model(self, model, trainable, **kw_args)
        self.resolver = CaffeResolver()

        # do nothing if 'rebuild' option is specified or `daal_model` is not initialized
        if self.do_rebuild or daal_model is None:
            return daal_model

        # do nothing if Caffe model path is not provided
        if 'caffe_model_path' not in model.__dict__ and not model_given:
            _show_fallback_msg()
            return daal_model

        if model_given:
            # overriding Caffe model path if explicitly given in `kw_args`
            model.caffe_model_path = kw_args['caffe_model_path']

        if 'data_dtype' not in kw_args:
            data_dtype = np.float32
        else:
            data_dtype = kw_args['data_dtype']
            
        layers = self.load_layers(model, data_dtype)

        # Set weights and biases for Intializable nodes
        for node in self.descriptor.nodes.values():
            if isinstance(node, BatchNormalization):
                # very special treatment of BatchNormalization op as it corresponds to BN+Scale in Caffe
                initialize_input(daal_model, node.daal_id, np.squeeze(layers[node.id][0]), populationMean, trainable)
                initialize_input(daal_model, node.daal_id, np.squeeze(layers[node.id][1]), populationVariance, trainable)
                scale_op_id = node.get_scale_op()

                if scale_op_id is not None:
                    biases = node.biases().shape_data(layers[scale_op_id][-1]) if node.has_variable('biases') else None
                    weights = node.weights().shape_data(layers[scale_op_id][0]) if node.has_variable('weights') else None
                    initialize_weights(daal_model, node.daal_id, weights, biases, trainable)
                    
            elif isinstance(node, Intializable):
                # Fetch all blobs/data from the Caffe layer
                biases = node.biases().shape_data(layers[node.id][-1]) if node.has_variable('biases') else None
                weights = node.weights().shape_data(layers[node.id][0]) if node.has_variable('weights') else None
                weights = _shape_tensor_to_daal_format(weights, node)
                # Initialize the layer with all weights and biases
                initialize_weights(daal_model, node.daal_id, weights, biases, trainable)

        return daal_model

    def load_layers(self, model, data_dtype):
        caffe_model = self.resolver.NetParameter()
        with open(model.caffe_model_path, 'rb') as file:
            caffe_model.MergeFromString(file.read())

        layers = caffe_model.layer or caffe_model.layers
        tuple = lambda layer: (layer.name, self.normalize_data(layer, data_dtype))
        return dict([tuple(layer) for layer in layers if layer.blobs])

    def normalize_data(self, layer, data_dtype):
        normalized = list()

        for blob in layer.blobs:
            if len(blob.shape.dim):
                dims = blob.shape.dim
                O, I, H, W = map(int, [1] * (4 - len(dims)) + list(dims))
            else:
                O, I, H, W = blob.num, blob.channels, blob.height, blob.width

            data = np.array(blob.data, dtype=data_dtype).reshape(O, I, H, W)
            normalized.append(data)

        return normalized

def _shape_tensor_to_daal_format(tensor, current):
    if tensor is None:
        return tensor

    if isinstance(current, Conv2D) and current.get_group() > 1:
        # reshaping to conform to the Intel DAAL format <GOIHW> with nGroups > 1
        new_shape = list(tensor.shape)
        new_shape[0] = int(new_shape[0] / current.get_group())
        new_shape.insert(0, current.get_group())
        tensor = tensor.reshape(new_shape)
        
    elif isinstance(current, FullyConnected):
        previous_2d = find_previous_ops(current, [Conv2D])
        previous_fc = find_previous_ops(current, [FullyConnected])
        previous_concat = find_previous_ops(current, [Concat])

        if len(previous_concat) > 0 and len(previous_fc) == 0:
            # reshaping to conform to the Intel DAAL format from 2d to FC layer
            # converting Caffe format <OI> for FC layer to the Intel DAAL format <OIHW>
            # from all Conv2D availablenodes preceeding the Concat node 
            conv2d_ops = find_previous_ops(previous_concat[0], [Conv2D])
            
            if len(conv2d_ops) > 0:
                kernel_output = 0
                for op in conv2d_ops:
                    kernel_output += op.kernel_output()

                kernel = int(np.sqrt(tensor.shape[1] / kernel_output))  
                new_shape = [tensor.shape[0], kernel_output, kernel, kernel]
                tensor = tensor.reshape(new_shape)

        elif len(previous_2d) > 0 and len(previous_fc) == 0:
            # reshaping to conform to the Intel DAAL format from 2d to FC layer
            # converting Caffe format <OI> for FC layer to the Intel DAAL format <OIHW> 
            kernel_output = previous_2d[0].kernel_output()
            kernel = int(np.sqrt(tensor.shape[1] / kernel_output))  
            new_shape = [tensor.shape[0], kernel_output, kernel, kernel]
            tensor = tensor.reshape(new_shape)
        
    return tensor

def _show_fallback_msg():
    msg = '''
--------------------------------------------------------------------------------------
    WARNING: Caffe model file is not provided or missing!
    Cannot fully initialize the Intel DAAL model from the non-existent Caffe model.
--------------------------------------------------------------------------------------

'''
    sys.stderr.write(msg)