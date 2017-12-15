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

import pydaaltensorflow as pydaal  
import tensorflow as tf

try: # experimental feature which is not included into distibution yet
    import daal.algorithms.neural_networks.layers.elu.forward
    elu_layer = True
except ImportError:
    elu_layer = False

# Create model

def activation_net(x):
    if elu_layer:
        x = tf.nn.elu(x)
    x = tf.tanh(x)
    x = tf.sigmoid(x)
    x = tf.nn.softmax(x)
    x = tf.nn.softplus(x)
    return tf.nn.relu(x)

# Run model
def run_case(data, checkpoint_dir):

    # reset the Graph
    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder(tf.float32)

    # Construct model
    pred = activation_net(x)

    # Launch the graph
    with tf.Session() as sess:
        predictions  = sess.run(pred, feed_dict={x: data})

    # Transform to the Intel DAAL model
    model = pydaal.transform(pred)
    # Provide a reference path to PyDAAL model
    pydaal.dump_model(model, checkpoint_dir)

    return predictions