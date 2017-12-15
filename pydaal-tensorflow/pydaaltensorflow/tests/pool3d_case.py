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
import numpy as np

def max_pool3d(x):
    # MaxPool2D wrapper
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1]*5, padding='VALID')

def avg_pool3d(x):
    # MaxPool2D wrapper
    return tf.nn.avg_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

# Create model
def pool_net(x):    
    # Max Pooling (down-sampling)
    max = max_pool3d(x)

    # Avg Pooling (down-sampling)
    avg = avg_pool3d(max)

    return avg

# Run model
def run_case(data, checkpoint_dir):
    # reshape data to TF format
    data = np.transpose(data, (0, 2, 3, 4, 1))

    # reset the Graph
    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=data.shape)

    # Construct model
    pred = pool_net(x)

    # Transform to the Intel DAAL model
    model = pydaal.transform(pred)

    # Launch the graph
    with tf.Session() as sess:
        predictions  = sess.run(pred, feed_dict={x: data})

    # Provide a reference path to PyDAAL model
    pydaal.dump_model(model, checkpoint_dir)

    return predictions