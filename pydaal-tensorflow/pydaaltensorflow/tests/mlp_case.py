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

'''
A Multi-Layer Perceptron Network example using TensorFlow library.

Adapted from: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import pydaaltensorflow as pydaal  
import tensorflow as tf
import numpy as np
import os

def get_variable(name, shape):
  return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(seed=0))

def fully_connected(x, weights='', w_shape=[], biases='', b_shape=[]):
    W = get_variable(name=weights, shape=w_shape)
    b = get_variable(name=biases, shape=b_shape)
    # Fully connected layer with added biases
    return tf.matmul(x, W) + b

def fully_connected_with_biass_add(x, weights='', w_shape=[], biases='', b_shape=[]):
    W = get_variable(name=weights, shape=w_shape)
    b = get_variable(name=biases, shape=b_shape)
    # Fully connected layer with added biases
    return tf.nn.bias_add(tf.matmul(x, W), b)

# Create model
def mlp_net(x, n_classes, dropout):

    full1 = fully_connected_with_biass_add(x, weights='wd1', w_shape=[2000, 1000], biases='bd1', b_shape=[1000])
    full1 = tf.nn.dropout(tf.tanh(full1), dropout)

    full2 = fully_connected(full1, weights='wd2', w_shape=[1000, 500], biases='bd2', b_shape=[500])
    full2 = tf.nn.dropout(tf.tanh(full2), dropout)
    
    # Output, class prediction
    full3 = fully_connected(full2, weights='w_out', w_shape=[500, n_classes], biases='b_out', b_shape=[n_classes])

    return full3

# Run model
def run_case(data, checkpoint_dir):
    # Network Parameters
    n_classes = 10

    # reset the Graph
    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder(tf.float32)
    dropout = tf.placeholder(tf.float32)

    # Construct model
    pred = mlp_net(x, n_classes, dropout)

    # Transform to the Intel DAAL model
    model = pydaal.transform_all()

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Create a saver 
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=0)
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=0)

        predictions  = sess.run(pred, feed_dict={x: data, dropout: 1.})

    # Provide a reference path to PyDAAL model
    pydaal.dump_model(model, checkpoint_dir)

    return predictions