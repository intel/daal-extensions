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
A De-Convolutional Network implementation example using TensorFlow library.

Adapted from: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import pydaaltensorflow as pydaal  
import tensorflow as tf
import numpy as np
import os

def get_variable(shape):
  return tf.Variable(tf.random_normal(shape, seed=0))

def conv2d_transpose(x, kernel, in_ch, out_ch, strides=[1, 2, 2, 1], padding='VALID', output_shape=[]):
    W = get_variable(shape=[kernel[0], kernel[1], out_ch, in_ch])
    b = get_variable(shape=[out_ch])

    conv2d_transpose = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, padding=padding)
    conv2d_transpose = tf.nn.bias_add(conv2d_transpose, b)
    
    return conv2d_transpose

# Create model
def conv2d_transpose_net(x):
    input_shape = x.get_shape().as_list()

    # 1st Transposed Convolution Layer
    with tf.variable_scope('convt1') as scope:
        convt1 = conv2d_transpose(x, [5, 5], 3, 3, output_shape=[input_shape[0], 15, 15, 3])

    # 2nd Transposed Convolution Layer
    with tf.variable_scope('convt2') as scope:
        convt2 = conv2d_transpose(convt1, [5, 5], 3, 6, output_shape=[input_shape[0], 33, 33, 6])

    # 3rd Transposed Convolution Layer
    with tf.variable_scope('convt3') as scope:
        convt3 = conv2d_transpose(convt2, [6, 6], 6, 4, output_shape=[input_shape[0], 66, 66, 4], padding='SAME')

    return convt3

# Run model
def run_case(data, checkpoint_dir):
    # reshape data to TF format
    data = np.transpose(data, (0, 2, 3, 1))

    # reset the Graph
    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=data.shape)

    # Construct model
    pred = conv2d_transpose_net(x)

    # Transform to the Intel DAAL model
    model = pydaal.transform(pred)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Create a saver 
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep = 0)

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step = 0)

        predictions  = sess.run(pred, feed_dict={x: data})

    # Provide a reference path to PyDAAL model
    pydaal.dump_model(model, checkpoint_dir)

    return predictions