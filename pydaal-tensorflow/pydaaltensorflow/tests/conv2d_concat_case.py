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
A Convolutional Network implementation example using TensorFlow library.

Adapted from: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import pydaaltensorflow as pydaal  
import tensorflow as tf
import numpy as np
import os

def get_variable(name, shape):
  return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(seed=0))

def conv2d(x, kernel, in_ch, out_ch, strides=[1, 1, 1, 1]):
    W = get_variable(name='wc', shape=[kernel[0], kernel[1], in_ch, out_ch])
    b = get_variable(name='bc', shape=[out_ch])
    # Conv2D wrapper, with bias and relu activation
    conv2d = tf.nn.conv2d(x, W, strides=strides, padding='SAME')
    return conv2d + b

# Create model
def concat_conv_net(x, use_gateway):
    if use_gateway:
        x = tf.sigmoid(x)
    
    # 1st Convolution Layer
    with tf.variable_scope('conv1') as scope:
        conv1 = conv2d(x, [5, 5], 3, 32)

    # 2nd Convolution Layer
    with tf.variable_scope('conv2') as scope:
        conv2 = conv2d(x, [5, 5], 3, 64)

    # 3rd Convolution Layer
    with tf.variable_scope('conv3') as scope:
        conv3 = conv2d(x, [5, 5], 3, 64)

    cat1 = tf.concat([conv1, conv2], 3)
    cat2 = tf.add_n([conv3, conv2])

    return tf.concat([cat2, cat1], 3)

# Run model
def run_case(data, checkpoint_dir, use_gateway=True):
    # reshape data to TF format
    data = np.transpose(data, (0, 2, 3, 1))

    # reset the Graph
    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=data.shape)

    # Construct model
    pred = concat_conv_net(x, use_gateway)

    # Transform to the Intel DAAL model
    model = pydaal.transform(pred)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Provide a reference path to PyDAAL model
    pydaal.dump_model(model, checkpoint_dir)

    # Create a saver 
    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep = 0)
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step = 0)

        predictions  = sess.run(pred, feed_dict={x: data})

    return predictions