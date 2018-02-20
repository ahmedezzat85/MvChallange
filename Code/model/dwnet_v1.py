"""MobileNet v1.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.slim as slim

def Conv(data, filters, kernel, stride, base_name):
  """ """
  end_point = base_name
  return slim.conv2d(data, filters, kernel, stride=stride, normalizer_fn=slim.batch_norm, scope=end_point)

def DepthWiseConv(data, filters, kernel, stride, base_name):
  """ """
  end_point = base_name + '_depthwise'
  net = slim.separable_conv2d(data, None, kernel, depth_multiplier=1, stride=stride, rate=1,
                              normalizer_fn=slim.batch_norm, scope=end_point)
  end_point = base_name + '_pointwise'
  net = slim.conv2d(net, filters, [1, 1], stride=1, normalizer_fn=slim.batch_norm, scope=end_point)
  return net

def ConvBlock(data, filters=[], stride=1, data_format='NHWC', base_name=None):
  """ """
  if filters[0] > 0:
    # 1x1 Conv
    end_point = base_name + '_b0_1x1'
    b0 = slim.conv2d(data, filters[0], [1, 1], stride=stride, normalizer_fn=slim.batch_norm, scope=end_point)
  
  # 3x3 Conv
  end_point = base_name + '_b1_3x3'
  b1 = DepthWiseConv(data, filters[1] , [3,3], stride, end_point)

  # 2x (3x3) Conv
  end_point = base_name + '_b20_3x3'
  b2 = DepthWiseConv(data, filters[2] , [3,3], stride, end_point)
  end_point = base_name + '_b21_3x3'
  b2 = DepthWiseConv(b2, filters[2] , [3,3], 1, end_point)
  concat_axis = 1 if data_format.startswith('NC') else 3
  if filters[0] > 0:
    net = tf.concat([b0, b1, b2], concat_axis, name=base_name+'_concat')
  else:
    net = tf.concat([b1, b2], concat_axis, name=base_name+'_concat')
  return net

def dwnet_v1_base(inputs, data_format, scope=None):
  """Mobilenet v1.

  Constructs a Mobilenet v1 network from inputs to the given final endpoint.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
  """
  with tf.variable_scope(scope, 'DwNet', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
      net = inputs
      net = Conv(net, 32, [3,3], 2, 'Conv_0')                           # 224 -> 112
      net = DepthWiseConv(net, 64  , [3,3], 1, 'Conv_1')                # 112 -> 112
      net = DepthWiseConv(net, 128 , [3,3], 2, 'Conv_2')                # 112 ->  56
      net = DepthWiseConv(net, 128 , [3,3], 1, 'Conv_3')                #  56 ->  56
      net = DepthWiseConv(net, 256 , [3,3], 2, 'Conv_4')                #  56 ->  28
      net = ConvBlock(net, [ 64,  96,  96], 1, data_format, 'Block_5')  #  28 ->  28
      net = ConvBlock(net, [ 64, 128, 128], 1, data_format, 'Block_6')  #  28 ->  28
      net = ConvBlock(net, [  0, 256, 256], 2, data_format, 'Block_7')  #  28 ->  14
      net = ConvBlock(net, [192, 160, 160], 1, data_format, 'Block_8')  #  14 ->  14
      net = ConvBlock(net, [192, 160, 160], 1, data_format, 'Block_9')  #  14 ->  14
      net = ConvBlock(net, [ 96, 208, 208], 1, data_format, 'Block_10') #  14 ->  14
      net = ConvBlock(net, [128, 320, 320], 1, data_format, 'Block_11') #  14 ->  14
      net = ConvBlock(net, [  0, 384, 384], 2, data_format, 'Block_12') #  14 ->  7
      net = ConvBlock(net, [384, 320, 320], 1, data_format, 'Block_13') #  7 ->   7
      net = ConvBlock(net, [256, 384, 384], 1, data_format, 'Block_14') #  7 ->   7
      return net

def dwnet_v1(inputs, num_classes=200, dropout_keep_prob=0.999, is_training=True, scope='DwNet',
  data_format='NHWC',
  weights_init=tf.variance_scaling_initializer(scale=2.34, distribution='uniform')):  
  """Mobilenet v1 model for classification.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    prediction_fn: a function to get predictions out of logits.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: Input rank is invalid.
  """
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'decay': 0.99,
      'epsilon': 0.001,
      'data_format': data_format,
      'fused': True
  }

  with slim.arg_scope([slim.conv2d, slim.separable_conv2d], weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm, data_format=data_format):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params) as sc:
      with tf.variable_scope(scope, 'DwNet', [inputs]) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
          net = dwnet_v1_base(inputs, data_format, scope=scope)
          with tf.variable_scope('Logits'):
            # Global average pooling.
            if data_format.startswith('NC'):
              net = tf.reduce_mean(net, [2, 3], keep_dims=True, name='global_pool')
            else:
              net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            # 1 x 1 x 1024
            net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv_softmax')
            logits = slim.flatten(logits, scope='Flatten')
            predictions = tf.nn.softmax(logits, name='Predictions')
  return logits, predictions