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

def Conv(data, filters, kernel, stride, index):
  """ """
  end_point = 'Conv_%d' % index
  return slim.conv2d(data, filters, kernel, stride=stride, normalizer_fn=slim.batch_norm, scope=end_point)

def DepthWiseConv(data, filters, kernel, stride, index):
  """ """
  base = 'Conv_%d' % index
  end_point = base + '_depthwise'
  net = slim.separable_conv2d(data, None, kernel, depth_multiplier=1, stride=stride, rate=1,
                              normalizer_fn=slim.batch_norm, scope=end_point)
  end_point = base + '_pointwise'
  net = slim.conv2d(net, filters, [1, 1], stride=1, normalizer_fn=slim.batch_norm, scope=end_point)
  return net

def dwnet_v1_base(inputs, scope=None):
  """Mobilenet v1.

  Constructs a Mobilenet v1 network from inputs to the given final endpoint.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
  """
  end_points = {}

  with tf.variable_scope(scope, 'DwNet', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
      net = inputs
      net = Conv(net, 32, [3,3], 2, 0)
      net = DepthWiseConv(net, 64  , [3,3], 1, 1)
      net = DepthWiseConv(net, 128 , [3,3], 2, 2)
      net = DepthWiseConv(net, 128 , [3,3], 1, 3)
      net = DepthWiseConv(net, 256 , [3,3], 2, 4)
      net = DepthWiseConv(net, 256 , [3,3], 1, 5)
      net = DepthWiseConv(net, 512 , [3,3], 2, 6)
      net = DepthWiseConv(net, 512 , [3,3], 1, 7)
      net = DepthWiseConv(net, 512 , [3,3], 1, 8)
      net = DepthWiseConv(net, 512 , [3,3], 1, 9)
      net = DepthWiseConv(net, 512 , [3,3], 1, 10)
      net = DepthWiseConv(net, 512 , [3,3], 1, 11)
      net = DepthWiseConv(net, 1024, [3,3], 2, 12)
      net = DepthWiseConv(net, 1024, [3,3], 1, 13)
      return net

def dwnet_v1(inputs,
              num_classes=200,
              dropout_keep_prob=0.999,
              is_training=True,
              prediction_fn=tf.contrib.layers.softmax,
              spatial_squeeze=True,
              reuse=None,
              scope='DwNet'):
  """Mobilenet v1 model for classification.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.

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
    raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                     len(input_shape))

  with tf.variable_scope(scope, 'DwNet', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
      net = dwnet_v1_base(inputs, scope=scope)
      with tf.variable_scope('Logits'):
        # Global average pooling.
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        predictions = prediction_fn(logits, scope='Predictions')
  return logits, predictions

def dwnet_v1_arg_scope(is_training=True):
  """Defines the default MobilenetV1 arg scope.

  Args:
    is_training: Whether or not we're training the model.

  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
  """
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'decay': 0.99,
      'epsilon': 0.001,
      'fused': True
  }

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.glorot_uniform_initializer()#tf.truncated_normal_initializer(stddev=stddev)
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d], weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=None):
        with slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as sc:
          return sc
