from __future__ import absolute_import

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from . dwnet_v1 import dwnet_v1, dwnet_v1_arg_scope

class TFModel(object):
    """
    """
    def __init__(self, dtype, data_format, num_classes, input_size=224, dropout=0.999):
        print ('dropout     : ', dropout)
        print ('input size  : ', input_size)
        print ('num_classes : ', num_classes)

        self.num_classes  = num_classes
        self.dropout_prob = float(dropout)

    def forward(self, data, is_training=True):
        """ """
        arg_scope = dwnet_v1_arg_scope(is_training)
        with slim.arg_scope(arg_scope):
            logits, predictions = dwnet_v1(data, num_classes=self.num_classes, 
                                            dropout_keep_prob=self.dropout_prob,
                                            is_training=is_training)
        probs = tf.identity(predictions, name='output')
        return logits, probs

    def weight_init(self, tf_sess):
        pass