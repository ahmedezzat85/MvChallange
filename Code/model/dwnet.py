from __future__ import absolute_import

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from . dwnet_v1 import dwnet_v1

class TFModel(object):
    """
    """
    def __init__(self, dtype, data_format, num_classes, input_size=224, dropout=0.999):
        self.config  = {'Input Size': input_size, 'Dropout': dropout}
        self.num_classes  = num_classes
        self.dropout_prob = float(dropout)

    def forward(self, data, is_training=True):
        """ """
        logits, predictions = dwnet_v1(data, num_classes=self.num_classes, is_training=is_training,
                                        dropout_keep_prob=self.dropout_prob)
        probs = tf.identity(predictions, name='output')
        return logits, probs

    def weight_init(self, tf_sess):
        pass