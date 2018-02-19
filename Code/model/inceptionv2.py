from __future__ import absolute_import

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from . inception_v2 import inception_v2_arg_scope, inception_v2

_CUR_DIR = os.path.dirname(__file__)
_MODEL_DIR = os.path.join(_CUR_DIR, '..', '..', 'pretrained', 'inception_v2')

class TFModel(object):
    """
    """
    def __init__(self, dtype, data_format, num_classes, input_size=224, dropout=0.999):
        print ('dropout     : ', dropout)
        print ('input size  : ', input_size)
        print ('num_classes : ', num_classes)
        self.chkpt   = os.path.join(_MODEL_DIR, 'inception_v2.ckpt')

        self.fmt          = data_format
        self.num_classes  = num_classes
        self.dropout_prob = float(dropout)

    def forward(self, data, is_training=True):
        """ """
        arg_scope = inception_v2_arg_scope(is_training, data_format=self.fmt)
        with slim.arg_scope(arg_scope):
            logits, end_points = inception_v2(data, num_classes=self.num_classes, data_fmt=self.fmt,
                                              dropout_keep_prob=self.dropout_prob,
                                              is_training=is_training)
        probs = tf.identity(end_points['Predictions'], name='output')
        return logits, probs

    def weight_init(self, tf_sess):
        if self.chkpt is not None:
            # Exclude classifier part from the loaded weights
            var_list = []
            for var in slim.get_model_variables('InceptionV2'):
                if not var.op.name.startswith('InceptionV2/Logits'):
                    var_list.append(var)
                else:
                    print ('EXCLUDE <', var.op.name, '>')

            # Initialize the model to the pretrained weights
            load_trained_weights = slim.assign_from_checkpoint_fn(self.chkpt, var_list)
            load_trained_weights(tf_sess)