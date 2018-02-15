from __future__ import absolute_import

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from . mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope, Conv, DepthSepConv

_CUR_DIR = os.path.dirname(__file__)
_MODEL_DIR = os.path.join(_CUR_DIR, 'mobilenet_v1')

# _CONV_DEFS specifies the MobileNet body
_FREEZE_ALL = [
    Conv(kernel=[3, 3], stride=2, depth=32, trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64  , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=False),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024, trainable=False),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024, trainable=False)
]

_TRAIN_ALL = [
    Conv(kernel=[3, 3], stride=2, depth=32, trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64  , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512 , trainable=True),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024, trainable=True),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024, trainable=True)
]

class TFModel(object):
    """
    """
    def __init__(self, dtype, data_format, num_classes, model='TRAIN', 
                    mult=1.0, input_size=224, dropout=0.999):
        model = str(model)
        print ('Pretrained mobilenet <', model, '>')
        self.net_cfg = _FREEZE_ALL if model.upper() == 'FREEZE' else _TRAIN_ALL
        self.chkpt   = os.path.join(_MODEL_DIR, 'mobilenet_v1_'+str(mult)+'_'+str(input_size)+'.ckpt')

        self.mult         = float(mult)
        self.num_classes  = num_classes
        self.dropout_prob = dropout

    def forward(self, data, is_training=True):
        """ """
        arg_scope = mobilenet_v1_arg_scope(is_training)
        with slim.arg_scope(arg_scope):
            logits, end_points = mobilenet_v1(data,
                                              num_classes=self.num_classes, 
                                              dropout_keep_prob=self.dropout_prob,
                                              depth_multiplier=self.mult, 
                                              is_training=is_training,
                                              conv_defs=self.net_cfg, 
                                              global_pool=True)
        probs = tf.identity(end_points['Predictions'], name='output')
        return logits, probs

    def weight_init(self, tf_sess):
        if self.chkpt is not None:
            # Exclude classifier part from the loaded weights
            var_list = []
            for var in slim.get_model_variables('MobilenetV1'):
                if not var.op.name.startswith('MobilenetV1/Logits'):
                    var_list.append(var)
                else:
                    print ('EXCLUDE <', var.op.name, '>')

            # Initialize the model to the pretrained weights
            load_trained_weights = slim.assign_from_checkpoint_fn(self.chkpt, var_list)
            load_trained_weights(tf_sess)