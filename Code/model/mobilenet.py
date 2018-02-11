from __future__ import absolute_import

import tensorflow as tf
from . tf_net import TFNet
from . mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
import tensorflow.contrib.slim as slim

def clean_model(input_data, num_classes, is_training):
    arg_scope = mobilenet_v1_arg_scope(is_training)
    with slim.arg_scope(arg_scope):
        logits, end_points = mobilenet_v1(input_data, num_classes=num_classes, dropout_keep_prob=0.8, 
                                is_training=is_training, global_pool=True)
    probs = tf.identity(end_points['Predictions'], name='output')
    return logits, probs

def pretrained_model(input_data, num_classes, is_training, chkpt_file):
    arg_scope = mobilenet_v1_arg_scope(is_training)
    with slim.arg_scope(arg_scope):
        logits, end_points = mobilenet_v1(input_data, num_classes=num_classes, dropout_keep_prob=0.8, 
                                is_training=is_training, global_pool=True)
    probs = tf.identity(end_points['Predictions'], name='output')
    init_fn = slim.assign_from_checkpoint_fn(chkpt_file, slim.get_model_variables('MobilenetV1'))

# class MobileNet(TFNet):
#     """
#     """
#     def __init__(self, data, data_format, num_classes, is_train=True):
#         dtype = data.dtype.base_dtype
#         super(MobileNet, self).__init__(dtype, data_format, train=is_train)
#         self.dw_alpa = 0.75
#         self.net_out = data
#         self.strip_dropout = (not is_train) if isinstance(is_train, bool) else False
#         self.num_classes = num_classes

#     def __call__(self, alpha=1.0):
#         """ """
#         net_out = self.net_out
#         net_out = self.convolution(net_out, 32, (3,3), stride=(2,2), act_fn='relu6', add_bn=True, name='Conv0')
#         net_out = self.conv_dw(net_out, 64  , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv1')
#         net_out = self.conv_dw(net_out, 128 , (3,3), stride=(2,2), act_fn='relu6', add_bn=True, name='Conv2')
#         net_out = self.conv_dw(net_out, 128 , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv3')
#         net_out = self.conv_dw(net_out, 256 , (3,3), stride=(2,2), act_fn='relu6', add_bn=True, name='Conv4')
#         net_out = self.conv_dw(net_out, 256 , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv5')
#         net_out = self.conv_dw(net_out, 512 , (3,3), stride=(2,2), act_fn='relu6', add_bn=True, name='Conv6')
#         net_out = self.conv_dw(net_out, 512 , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv7')
#         net_out = self.conv_dw(net_out, 512 , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv8')
#         net_out = self.conv_dw(net_out, 512 , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv9')
#         net_out = self.conv_dw(net_out, 512 , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv10')
#         net_out = self.conv_dw(net_out, 512 , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv11')
#         net_out = self.conv_dw(net_out, 1024, (3,3), stride=(2,2), act_fn='relu6', add_bn=True, name='Conv12')
#         net_out = self.conv_dw(net_out, 1024, (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv13')
#         net_out = self.global_pool(net_out, 'avg', name="global_pool")
#         if self.strip_dropout is False:
#             net_out = self.dropout(net_out, 0.5)
#         net_out = self.Softmax(net_out, self.num_classes, fc=False)
#         return net_out

def net_create(num_classes, input_data, data_format="NHWC", is_training=True):
    """ """
    logits, probs = clean_model(input_data, num_classes, is_training)
    return logits, probs