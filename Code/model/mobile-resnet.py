from __future__ import absolute_import

import tensorflow as tf
from . tf_net import TFNet

class MobileResNet(TFNet):
    """
    """
    def __init__(self, data, data_format, num_classes, is_train=True):
        dtype = data.dtype.base_dtype
        super(MobileNet, self).__init__(dtype, data_format, train=is_train)
        self.net_out = tf.identity(data, name='data')
        self.num_classes = num_classes

    def _resnet_block(self, filters, kernel, stride=(1,1), act_fn='relu', conv_1x1=0, name=None):
        """ """
        data = self.net_out
        shortcut  = data
        if conv_1x1:
            shortcut = self.convolution(data, filters, (1,1), stride, pad='same', act_fn='',
                                        add_bn=True, name=name+'_1x1_conv')
        
        net_out = self.conv_dw(data   , filters, kernel, stride      , act_fn=act_fn, add_bn=True, name=name+'_Conv1')
        net_out = self.conv_dw(net_out, filters, kernel, stride=(1,1), act_fn=''    , add_bn=True, name=name+'_Conv2')

        net_out = net_out + shortcut
        self.net_out = tf.nn.relu(net_out, name=name+'_Relu')

    def __call__(self, alpha=1.0):
        """ """
        net_out = self.net_out
        net_out = self.convolution(net_out, 32, (3,3), stride=(2,2), act_fn='relu', add_bn=True, name='Conv0')
        net_out = self.conv_dw(net_out, 64  , (3,3), stride=(1,1), act_fn='relu', add_bn=True, name='Conv1')
        self._resnet_block(128, (3,3), stride=(2,2), act_fn='relu', conv_1x1=1, name='Block1')
        self._resnet_block(256, (3,3), stride=(2,2), act_fn='relu', conv_1x1=1, name='Block2')
        self._resnet_block(512, (3,3), stride=(2,2), act_fn='relu', conv_1x1=1, name='Block3')
        self._resnet_block(512, (3,3), stride=(1,1), act_fn='relu', conv_1x1=1, name='Block4')
        self._resnet_block(512, (3,3), stride=(1,1), act_fn='relu', conv_1x1=1, name='Block5')
        net_out = self.conv_dw(net_out, 1024, (3,3), stride=(2,2), act_fn='relu', add_bn=True, name='Conv12')
        net_out = self.conv_dw(net_out, 1024, (3,3), stride=(2,2), act_fn='relu', add_bn=True, name='Conv13')
        net_out = self.global_pool(net_out, 'avg', name="global_pool")
        net_out = self.flatten(net_out)
        net_out = self.Softmax(net_out, self.num_classes)
        return net_out

def snpx_net_create(num_classes, input_data, data_format="NHWC", is_training=True):
    """ """
    net = MobileResNet(input_data, data_format, num_classes, is_training)
    net_out = net()
    return net_out