import tensorflow as tf
from . tf_net import TFNet

class CNN(TFNet):
    """
    """
    def __init__(self, data, data_format, num_classes, is_train=True):
        dtype = data.dtype.base_dtype
        super(CNN, self).__init__(dtype, data_format, train=is_train)
        self.net_out = data
        self.num_classes = num_classes

    def __call__(self, blocks=[], filters=[16, 32, 64], strides=[1,2,2]):
        """ """
        net_out = self.convolution(self.net_out, 32, (7,7), (2,2), act_fn='relu', add_bn=True, name='Conv0')
        net_out = self.conv_dw(net_out, 32 , (3,3), stride=(2,2), act_fn='relu6', add_bn=True, name='Conv_dw_1')
        net_out = self.conv_dw(net_out, 64 , (3,3), stride=(1,1), act_fn='relu6', add_bn=True, name='Conv_dw_2')
        net_out = self.convolution(net_out, 64, (3,3), (1,1), act_fn='relu', add_bn=True, name='Conv1')
        net_out = self.pooling(net_out, 'max', (2,2), name='Pool2')
        net_out = self.convolution(net_out, 128, (3,3), (1,1), act_fn='relu', add_bn=True, name='Conv2')
        net_out = self.pooling(net_out, 'max', (2,2), name='Pool3')
        net_out = self.global_pool(self.net_out, name='global_pool')
        net_out = self.dropout(net_out, 0.5)
        net_out = self.flatten(net_out)
        net_out = self.Softmax(net_out, self.num_classes)
        return net_out

def snpx_net_create(num_classes, input_data, data_format="NHWC", is_training=True):
    """ """
    net = CNN(input_data, data_format, num_classes, is_training)
    net_out = net()
    return net_out
