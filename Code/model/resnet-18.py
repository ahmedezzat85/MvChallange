import tensorflow as tf
from . tf_net import TFNet

class TFModel(TFNet):
    """
    """
    def __init__(self, dtype, data_format, num_classes):
        super(Resnet, self).__init__(dtype, data_format)
        self.num_classes = num_classes

    def _resnet_block(self, filters, kernel, stride=(1,1), act_fn='relu', conv_1x1=0, name=None):
        """ """
        data = self.net_out
        shortcut  = data
        if conv_1x1:
            shortcut = self.convolution(data, filters, (1,1), stride, pad='same', act_fn='',
                                        add_bn=True, name=name+'_1x1_conv')
        
        net_out = self.convolution(data, filters, kernel, stride, act_fn=act_fn, add_bn=True,
                                    name=name+'_conv1')
        net_out = self.convolution(net_out, filters, kernel, (1,1), act_fn='', add_bn=True, 
                                    name=name+'_conv2')

        net_out = net_out + shortcut
        self.net_out = tf.nn.relu(net_out, name=name+'_Relu')

    def _resnet_unit(self, num_blocks, filters, kernel, stride=1, act_fn='relu', name=None):
        """ """
        strides = (stride, stride)
        self._resnet_block(filters, kernel, strides, act_fn, conv_1x1=1, name=name+'_block0')
        for i in range(1, num_blocks):
            self._resnet_block(filters, kernel, (1,1), act_fn, name=name+'_block'+str(i))

    def forward(self, data, is_training=True):
        """ """
        self.train = is_training
        net_out = data
        net_out = self.convolution(net_out, 64, (7,7), (2,2), act_fn='relu', add_bn=True, name='Conv0')
        self.net_out = self.pooling(net_out, 'max', (2,2), name='Pool1')

        blocks  = [ 2,   2,   2,   2]
        filters = [64, 128, 256, 512]
        strides = [ 1,   2,   2,   2]
        for k in range(len(blocks)):
            self._resnet_unit(blocks[k], filters[k], (3,3), strides[k], name='stage'+str(k))

        net_out = self.global_pool(self.net_out, name='global_pool')
        net_out = self.dropout(net_out, 0.5)
        net_out = self.flatten(net_out)
        net_out = self.Softmax(net_out, self.num_classes)
        return net_out

    def weight_init(self, tf_sess):
        pass