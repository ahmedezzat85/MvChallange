import os
import cv2
import argparse
import numpy as np
from time import time
import tensorflow as tf
from importlib import import_module
from base_inference import BaseInference

_CUR_DIR           = os.path.dirname(__file__)
_MODEL_BIN_DIR     = os.path.join(_CUR_DIR, '..', 'bin')

class TensorflowInference(BaseInference):
    """
    """
    def __init__(self, 
                 model,
                 dataset_key='eval',
                 inference_file='inferences.csv',
                 score_inference=True,
                 input_size=224,
                 preserve_aspect=True):
        super(TensorflowInference, self).__init__(dataset_key, inference_file, not(dataset_key=='prov'), 
                                                    input_size, preserve_aspect)

        # Disable Tensorflow logs except for errors
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        self._load_model(model)

    def _load_model(self, model):
        net_module = import_module('model.' + model)
        self.model = net_module.TFModel(tf.float32, "NHWC", 200)
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.input = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, 3], name='input')
            _, self.output = self.model.forward(self.input, is_training=False)

            saver = tf.train.Saver()
            self.tf_sess = tf.Session()

            # Load the saved model from a checkpoint
            chkpt = os.path.join(_MODEL_BIN_DIR, model, 'network')
            saver.restore(self.tf_sess, chkpt)

    def forward(self, image):
        t_start = time()
        probs = self.tf_sess.run(self.output, {self.input: image})[0]
        fw_time = (time() - t_start) * 1000
        return probs, fw_time
        
    def close(self):
        self.tf_sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MvNCS Inference Script')
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    args = parser.parse_args()

    infer = TensorflowInference(model=args.model, dataset_key=args.dataset)
    infer()
    infer.close()
