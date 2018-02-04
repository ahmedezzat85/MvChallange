import os
from importlib import import_module

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from mvnc_dev import MvNCS

_CUR_DIR           = os.path.dirname(__file__)
_DATASET_ROOT_DIR  = os.path.join(_CUR_DIR, '..', 'dataset')
_DATA_DIR          = os.path.join(_DATASET_ROOT_DIR, 'training')
_CSV_TEST_SET_FILE = os.path.join(_DATASET_ROOT_DIR, 'eval_set.csv')
_MODEL_BIN_DIR     = os.path.join(_CUR_DIR, '..', 'bin')

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RGB_MEAN = np.array([_R_MEAN, _G_MEAN, _B_MEAN]) / 255

class Inference(object):
    """
    """
    def __init__(self, num_classes, input_size=224, preserve_aspect=True):
        self.img_size        = input_size
        self.num_classes     = num_classes
        self.preserve_aspect = preserve_aspect
        self.resize_side     = 256

    def _resize_keep_aspect(self, image):
        """ """
        h, w, _ = image.shape
        scale = self.resize_side / min(h, w)
        h = int(h * scale)
        w = int(w * scale)
        image = cv2.resize(image, (w, h))
        hs = (h - self.img_size) // 2
        ws = (w - self.img_size) // 2
        image = image[hs:hs+self.img_size, ws:ws+self.img_size]
        return image

    def preprocess(self, image):
        """ """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255

        if self.preserve_aspect is True:
            image = self._resize_keep_aspect(image)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))

        image = image - _RGB_MEAN
        return np.expand_dims(image, 0)

    def run(self):
        df = pd.read_csv(_CSV_TEST_SET_FILE, sep=',')
        images = df['IMAGE_NAME']
        labels = df['CLASS_INDEX']

        top1_acc = 0
        top5_acc = 0
        for i, (image, label) in enumerate(zip(images, labels)):
            # Read and decode image
            image_path = os.path.join(_DATA_DIR, image)
            im = cv2.imread(image_path)
            # Perform Preprocessing
            im = self.preprocess(im)
            # Run the classification model
            probs = self.__call__(im)
            # Get top-5 predicted classes
            top5  = np.argsort(-probs)[:5]
            top5_classes = top5 + 1 # Labels are stored starting from 1
            top5_probs   = probs[top5]
            # Compute the accuracy
            if label == top5_classes[0]:
                top1_acc += 1
                top5_acc += 1
            else:
                if label in top5_classes: top5_acc += 1
            
            if i % 100 == 0:
                print('(',i,')', (top1_acc * 100.0) / (i+1), (top5_acc * 100.0) / (i+1))

        print ('TOP-1 Accuracy = ', top1_acc, top1_acc * 100 / len(labels))
        print ('TOP-5 Accuracy = ', top5_acc, top5_acc * 100 / len(labels))

        

class MvNCSInference(Inference):
    def __init__(self, model='compiled.graph', num_classes=200, input_size=224, preserve_aspect=True):
        super(MvNCSInference, self).__init__(num_classes, input_size, preserve_aspect)
        self.mvncs = MvNCS()
        self.mvncs.load_model(model)

    def __call__(self, image):
        return self.mvncs.forward(image)

    def close(self):
        self.mvncs.close()

class TensorflowInference(Inference):
    """
    """
    def __init__(self, model, num_classes=200, input_size=224, preserve_aspect=True):
        super(TensorflowInference, self).__init__(num_classes, input_size, preserve_aspect)
        # Disable Tensorflow logs except for errors
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        net_module   = import_module('model.' + model)
        self.forward = net_module.snpx_net_create

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.input = tf.placeholder(tf.float32, [1, 224, 224, 3], name='input')
            _, self.output = self.forward(200, self.input, 'NHWC', is_training=False)

            saver = tf.train.Saver()
            self.tf_sess = tf.Session()

            # Load the saved model from a checkpoint
            chkpt = os.path.join(_MODEL_BIN_DIR, model, model)
            saver.restore(self.tf_sess, chkpt)

    def __call__(self, image):
        probs = self.tf_sess.run(self.output, {self.input: image})[0]
        return probs
        
    def close(self):
        self.tf_sess.close()

if __name__ == '__main__':
    infer = TensorflowInference('mobilenet')
    infer.run()
    infer.close()
