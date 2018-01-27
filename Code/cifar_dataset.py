# """ Load datasets in memory

# """
from __future__ import absolute_import

import os
import sys
import argparse
import threading
from datetime import datetime
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np 
import tensorflow as tf
from pandas import read_csv

import utils
from vgg_preprocessing import preprocess_image

##=======##=======##=======##
# CONSTANTS
##=======##=======##=======##
_DATASET_SIZE    = 50000
_NUM_CLASSES     = 10
_IMG_PER_CLASS   = _DATASET_SIZE // _NUM_CLASSES
_TRAIN_SET_SIZE  = 50000
_VAL_SET_SIZE    = 10000

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'CIFAR-10')

_IMAGE_TFREC_STRUCTURE = {
        'image' : tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([], tf.int64)
    }

##=======##=======##=======##=======##=======##=======##=======##
# Dataset Writer Functions (Save dataset to TFRECORD files)
##=======##=======##=======##=======##=======##=======##=======##
def _int64_feature(value):
    val = value if isinstance(value, list) else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class TFRecFile(object):
    """ """
    def __init__(self, tf_rec_out_file):
        self.writer   = tf.python_io.TFRecordWriter(tf_rec_out_file)

    def add_image(self, image, label):
        image = image.tostring()
        features = tf.train.Features(feature={
            'image' : _bytes_feature(image),
            'label' : _int64_feature(label)
        })
        tf_rec_proto = tf.train.Example(features=features)
        self.writer.write(tf_rec_proto.SerializeToString())

    def close(self):
        self.writer.close()

def _load_CIFAR_batch(batch_file):
    """ Read a CIFAR-10 batch file into numpy arrays """
    with open(os.path.join(DATASET_DIR, 'raw', batch_file), 'rb') as f:
        if sys.version_info.major == 3:
            datadict = pickle.load(f, encoding='bytes')
        else:
            datadict = pickle.load(f)
        images = datadict[b'data']
        labels = datadict[b'labels']
        
        labels = np.array(labels)
        images = np.reshape(images,[-1, 3, 32, 32])
        images = images.swapaxes(1,3)
        return images, labels

def get_CIFAR10_data():
    """   """
    for b in range(1,6):
        f = 'data_batch_' + str(b)
        xb, yb = _load_CIFAR_batch(f)
        if b > 1:
            x_train = np.concatenate((x_train, xb))
            y_train = np.concatenate((y_train, yb))
            del xb, yb
        else:
            x_train = xb
            y_train = yb

    x_test, y_test = _load_CIFAR_batch('test_batch')
    train_set = (x_train, y_train)
    eval_set  = (x_test, y_test)
    return train_set, eval_set


class TFDatasetWriter(object):
    """
    """
    def __init__(self):
        pass

    def write(self):
        def create_tf_record(rec_file, dataset):
            rec_file = TFRecFile(os.path.join(DATASET_DIR, rec_file))
            images, labels = dataset
            for i in range(len(images)):
                rec_file.add_image(images[i], labels[i])
            rec_file.close()

        start_time = datetime.now()
        t_start    = start_time
        train_set, eval_set = get_CIFAR10_data()
        create_tf_record('train.tfrecords', train_set)
        create_tf_record('eval.tfrecords', eval_set)
        print ('ELAPSED TIME:  ', datetime.now() - t_start)
            


##=======##=======##=======##=======##=======##=======##=======##
# Dataset Reader Functions (Load dataset from TFRECORD files)
##=======##=======##=======##=======##=======##=======##=======##
class TFDatasetReader(object):
    """ 
    """
    def __init__(self, image_size=32, shuffle_buff_sz=5000):

        self.name        = 'CIFAR-10'
        self.dataset_sz  = _TRAIN_SET_SIZE
        self.shape       = (image_size, image_size, 3)
        self.num_classes = _NUM_CLASSES
        self.scale_min   = image_size + 32
        self.scale_max   = self.scale_min
        self.train_file  = os.path.join(DATASET_DIR, 'train.tfrecords')
        self.eval_file   = os.path.join(DATASET_DIR, 'eval.tfrecords')
        self.shuffle_sz  = shuffle_buff_sz

    def _parse_eval_rec(self, tf_record, dtype):
        """ """
        feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
        image = tf.decode_raw(feature['image'], tf.uint8)
        image = tf.reshape(image, (32, 32, 3))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.cast(image, dtype)
        label = tf.cast(feature['label'], tf.int64)
        return image, label

    def _parse_train_rec(self, tf_record, dtype):
        """ """
        feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
        image = tf.decode_raw(feature['image'], tf.uint8)
        image = tf.reshape(image, (32, 32, 3))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.cast(image, dtype)
        label = tf.cast(feature['label'], tf.int64)
        return image, label

    def read(self, batch_size, for_training=True, data_format='NCHW', data_aug=False, dtype=tf.float32):
        """ """
        self.dtype = dtype

        eval_dataset  = tf.data.TFRecordDataset(self.eval_file)
        eval_dataset  = eval_dataset.map(lambda tf_rec: self._parse_eval_rec(tf_rec, dtype))
        eval_dataset  = eval_dataset.prefetch(batch_size)
        eval_dataset  = eval_dataset.batch(batch_size)
        out_types = eval_dataset.output_types
        out_shapes = eval_dataset.output_shapes
        data_iter  = tf.data.Iterator.from_structure(out_types, out_shapes)
        self.eval_init_op = data_iter.make_initializer(eval_dataset)

        if for_training is True:
            train_dataset = tf.data.TFRecordDataset(self.train_file)
            train_dataset = train_dataset.map(lambda tf_rec: self._parse_train_rec(tf_rec, dtype), 4)
            train_dataset = train_dataset.prefetch(batch_size)
            train_dataset = train_dataset.shuffle(self.shuffle_sz)
            train_dataset = train_dataset.batch(batch_size)
            self.train_init_op = data_iter.make_initializer(train_dataset)
            
        self.images, labels = data_iter.get_next()
        self.labels  = tf.one_hot(labels, self.num_classes)
        if data_format.startswith('NC'):
            self.images = tf.transpose(self.images, [0, 3, 1, 2])

def main():
    """ """
    parser = argparse.ArgumentParser('IntelMovidius Dataset Processing Module')
    parser.add_argument('-w', '--writer', action='store_true', help='Dataset Writer Mode. Save the raw image\
                                                                    dataset into tfrecord files')
    parser.add_argument('-r', '--reader', action='store_true', help='Dataset Reader Mode. Parse tfreord files')
    parser.add_argument('-n', '--num-files', type=int, default=5, help='Number of TFRECORD files for training dataset')
    parser.add_argument('-j', '--jpg', action='store_true', help='JPEG Test.')

    args = parser.parse_args()

    # Disable Tensorflow logs except for errors
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.writer is True:
        print ('Dataset Writer ...')
        writer = TFDatasetWriter()
        writer.write()
    elif args.reader is True:
        print ('Reader Test ....')
        reader = TFDatasetReader(image_size=192, shuffle_buff_sz=2000)
        reader.read(200, True, 'NCHW')
        with tf.Session() as sess:
            t_start = datetime.now()
            sess.run(reader.train_init_op)
            i = 0
            while True:
                try:
                    images, labels = sess.run([reader.images, reader.labels])
                    i += 1
                except tf.errors.OutOfRangeError:
                    break
        print ('batch ', i)
        print ('Time for training set: ', datetime.now() - t_start)
        t_start = datetime.now()
            
    else:
        parser.print_help()

if __name__ == '__main__':
    main()