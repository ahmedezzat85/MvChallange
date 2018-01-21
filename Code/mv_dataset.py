# """ Load datasets in memory

# """
from __future__ import absolute_import

import os
import argparse
import threading
from datetime import datetime

import numpy as np 
import tensorflow as tf
from pandas import read_csv

import utils

##=======##=======##=======##
# CONSTANTS
##=======##=======##=======##
_DATASET_SIZE    = 80000
_NUM_CLASSES     = 200
_IMG_PER_CLASS   = _DATASET_SIZE // _NUM_CLASSES
_TRAIN_SET_SIZE  = 75000
_VAL_SET_SIZE    = 5000
_TRAIN_PER_CLASS = _IMG_PER_CLASS * (_TRAIN_SET_SIZE / _DATASET_SIZE)
_VAL_PER_CLASS   = _IMG_PER_CLASS * (_VAL_SET_SIZE / _DATASET_SIZE)

_DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

_IMAGE_TFREC_STRUCTURE = {
        'image' : tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width' : tf.FixedLenFeature([], tf.int64)
    }


##=======##=======##=======##=======##=======##=======##=======##
# Dataset Writer Functions (Save dataset to TFRECORD files)
##=======##=======##=======##=======##=======##=======##=======##
def _int64_feature(value):
    val = value if isinstance(value, list) else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class JpegDecoder(object):
    """ Decode JPG image from file.
    """
    def __init__(self):
        self.tf_sess = tf.Session()
        self._jpeg_img = tf.placeholder(dtype=tf.string)
        self._jpeg_dec = tf.image.decode_jpeg(self._jpeg_img, channels=3)

    def __call__(self, jpeg_image_file):
        try:
            with tf.gfile.FastGFile(jpeg_image_file, mode='rb') as fp:
                jpeg_image = fp.read()
            
            image = self.tf_sess.run(self._jpeg_dec, feed_dict={self._jpeg_img: jpeg_image})
            h, w, c = image.shape
            return jpeg_image, h, w, c
        except:
            print ('EXCEPTION --> ', jpeg_image_file)
    
    def close(self):
        self.tf_sess.close()

class TFRecFile(object):
    """ """
    def __init__(self, tf_rec_out_file, jpeg_decoder):
        self.writer   = tf.python_io.TFRecordWriter(tf_rec_out_file)
        self.jpeg_dec = jpeg_decoder

    def add_image(self, image_file, label):
        image, h, w, c = self.jpeg_dec(image_file)
        features = tf.train.Features(feature={
            'image' : _bytes_feature(image),
            'label' : _int64_feature(label),    
            'height': _int64_feature(h),
            'width' : _int64_feature(w)
        })
        tf_rec_proto = tf.train.Example(features=features)
        self.writer.write(tf_rec_proto.SerializeToString())

    def close(self):
        self.writer.close()

class TFDatasetWriter(object):
    """
    """
    def __init__(self, num_rec_files=5):
        # TFRecord Size 
        rec_size = _TRAIN_SET_SIZE // num_rec_files
        if (rec_size * num_rec_files) != _TRAIN_SET_SIZE: 
            raise ValueError('num_train_rec not suitable')
        self.num_rec = num_rec_files
        self.jpg_dec = JpegDecoder()

    def _split_train_eval(self):
        """ """
        index_file = os.path.join(_DATASET_DIR, 'training_ground_truth.csv')
        data = read_csv(index_file, sep=',')
        image_files = data['IMAGE_NAME']
        labels      = data['CLASS_INDEX']

        # Random Shuffle with repeatable pattern
        index = list(range(_DATASET_SIZE))
        np.random.seed(12345)
        np.random.shuffle(index)

        # Split training set into train/val
        self.train_set = []
        self.eval_set  = []
        counters  = np.zeros([_NUM_CLASSES, 1])
        for i in index:
            label = labels[i]
            file  = image_files[i]
            if counters[label - 1] < _TRAIN_PER_CLASS:
                self.train_set.append((file, label))
                counters[label - 1] += 1
            else:
                self.eval_set.append((file, label))

    def write(self):
        def create_tf_record(rec_file, image_list):
            rec_file = TFRecFile(os.path.join(_DATASET_DIR, rec_file), self.jpg_dec)
            for t in image_list:
                image, label = t
                rec_file.add_image(os.path.join(_DATASET_DIR, 'training', image), label)
            rec_file.close()

        start_time = datetime.now()
        t_start    = start_time
        self._split_train_eval()
        train_set = utils.list_split(self.train_set, self.num_rec)

        coord = tf.train.Coordinator()
        threads = []
        for rec_id in range(self.num_rec):
            args = ('train_0' + str(rec_id+1) + '.tfrecords', train_set[rec_id])
            th = threading.Thread(target=create_tf_record, args=args)
            th.start()
            threads.append(th)
        args = ('val.tfrecords', val_set)
        th = threading.Thread(target=create_tf_record, args=args)
        th.start()
        threads.append(th)
        coord.join(threads)
        self.jpg_dec.close()
        print ('ELAPSED TIME:  ', datetime.now() - t_start)
            


##=======##=======##=======##=======##=======##=======##=======##
# Dataset Reader Functions (Load dataset from TFRECORD files)
##=======##=======##=======##=======##=======##=======##=======##
class TFDatasetReader(object):
    """ 
    """
    def __init__(self, dtype=tf.float32, image_size=224, shuffle_buff_sz=5000):

        self.dtype       = dtype
        self.shape       = (image_size, image_size, 3)
        self.num_classes = 200
        train_file_name  = os.path.join(_DATASET_DIR, 'train_{:02d}.tfrecords')
        self.train_files = [train_file_name.format(i+1) for i in range(5)]
        self.val_file    = os.path.join(_DATASET_DIR, 'val.tfrecords')
        self.shuffle_sz  = shuffle_buff_sz

    def _parse_eval_rec(self, tf_record):
        """ """
        feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
        image = tf.image.decode_jpeg(feature['image'], channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, self.shape[0], self.shape[1])
        image = tf.cast(image, self.dtype)
        label = tf.cast(feature['label'], tf.int64)
        return image, label

    def _parse_train_rec(self, tf_record):
        """ """
        feature = tf.parse_single_example(tf_record, features=_IMAGE_TFREC_STRUCTURE)
        image = tf.image.decode_jpeg(feature['image'], channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, self.shape[0], self.shape[1])
        image = tf.cast(image, self.dtype)
        label = tf.cast(feature['label'], tf.int64)
        return image, label

    def read(self, batch_size, for_training=True, data_format='NCHW', data_aug=False):
        """ """
        eval_dataset  = tf.data.TFRecordDataset(self.val_file)
        eval_dataset  = eval_dataset.map(lambda tf_rec: self._parse_eval_rec(tf_rec), 5)
        eval_dataset  = eval_dataset.prefetch(batch_size)
        eval_dataset  = eval_dataset.batch(batch_size)
        out_types = eval_dataset.output_types
        out_shapes = eval_dataset.output_shapes
        data_iter  = tf.data.Iterator.from_structure(out_types, out_shapes)
        self.eval_init_op = data_iter.make_initializer(eval_dataset)

        if for_training is True:
            train_dataset = tf.data.Dataset.from_tensor_slices(self.train_files)
            train_dataset = train_dataset.flat_map(tf.data.TFRecordDataset)
            train_dataset = train_dataset.shuffle(len(self.train_files))
            train_dataset = train_dataset.map(lambda tf_rec: self._parse_train_rec(tf_rec), 10)
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

    args = parser.parse_args()

    # Disable Tensorflow logs except for errors
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if args.writer is True:
        print ('Dataset Writer ...')
        writer = TFDatasetWriter(args.num_files)
        writer.write()
    elif args.reader is True:
        print ('Reader Test ....')
        reader = TFDatasetReader(image_size=192, shuffle_buff_sz=2000)
        reader.read(128, True, 'NCHW', True)
        with tf.Session() as sess:
            t_start = datetime.now()
            sess.run(reader.train_init_op)
            images, labels = sess.run([reader.images, reader.labels])
            print ('Time for training set: ', datetime.now() - t_start)
            t_start = datetime.now()
            
            print (images.shape)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()