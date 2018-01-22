from __future__ import absolute_import

import os
import sys
from time import time
from datetime import datetime
from importlib import import_module

import numpy as np
import tensorflow as tf

import utils

class TFClassifier(object):
    """ Abtraction of TensorFlow functionality.
    """
    def __init__(self, 
                 model_name, 
                 dataset,
                 data_format='NHWC',
                 data_aug=False, 
                 logs_dir=None,
                 bin_dir=None):

        # Disable Tensorflow logs except for errors
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Parameter initializations
        self.logger       = None
        self.dataset      = dataset
        self.model_fn     = None
        self.model_name   = model_name
        self.base_tick    = time()
        self.dtype        = tf.float32
        self.tf_sess      = None
        self.eval_op      = None
        self.loss         = None
        self.train_op     = None
        self.summary_op   = None
        self.data_format  = data_format
        self.global_step  = None
        self.data_aug     = data_aug
        self.log_dir      = logs_dir

        # Get the neural network model function
        net_module    = import_module('model.' + model_name)
        self.model_fn = net_module.snpx_net_create

        # # Directory for Model and deployment
        # utils.create_dir(bin_dir)
        # self.model_prfx = os.path.join(self.bin_dir, model_name)

    def tick(self):
        return time() - self.base_tick
    
    def _load_dataset(self, training=True):
        """ """
        with tf.device('/cpu:0'):
            self.dataset.read(self.batch_size, training, self.data_format, self.data_aug)

    def _forward_prop(self, batch, num_classes, training=True):
        """ """
        logits, predictions = self.model_fn(num_classes, batch, self.data_format, is_training=training)
        return logits, predictions

    def _create_train_op(self, logits):
        """ """
        self.global_step = tf.train.get_or_create_global_step()

        # Get the optimizer
        if self.hp.lr_decay:
            lr = tf.train.exponential_decay(self.hp.lr, self.global_step, self.hp.lr_decay, 
                                                self.hp.lr_step, True)
        else:
            lr = self.hp.lr
        tf.summary.scalar("Learning Rate", lr)

        optmz = self.hp.optimizer.lower()
        if optmz == 'sgd':
            opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
        elif optmz == 'adam':
            opt = tf.train.AdamOptimizer(lr)
        elif optmz == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(lr)
 
        # Compute the loss and the train_op
        cross_entropy = tf.losses.softmax_cross_entropy(self.dataset.labels, logits) # needs wrapping
        self.loss = cross_entropy
        if self.hp.wd > 0:
            l2_loss = self.hp.wd * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() 
                                                if 'batch_normalization' not in v.name])
            self.loss = self.loss + l2_loss
        tf.summary.scalar("Cross Entropy", cross_entropy)

        update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.minimize(self.loss, self.global_step)

        self._create_eval_op(logits, self.dataset.labels)

    def _create_eval_op(self, predictions, labels):
        """ """
        acc_tensor   = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
        self.eval_op = tf.reduce_mean(tf.cast(acc_tensor, tf.float32))

    def _train_loop(self):
        """ """
        # Initialize the training Dataset Iterator
        self.tf_sess.run(self.dataset.train_init_op)

        epoch_start_time = self.tick()
        last_log_tick    = epoch_start_time
        last_step        = self.tf_sess.run(self.global_step)
        while True:
            try:
                feed_dict = {self.training: True}
                fetches   = [self.loss, self.train_op, self.summary_op, self.global_step]
                loss, _, s, step = self.tf_sess.run(fetches, feed_dict)
                self.tb_writer.add_summary(s, step)
                self.tb_writer.flush()
                elapsed = self.tick() - last_log_tick
                if elapsed >= self.log_freq:
                    speed = ((step - last_step)  * self.batch_size) / elapsed
                    last_step = step
                    last_log_tick  = self.tick()
                    self.logger.info('(%.3f)Epoch[%d] Batch[%d]\tloss: %.3f\tspeed: %.3f samples/sec', 
                                      self.tick(), self.epoch, step, loss, speed)
            except tf.errors.OutOfRangeError:
                break
        self.logger.info('Epoch Training Time = %.3f', self.tick() - epoch_start_time)
        self.saver.save(self.tf_sess, self.chkpt_prfx, self.epoch)

    def _eval_loop(self):
        """ """
        self.tf_sess.run(self.dataset.eval_init_op)
        epoch_start_time = self.tick()
        val_acc = 0
        n = 0
        while True:
            try:
                feed_dict = {self.training: False}
                batch_acc = self.tf_sess.run(self.eval_op, feed_dict)
                val_acc += batch_acc
                n += 1
            except tf.errors.OutOfRangeError:
                break
        eval_acc = (val_acc * 100.0) / n
        self.logger.info('Validation Time = %.3f', self.tick() - epoch_start_time)
        return eval_acc

    def create_tf_session(self):
        """ """
        # Session Configurations 
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 4
        config.gpu_options.allow_growth = True # Very important to avoid OOM errors
        config.gpu_options.per_process_gpu_memory_fraction = 1.0 #0.4

        # Create and initialize a TF Session
        self.tf_sess = tf.Session(config=config)
        self.tf_sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    def train(self, num_epoch, begin_epoch, hp, log_freq_sec=1):
        """ """
        t_start = datetime.now()
        self.hp = hp
        self.log_freq   = log_freq_sec
        self.batch_size = hp.batch_size

        utils.create_dir(os.path.join(self.log_dir, 'chkpt'))        
        self.chkpt_prfx = os.path.join(self.log_dir, 'chkpt', 'CHKPT')
        self.logger = utils.create_logger(self.model_name, os.path.join(self.log_dir, 'Train.log'))
        self.logger.info("Training Started at  : " + t_start.strftime("%Y-%m-%d %H:%M:%S"))

        with tf.Graph().as_default():
            self._load_dataset()

            # Forward Propagation
            self.training = tf.placeholder(tf.bool, name='Train_Flag')
            logits, _ = self._forward_prop(self.dataset.images, self.dataset.num_classes, self.training)
        
            self._create_train_op(logits)

            # Create a TF Session
            self.create_tf_session()

            # Create Tensorboard stuff
            self.summary_op = tf.summary.merge_all()
            self.tb_writer  = tf.summary.FileWriter(self.log_dir, graph=self.tf_sess.graph)

            if begin_epoch > 0:
                # Load the saved model from a checkpoint
                chkpt = self.chkpt_prfx + '-' + str(begin_epoch)
                self.logger.info("Loading Checkpoint " + chkpt)
                begin_epoch += 1
                num_epoch   += begin_epoch
                self.saver = tf.train.Saver(max_to_keep=50)
                self.saver.restore(self.tf_sess, chkpt)
                self.tb_writer.reopen()
            else:
                self.saver = tf.train.Saver(max_to_keep=200)

            # Training Loop
            for self.epoch in range(begin_epoch, num_epoch):
                # Training
                self._train_loop()
                # Validation
                val_acc = self._eval_loop()
                # Visualize Training
                acc_summ = tf.summary.Summary()
                summ_val = acc_summ.value.add(simple_value=val_acc, tag="Validation-Accuracy")
                self.tb_writer.add_summary(acc_summ, self.epoch)
                self.logger.info('Epoch[%d] Validation-Accuracy = %.2f%%', self.epoch, val_acc)
                # Flush Tensorboard Writer
                self.tb_writer.flush()

            # Close and terminate
            self.tb_writer.close()
            self.tf_sess.close()

        self.logger.info("Training Finished at : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Total Training Time  : " + str(datetime.now() - t_start))

    def evaluate_model(self):
        """ """
        with tf.Graph().as_default():
            # Load the Evaluation Dataset
            self._load_dataset(training=False)

            # Forward Prop
            self.training = tf.placeholder(tf.bool, name='Train_Flag')
            predictions, _ = self._forward_prop(self.dataset.images, self.dataset.num_classes, False)

            # Create a TF Session
            self.create_tf_session()
 
            # Load the saved model from a checkpoint
            chkpt_state = tf.train.get_checkpoint_state(self.model_dir)
            self.logger.info("Loading Checkpoint " + chkpt_state.model_checkpoint_path)
            tf_model = tf.train.Saver()
            tf_model.restore(self.tf_sess, chkpt_state.model_checkpoint_path)

            # Perform Model Evaluation
            self._create_eval_op(predictions, self.dataset.labels)
            acc = self._eval_loop()

            self.tf_sess.close()
        return acc

    def deploy(self, chkpt_id, img_size, num_classes):
        """ """
        with tf.Graph().as_default():
            if self.data_format.startswith('NC'):
                in_shape = [1, 3, img_size, img_size]
            else:
                in_shape = [1, img_size, img_size, 3]

            input_image = tf.placeholder(self.dtype, in_shape, name='input')
            logits, predictions = self._forward_prop(input_image, num_classes, training=False)
            out = tf.identity(predictions, 'output')

            saver = tf.train.Saver()
            self.create_tf_session()
            self.tb_writer  = tf.summary.FileWriter(self.model_dir, graph=self.tf_sess.graph)
            chkpt = self.chkpt_prfx + '-' + str(chkpt_id)
            saver.restore(self.tf_sess, chkpt)

            saver.save(self.tf_sess, self.model_prfx)
            self.tb_writer.close()
            self.tf_sess.close()
