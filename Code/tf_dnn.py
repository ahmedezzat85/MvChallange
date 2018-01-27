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
    def __init__(self, model_name, data_format='NHWC', logs_dir=None):

        # Disable Tensorflow logs except for errors
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Parameter initializations
        self.logger       = None
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
        self.log_dir      = logs_dir
        self.summary_list = []

        # Get the neural network model function
        net_module    = import_module('model.' + model_name)
        self.model_fn = net_module.snpx_net_create

        self.chkpt_dir = os.path.join(self.log_dir, 'chkpt')
        utils.create_dir(self.chkpt_dir)
        self.chkpt_prfx = os.path.join(self.chkpt_dir, 'CHKPT')

    def tick(self):
        return time() - self.base_tick

    def _dump_hyperparameters(self, begin_epoch):
        """ """
        hp = [
            ['**Input Size**', str(self.dataset.shape)],
            ['**Scales**', '{'+str(self.dataset.scale_min)+', '+str(self.dataset.scale_max)+'}'],
            ['**Learning Rate**', str(self.hp.lr)], 
            ['**Optimizer**', self.hp.optimizer], 
            ['**Weight Decay**', str(self.hp.wd)], 
            ['**Batch Size**', str(self.hp.batch_size)]]
        summ_op = tf.summary.merge(
                    [tf.summary.text(self.model_name + '/HyperParameters', tf.convert_to_tensor(hp)),
                    tf.summary.text(self.model_name + '/Dataset', tf.convert_to_tensor(self.dataset.name))])
        s = self.tf_sess.run(summ_op)
        self.tb_writer.add_summary(s, begin_epoch)
        self.tb_writer.flush()

    def _load_dataset(self, training=True):
        """ """
        with tf.device('/cpu:0'):
            self.dataset.read(self.hp.batch_size, training, self.data_format, self.hp.data_aug, self.dtype)
    
    def _forward_prop(self, batch, num_classes, training=True):
        """ """
        logits, predictions = self.model_fn(num_classes, batch, self.data_format, is_training=training)
        return logits, predictions

    def loss_fn(self, logits):
        cross_entropy = tf.losses.softmax_cross_entropy(self.dataset.labels, logits)
        self.summary_list.append(tf.summary.scalar("Loss", cross_entropy))
        return cross_entropy

    def _create_train_op(self, logits):
        """ """
        self.global_step = tf.train.get_or_create_global_step()
        # Get the optimizer
        if self.hp.lr_decay:
            lr = tf.train.exponential_decay(self.hp.lr, self.global_step, self.hp.lr_decay_steps,  
                                                self.hp.lr_decay, True)
        else:
            lr = self.hp.lr
        self.summary_list.append(tf.summary.scalar("Learning-Rate", lr))

        optmz = self.hp.optimizer.lower()
        if optmz == 'sgd':
            opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
        elif optmz == 'adam':
            opt = tf.train.AdamOptimizer(lr)
        elif optmz == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(lr)
 
        # Compute the loss and the train_op
        self.loss = self.loss_fn(logits)
        if self.hp.wd > 0:
            l2_loss = self.hp.wd * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() 
                                                if 'batch_normalization' not in v.name])
            self.loss = self.loss + l2_loss

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
                    speed = ((step - last_step)  * self.hp.batch_size) / elapsed
                    last_step = step
                    last_log_tick  = self.tick()
                    self.logger.info('(%.3f)Epoch[%d] Batch[%d]\tloss: %.3f\tspeed: %.3f samples/sec', 
                                      self.tick(), self.epoch, step, loss, speed)
            except tf.errors.OutOfRangeError:
                break
        self.logger.info('Epoch Training Time = %.3f', self.tick() - epoch_start_time)
        self.saver.save(self.tf_sess, self.chkpt_prfx, self.epoch + 1)

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

    def train(self, dataset, hp, num_epoch, begin_epoch, log_freq_sec=1):
        """ """
        self.hp         = hp
        self.dataset    = dataset
        self.log_freq   = log_freq_sec

        t_start = datetime.now()
        self.logger = utils.create_logger(self.model_name, os.path.join(self.log_dir, 'Train.log'))
        self.logger.info("Training Started at  : " + t_start.strftime("%Y-%m-%d %H:%M:%S"))

        with tf.Graph().as_default():
            # Load the dataset
            self._load_dataset()

            # Forward Propagation
            self.training = tf.placeholder(tf.bool, name='Train_Flag')
            logits, _ = self._forward_prop(self.dataset.images, self.dataset.num_classes, self.training)
        
            self._create_train_op(logits)

            # Create a TF Session
            self.create_tf_session()

            # Create Tensorboard stuff
            self.summary_op = tf.summary.merge(self.summary_list)
            self.tb_writer  = tf.summary.FileWriter(self.log_dir, graph=self.tf_sess.graph)
            self._dump_hyperparameters(begin_epoch)

            if begin_epoch > 0:
                # Load the saved model from a checkpoint
                chkpt = self.chkpt_prfx + '-' + str(begin_epoch)
                self.logger.info("Loading Checkpoint " + chkpt)
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
                acc_summ.value.add(simple_value=val_acc, tag="Validation-Accuracy")
                self.tb_writer.flush()

                self.tb_writer.add_summary(acc_summ, self.epoch)
                self.logger.info('Epoch[%d] Validation-Accuracy = %.2f%%', self.epoch, val_acc)

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

    def deploy(self, deploy_dir, img_size, num_classes, chkpt_id=0):
        """ """
        utils.create_dir(deploy_dir)
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
            self.tb_writer  = tf.summary.FileWriter(deploy_dir, graph=self.tf_sess.graph)
            if chkpt_id == 0:
                chkpt_state = tf.train.get_checkpoint_state(self.chkpt_dir)
                chkpt = chkpt_state.model_checkpoint_path
            else:
                chkpt = self.chkpt_prfx + '-' + str(chkpt_id)
            saver.restore(self.tf_sess, chkpt)
            saver.save(self.tf_sess, os.path.join(deploy_dir, self.model_name))

            tf.train.write_graph(self.tf_sess.graph_def, deploy_dir, self.model_name+'.pb', False)
            tf.train.write_graph(self.tf_sess.graph_def, deploy_dir, self.model_name+'.pbtxt')

            self.tb_writer.close()
            self.tf_sess.close()

