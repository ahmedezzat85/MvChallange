import os
import numpy as np
import tensorflow as tf

# Disable Tensorflow logs except for errors
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Generate random data
predictions = np.random.rand(20, 10)
labels      = np.random.randint(0, 10, 20)
print ('labels\n', labels)
print ('-----------------')

# Labels to one hot
labels = tf.convert_to_tensor(labels)
one_hot_labels = tf.one_hot(labels, 10)

l1 = tf.argmax(one_hot_labels, axis=1)
top1_tensor = tf.equal(tf.argmax(predictions, axis=1), l1)
top1_op     = tf.reduce_mean(tf.cast(top1_tensor, tf.float32))
top5_tensor = tf.nn.in_top_k(predictions, tf.argmax(one_hot_labels, axis=1), 5)
top5_op     = tf.reduce_mean(tf.cast(top5_tensor, tf.float32))
top5_op_2   = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions, labels, 5), tf.float32))
Op = [l1, top1_op, top5_op, top5_op_2]
# top_5_2 = tf.metrics.mean(tf.cast(tf.nn.in_top_k(predictions, labels, 5), tf.float32))

t = tf.convert_to_tensor(predictions)
with tf.Session() as sess:
    val, idx = sess.run(tf.nn.top_k(t, k=5, sorted=True, name='TOP-5'))
    a, b, c, d = sess.run(Op)
    print (idx)
    print ('-------')
    print (a)
    print ('*****')
    print (b)
    print ('*****')
    print (c)
    print ('*****')
    print (d)
    print ('-------')
