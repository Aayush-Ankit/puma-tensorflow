## Study - grad computation, modificationa and apply in tensorflow

import tensorflow as tf
import numpy as np
import math
import puma_models as puma

nonideality = puma.nonideality()

# random data to feed to network - 4 input, 2 target
inp = tf.get_variable("inp", [1,4], trainable=False)
target = tf.get_variable("target", [1,2], trainable=False)

# define model - two layer
out1 = tf.layers.dense(inputs=inp, units=4,kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1')
out2 = tf.layers.dense(inputs=out1, units=2,kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2')

# define loss function
loss = tf.losses.mean_squared_error(target, out2)

# look at gradients and variables
var_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

opt = tf.train.GradientDescentOptimizer(0.1)
grads = opt.compute_gradients(loss)
grads_n = nonideality.apply_n(grads)
update = opt.apply_gradients(grads_n)

# see variable values before/after gradient computation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x = sess.run(var_val) # variable before gradient
    y = sess.run(grads) # sw-gradient values
    y_n = sess.run(grads_n) # gradient values
    sess.run(update)

