## Study - weight representation a tensorflow kernel, modification and apply in tensorflow
import tensorflow as tf
import numpy as np
import math
import puma_models as puma

nonideality = puma.nonideality()
#quant1d = puma.quant1d()

# random data to feed to network - 4 input, 2 target
inp = tf.get_variable("inp", [1,4,4,5], trainable=False)
target = tf.get_variable("target", [1,2], trainable=False)

# define model - two layer (conv >> dense)
conv1 = tf.layers.conv2d(inputs=inp, filters=2,kernel_size=(3,3),padding='SAME',
            use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(), name="layer1")

flattened = tf.contrib.layers.flatten(conv1)

dense1 = tf.layers.dense(inputs=flattened, units=2,kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2')

# define loss function
loss = tf.losses.mean_squared_error(target, dense1)

# look at gradients and variables
var_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

opt = tf.train.GradientDescentOptimizer(0.1)
grads = opt.compute_gradients(loss)
update = opt.apply_gradients(grads)

# read kernel from layers (reuse=True, makes sure that the variable being accessed already exists)
with tf.variable_scope('layer1', reuse=True):
    w1 = tf.get_variable('kernel')

conv1_puma = puma.conv2d(name='layer1', precision=16)
conv1_puma.compute(inp, w1)

with tf.variable_scope('layer2', reuse=True):
    w2 = tf.get_variable('kernel')

dense1_puma = puma.dense(name='layer2', precision=16)
dense1_puma.compute(flattened, w2)

# quatize a 1d tensor
prec = 1


# see variable values before/after gradient computation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
