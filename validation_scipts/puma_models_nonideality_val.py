## Study - grad computation, modificationa and apply in tensorflow

import tensorflow as tf
import numpy as np
import math
import puma_models as puma

from tensorflow.python import debug as tf_debug

nonideality = puma.nonideality()

# random data to feed to network - 4 input, 2 target
inp = tf.get_variable("inp", [1,4], trainable=False)
target = tf.get_variable("target", [1,2], trainable=False)

# define model - two layer
out1 = tf.layers.dense(inputs=inp, units=4,kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1')
out2 = tf.layers.dense(inputs=out1, units=2,kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2')

# define loss function
loss = tf.losses.mean_squared_error(target, out2)

# collect all trainable variables from the graph
var_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

lr = 0.1
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(loss)
    grads_n = nonideality.apply_n(grads)
    update = opt.apply_gradients(grads_n)
    #update = opt.apply_gradients(grads)

# see variable values before/after gradient computation
#sess = tf.Session()
with tf.Session() as sess:
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "cbric-gpu3.ecn.purdue.edu:6064")
    summary_writer = tf.summary.FileWriter(logdir="./"+"puma_nonideal", graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    x,y,y_n,_ = sess.run([var_val, grads, grads_n, update]) # kernel before apply_gradient
    x1 = sess.run(var_val) # kernel after apply_update

# print values - a and b should be same
a = x[0] - x1[0]
b = lr*y[0][0]
c = lr*y_n[0][0]

## check functionality of _appy_noise function nonideality class (zeroing the noise of zero grad)
#a = grads[0][0] * tf.zeros(tf.shape(grads[0][0]))
#y = nonideality._apply_noise(grads[0][0], grads[0][1])
#y1 = nonideality._apply_noise(a, grads[0][1])
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    n, n1, a, g, w = sess.run([y, y1, a, grads[0][0], grads[0][1]])
#
#print(n)
#print("\n")
#print(n1)
#print("\n")
#print(g)

# check functionality of _appy_noise function nonideality class (zeroing the noise of zero grad)
y1 = nonideality._compute_noise(grads[0][0], grads[0][1])
y2 = nonideality._compute_nonlinearity(grads[0][0], grads[0][1])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #n, n1, a, g, w = sess.run([y, y1, a, grads[0][0], grads[0][1]])
    n1, n2, g = sess.run([y1, y2, grads[0][0]])

print(n1)
print("\n")
print(n2)
print("\n")
print(g)



