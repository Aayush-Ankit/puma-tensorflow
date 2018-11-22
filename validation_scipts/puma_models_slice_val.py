## Validate puma_models dense and conv2d fucntionalities (apply functions)
import tensorflow as tf
import numpy as np
import math
import puma_models as puma

# model definiton
batch_size=2
inp = tf.get_variable("inp", [batch_size,4,4,1], trainable=False)
target = tf.get_variable("target", [batch_size], trainable=False, dtype=tf.int32)

conv1 = tf.layers.conv2d(inp, filters=3,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(), name = "conv1")
bn_1 = tf.layers.batch_normalization(conv1,training=True, name="bn1")
relu_1 = tf.nn.relu(bn_1, name="relu1")
flattened = tf.contrib.layers.flatten(relu_1)
dense1 = tf.layers.dense(inputs=flattened, units=10,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense1")
#dense2 = tf.layers.dense(inputs=dense1, units=10,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense2")

# define loss function
softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,logits=dense1,name="softmax_cross_entropy")
loss = tf.reduce_mean(softmax)

# collect all trainable variables from the graph
var_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# define variable init
init = tf.global_variables_initializer()


###### validate puma slice object and helper functions
## store layer 1 matrix in sliced format
#grad = var_val[0][0][0]*0.5
#l1_matrix = var_val[0][0][0]
#l1_slice = puma.sliced_data(matrix_in=l1_matrix, name=l1_matrix.name.split(":")[0])
#
#write_op = l1_slice.write_data(l1_matrix) # op to write sliced_data
#with tf.control_dependencies([write_op]):
#    l1_val = l1_slice.read_data()
#
#grad_out = l1_slice.update_data(grad) # op to update sliced_data
#
#with tf.Session() as sess:
#    summary_writer = tf.summary.FileWriter(logdir="./"+"puma_test", graph=sess.graph)
#    sess.run(tf.global_variables_initializer())
#
#    x, g = sess.run([l1_matrix, grad])
#    sess.run([write_op])
#    y = sess.run(l1_val)
#
#    sess.run(grad_out)
#    sess.run(grad_out)
#    sess.run(grad_out)
#    x3, y3 = sess.run([l1_matrix+3*grad, l1_slice.read_data()])
#
#    sess.run([write_op])
#    x4, y4 = sess.run([l1_matrix, l1_slice.read_data()])


##### check slicing for positive and negative numbers
#val_pos = l1_matrix[0][0][0][0]
#val_neg = -1*l1_matrix[0][0][0][0]
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    x = sess.run(val_pos)
#    y = sess.run(puma._slice(val_pos, 8, 0.001, None))
#    z = sess.run(puma._slice(val_neg, 8, 0.001, None))
#
#print (y)
#print ("\n")
#print (z)


#### validate slicing with puma outer_rpoduct object
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
puma_op = puma.outer_product(var_list=var_list, sigma=0.5, alpha=0.5)

reset_op = puma_op.crs_sync([var*0.0 for var in var_list])
crs_op = puma_op.crs_sync([var for var in var_list])

grad = [var*2.0 for var in var_list]
#update_op = puma_op.update(grad)
update_op = puma_op.apply_batch(zip(grad,var_list))

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(logdir="./"+"puma_test", graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    x = sess.run(var_list)
    g = sess.run(grad)

    sess.run(reset_op)
    x1 = sess.run(puma_op.read())

    sess.run(update_op)
    sess.run(update_op)
    sess.run(update_op)
    sess.run(update_op)
    x2 = sess.run(puma_op.read())

    sess.run(crs_op)
    x3 = sess.run(puma_op.read())














