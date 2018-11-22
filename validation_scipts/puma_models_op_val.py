## Validate puma_models dense and conv2d fucntionalities (apply functions)
import tensorflow as tf
import numpy as np
import math
import puma_models as puma

#nonideality = puma.nonideality()
puma_op = puma.outer_product()

# model definiton
batch_size=8
inp = tf.get_variable("inp", [batch_size,4,4,1], trainable=False)
target = tf.get_variable("target", [batch_size], trainable=False, dtype=tf.int32)

conv1 = tf.layers.conv2d(inp, filters=3,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(), name = "conv1")
#conv1_l = puma.conv2d (inputs=x, filters=32, kernel_size=(3,3), name="conv1")
#conv1 = conv1_l.apply(x)
bn_1 = tf.layers.batch_normalization(conv1,training=True, name="bn1")
relu_1 = tf.nn.relu(bn_1, name="relu1")
flattened = tf.contrib.layers.flatten(relu_1)
dense1 = tf.layers.dense(inputs=flattened, units=10,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense1")
dense2 = tf.layers.dense(inputs=dense1, units=10,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense2")
#dense1_l = puma.dense(inputs=flattened, units=100, name="dense1")
#dense1 = dense1_l.apply(flattened)

# define loss function
softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,logits=dense2,name="softmax_cross_entropy")
loss = tf.reduce_mean(softmax)

## THIS doesn't work - SURPRISINGLY but sliced softmax (softmax[i]) does ???
## split tensor components to make a list (length of list = tensor size) to get a list of per-example loss
# softmax_split = tf.split(softmax, batch_size)

# collect all trainable variables from the graph
var_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

lr = 0.1
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(loss)

    ## tf.gradeints returns sum(dy/dx) for all x
    ## split gradient computation for each example to prevent summation
    #grads_split = [tf.gradients(softmax[i], var_val) for i in range(batch_size)]

    ## compute grad_avg over batch with puma outer product
    #grads_puma = puma_op.apply(softmax, var_val, update_ops)

    update = opt.apply_gradients(grads)

## compute batch-averaged gradients and per-example gradients and validate
#with tf.Session() as sess:
#    summary_writer = tf.summary.FileWriter(logdir="./"+"puma_test", graph=sess.graph)
#    sess.run(tf.global_variables_initializer())
#    x, x1, x2 = sess.run([grads, grads_split, grads_puma]) # kernel before apply_gradient
#
#x1_avg = np.mean(x1, axis=0) # this should be equal to x for VALIDATION PASS
#
#for i in range(len(x)):
#    err = np.sum(np.sum(abs(x1_avg[i]-x[i][0])))
#    print("layer " + str(i) + " error: " + str(err))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x =  sess.run(var_val)
    g = sess.run(grads)
    sess.run(update)
    y = sess.run(var_val)


##################################### LEARNING/NOTES #####################################
# 1. Gradient computed in opt.compute_gradients/tf.gradeints would follow the same reduction scheme across batches - as done for loss across batch

# 2. Split the input and models for per-example gradient is not required here
        #inp_split = tf.split(inp,batch_size)
        #conv1_split = [tf.layers.conv2d(inp_split[i], filters=32,kernel_size=(3,3),padding='SAME',
            #use_bias=True, name = "conv1", reuse=True) for i in range(self.batch_size)]

# 3. Output of soft-max is itself per example loss - hence, can be used to compute per example gradeint in back-prop

# 4. If loss was of form mean-square (which uses other reduction like across batch etc.), 2 would be required
    #loss = tf.losses.mean_squared_error(target, dense1)
    #loss_tmp = tf.losses.mean_squared_error(y, dense1,reduction=tf.losses.Reduction.NONE)
