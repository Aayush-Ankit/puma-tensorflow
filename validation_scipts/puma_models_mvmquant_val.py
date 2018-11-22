## Validate puma_models dense and conv2d fucntionalities (apply functions)
import tensorflow as tf
import numpy as np
import math
import puma_models as puma

# validate dense layer - 2X2 layer
inp_dense = tf.get_variable("inp_dense", [1,3], trainable=False)
dense_l = puma.dense (inputs=inp_dense, units=3, name="dense1", precision=1) # prec1 gives viaualization!
[out_full, out_exp, out_act, kernel, kernel_quant] = dense_l.validate(inp_dense)

## validate conv layer
# not doing (validate for puma.conv2d needs to eb written), as process for conv and dense are same

# see variable values before/after gradient computation
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    [a,b,c,d,e] = sess.run([out_full, out_exp, out_act, kernel, kernel_quant])
