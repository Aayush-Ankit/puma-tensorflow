import tensorflow as tf
from data_loader import Loader
import matplotlib.pyplot as plt
from absl import flags

# add flags for user defined parameters
from absl import flags
flags = tf.app.flags
FLAGS = flags.FLAGS

import puma_models as puma

# Model definition of dnn
class Model():
    def __init__(self, batch_size):
        loader = Loader(batch_size)
        iterator = loader.get_dataset()
        self.batch_size = batch_size

        def build_model():
            with tf.device("/device:GPU:0"):
                x_loaded,y_loaded = iterator.get_next()

                x = tf.placeholder_with_default(x_loaded,(None,32,32,3),name="x_placeholder")
                y = tf.placeholder_with_default(y_loaded,(None),name="y_placeholder")

                training = tf.placeholder_with_default(True,name="training_bool",shape=())

                #Layer1 - 64 channels
                conv1 = tf.layers.conv2d(x, filters=32,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer(), name="conv1")
                #conv1_l = puma.conv2d (inputs=x, filters=32, kernel_size=(3,3), name="conv1")
                #conv1 = conv1_l.apply(x)

                #bn_1 = tf.contrib.layers.batch_norm(conv1,activation_fn=tf.nn.relu,is_training=training)
                bn_1 = tf.layers.batch_normalization(conv1,training=training, name="bn1")
                relu_1 = tf.nn.relu(bn_1, name="relu1")

                flattened = tf.contrib.layers.flatten(relu_1)

                ## Layer 4
                dense1 = tf.layers.dense(inputs=flattened, units=100,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense1")
                #dense1_l = puma.dense(inputs=flattened, units=100, name="dense1")
                #dense1 = dense1_l.apply(flattened)

                tf.summary.scalar("dense1_mean",tf.reduce_mean(dense1))

                ### add a list of logits for propagating individual examples separately
                #x_split = tf.split(x,batch_size)
                #y_split = tf.split(y,batch_size)

                #conv1_split = [tf.layers.conv2d(x_split[i], filters=32,kernel_size=(3,3),padding='SAME',
                #    use_bias=True, name = "conv1", reuse=True) for i in range(self.batch_size)]

                #bn_1_split = [tf.layers.batch_normalization(conv1_split[i],training=training, name="bn1", reuse=True) for i in range(self.batch_size)]

                #relu_1_split = [tf.nn.relu(bn_1_split[i], name="relu_split"+str(i)) for i in range(self.batch_size)]

                #flattened_split = [tf.contrib.layers.flatten(relu_1_split[i]) for i in range(self.batch_size)]

                #dense1_split = [tf.layers.dense(flattened_split[i], units=100,activation=None, name="dense1", reuse=True) for i in range(self.batch_size)]

            # data used in the model_op
            self.x_placeholder = x
            self.y_placeholder = y
            self.training = training
            self.logits = dense1
            #self.logits_split = [dense1_split[i] for i in range(self.batch_size)]

            # list of neuron layers to monitor for error resiliency
            self.layers = [conv1, dense1]

        build_model()

