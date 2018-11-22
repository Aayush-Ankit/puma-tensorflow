import tensorflow as tf
from data_loader import Loader
import matplotlib.pyplot as plt
from absl import flags

# add flags for user defined parameters
from absl import flags
flags = tf.app.flags
FLAGS = flags.FLAGS


# Model definition of dnn
class Model():
    def __init__(self, batch_size):
        loader = Loader(batch_size)
        iterator = loader.get_dataset()

        def build_model():
            with tf.device("/device:GPU:0"):
                x_loaded,y_loaded = iterator.get_next()

                x = tf.placeholder_with_default(x_loaded,(None,32,32,3),name="x_placeholder")
                y = tf.placeholder_with_default(y_loaded,(None),name="y_placeholder")

                training = tf.placeholder_with_default(True,name="training_bool",shape=())

                #Layer1 - 64 channels
                conv1 = tf.layers.conv2d(x, filters=64,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_1 = tf.contrib.layers.batch_norm(conv1,activation_fn=tf.nn.relu,is_training=training)

                # Layer2 - 64 channels
                conv2 = tf.layers.conv2d(bn_1, filters=64,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_2 = tf.contrib.layers.batch_norm(conv2,activation_fn=tf.nn.relu,is_training=training)

                pool2 = tf.layers.max_pooling2d(bn_2, (2,2), (2,2), padding='SAME')

                dropout_2 = tf.layers.dropout(pool2,training=training,rate=0.5)

                flattened = tf.contrib.layers.flatten(dropout_2)

                # Layer 14
                dense14 = tf.layers.dense(inputs=flattened, units=4096,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_14 = tf.contrib.layers.batch_norm(dense14,activation_fn=tf.nn.relu,is_training=training)

                dropout_14 = tf.layers.dropout(bn_14,training=training,rate=0.5)

                ## Layer 16
                dense16 = tf.layers.dense(inputs=dropout_14, units=100,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())

                tf.summary.scalar("dense16_mean",tf.reduce_mean(dense16))

            # data used in the model_op
            self.x_placeholder = x
            self.y_placeholder = y
            self.training = training
            self.logits = dense16

        build_model()

