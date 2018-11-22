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

                #Layer 3 - 128 channels
                conv3 = tf.layers.conv2d(dropout_2, filters=128,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_3 = tf.contrib.layers.batch_norm(conv3,activation_fn=tf.nn.relu,is_training=training)

                # Layer 4 - 128 channels
                conv4 = tf.layers.conv2d(bn_3, filters=128,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_4 = tf.contrib.layers.batch_norm(conv4,activation_fn=tf.nn.relu,is_training=training)

                pool4 = tf.layers.max_pooling2d(bn_4, (2,2), (2,2), padding='SAME')

                dropout_4 = tf.layers.dropout(pool4,training=training,rate=0.5)

                #Layer 5 - 256 channels
                conv5 = tf.layers.conv2d(dropout_4, filters=256,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_5 = tf.contrib.layers.batch_norm(conv5,activation_fn=tf.nn.relu,is_training=training)

                # Layer 6 - 256 channels
                conv6 = tf.layers.conv2d(bn_5, filters=256,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_6 = tf.contrib.layers.batch_norm(conv6,activation_fn=tf.nn.relu,is_training=training)

                # Layer 7 - 256 channels
                conv7 = tf.layers.conv2d(bn_6, filters=256,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_7 = tf.contrib.layers.batch_norm(conv7,activation_fn=tf.nn.relu,is_training=training)

                pool7 = tf.layers.max_pooling2d(bn_7, (2,2), (2,2), padding='SAME')

                dropout_7 = tf.layers.dropout(pool7,training=training,rate=0.5)

                # Layer 8 - 512 channels
                conv8 = tf.layers.conv2d(dropout_7, filters=512,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_8 = tf.contrib.layers.batch_norm(conv8,activation_fn=tf.nn.relu,is_training=training)

                # Layer 9 - 512 channels
                conv9 = tf.layers.conv2d(bn_8, filters=512,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_9 = tf.contrib.layers.batch_norm(conv9,activation_fn=tf.nn.relu,is_training=training)

                # Layer 10 - 512 channels
                conv10 = tf.layers.conv2d(bn_9, filters=512,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_10 = tf.contrib.layers.batch_norm(conv10,activation_fn=tf.nn.relu,is_training=training)

                pool10 = tf.layers.max_pooling2d(bn_10, (2,2), (2,2), padding='SAME')

                dropout_7 = tf.layers.dropout(pool10,training=training,rate=0.5)

                # Layer 11 - 512 channels
                conv11 = tf.layers.conv2d(dropout_7, filters=512,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_11 = tf.contrib.layers.batch_norm(conv11,activation_fn=tf.nn.relu,is_training=training)

                # Layer 12 - 512 channels
                conv12 = tf.layers.conv2d(bn_11, filters=512,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_12 = tf.contrib.layers.batch_norm(conv12,activation_fn=tf.nn.relu,is_training=training)

                # Layer 13 - 512 channels
                conv13 = tf.layers.conv2d(bn_12, filters=512,kernel_size=(3,3),padding='SAME',
                    use_bias=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_13 = tf.contrib.layers.batch_norm(conv13,activation_fn=tf.nn.relu,is_training=training)

                pool13 = tf.layers.max_pooling2d(bn_13, (2,2), (2,2), padding='SAME')

                dropout_13 = tf.layers.dropout(pool13,training=training,rate=0.5)

                flattened = tf.contrib.layers.flatten(dropout_13)

                # Layer 14
                dense14 = tf.layers.dense(inputs=flattened, units=4096,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_14 = tf.contrib.layers.batch_norm(dense14,activation_fn=tf.nn.relu,is_training=training)

                dropout_14 = tf.layers.dropout(bn_14,training=training,rate=0.5)

                # Layer 15
                dense15 = tf.layers.dense(inputs=dropout_14, units=4096,kernel_initializer=tf.contrib.layers.xavier_initializer())

                bn_15 = tf.contrib.layers.batch_norm(dense15,activation_fn=tf.nn.relu,is_training=training)

                ## Layer 16
                dense16 = tf.layers.dense(inputs=bn_15, units=100,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer())

                tf.summary.scalar("dense16_mean",tf.reduce_mean(dense16))

            # data used in the model_op
            self.x_placeholder = x
            self.y_placeholder = y
            self.training = training
            self.logits = dense16

        build_model()

