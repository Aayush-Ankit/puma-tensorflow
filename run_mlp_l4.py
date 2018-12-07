import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# add flags for user defined parameters
from absl import flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# define command line parameters
flags.DEFINE_string('optimizer', 'adam', 'spcify the optimizer to use - adam/vanilla/momentum/nestrov')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to run the training')
flags.DEFINE_integer('chpk_freq', 10, 'How often models are checkpointer during training')
flags.DEFINE_integer('batch_size', 32, 'batch size used in training')
flags.DEFINE_float('lr', 0.0001, 'Initial learning rate')
flags.DEFINE_string('log_dir', 'mlpL4_train', 'checkpoint directory where model and logs will be saved')
flags.DEFINE_boolean('restore', False, 'whether to restore training from checkpoint and log directory')
flags.DEFINE_integer('quant_bits', 8, 'number of bits for weight/activation quantization')
flags.DEFINE_integer('quant_delay', 101, 'when to start quantization during training')
flags.DEFINE_string('dataset', "/local/scratch/a/aankit/tensorflow/approx_memristor/cifar100/dataset/", 'what is the path to dataset')


# API for taining a dnn model
def train():

    print("Batch size: ", FLAGS.batch_size)

    with tf.device("/device:GPU:0"):
        training=True
        # random data to feed to network - 4 input, 2 target
        inp = tf.get_variable("inp", [FLAGS.batch_size, 1024], trainable=False)
        target = tf.get_variable("target", [FLAGS.batch_size], trainable=False)

        dense1 = tf.layers.dense(inputs=inp, units=256, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1')
        bn_1 = tf.contrib.layers.batch_norm(dense1,activation_fn=tf.nn.relu,is_training=training)
        dropout_1 = tf.layers.dropout(bn_1,training=training,rate=0.5)

        dense2 = tf.layers.dense(inputs=dropout_1, units=512, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2')
        bn_2 = tf.contrib.layers.batch_norm(dense2,activation_fn=tf.nn.relu,is_training=training)
        dropout_2 = tf.layers.dropout(bn_2,training=training,rate=0.5)

        dense3 = tf.layers.dense(inputs=dropout_2, units=512, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3')
        bn_3 = tf.contrib.layers.batch_norm(dense3,activation_fn=tf.nn.relu,is_training=training)
        dropout_3 = tf.layers.dropout(bn_3,training=training,rate=0.5)

        dense4 = tf.layers.dense(inputs=dropout_3, units=10, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer4')

    # load model
    logits = dense4
    labels = tf.cast(target, tf.int32)

    # create training op - add fake nodes to simulate quantization in inference
    # notice the qunat_delay of num_epochs+1, just adds fake nodes to be used later in inference
    softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="softmax_cross_entropy")
    loss = tf.reduce_mean(softmax)

    #var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        if (FLAGS.optimizer == "vanilla"):
            opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)
        elif (FLAGS.optimizer == "momentum"):
            opt = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9)
        elif (FLAGS.optimizer == "nesterov"):
            opt = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9, use_nesterov=True)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

        grad = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grad)

    # run training within a session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
        try:
            # setup logfile for this training session
            train_writer = tf.summary.FileWriter(logdir="./"+FLAGS.log_dir, graph=sess.graph)

            # setup counter, summary writes and model based on if restore required
            # keep track of progress during training with counter - for correct restoring from last checkpoint
            if FLAGS.restore:
                assert (tf.gfile.Exists(FLAGS.log_dir)), 'Restore requires log file from previous run, set restore to False and run...'
                print ('restoring train from: %s' % FLAGS.log_dir)
                saver.restore(sess, tf.train.latest_checkpoint("./"+FLAGS.log_dir))
                ckpt_name = tf.train.get_checkpoint_state(FLAGS.log_dir).model_checkpoint_path # extract the latest checkpoint
                counter = int(ckpt_name.split('-')[1]) # extract the counter from checkpoint
            else:
                counter = 0
                sess.run(tf.global_variables_initializer())

            # train network for user-defined epochs
            num_batch_per_epoch_train = math.ceil(10000 / FLAGS.batch_size)

            while (counter < FLAGS.epochs*num_batch_per_epoch_train):
                counter += 1
                # a batch of training
                start_time = time.time()
                _ = sess.run([train_op])
                duration = time.time() - start_time
                print("Step: %d \t Training time (1 batch): %0.4f" % (counter, duration))

        except KeyboardInterrupt:
            print("Interupted... saving model.")
            save_path = saver.save(sess, "./" + FLAGS.log_dir + "/model.ckpt-" + str(counter))


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()

