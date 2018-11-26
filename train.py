import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# Specify the model and dataloader
from data_loader import Loader
#from vgg16 import Model
from vgg16 import Model

import puma_models as puma
#nonideality = puma.nonideality()
#puma_op = puma.outer_product()

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
flags.DEFINE_string('log_dir', 'cifar100_train', 'checkpoint directory where model and logs will be saved')
flags.DEFINE_boolean('restore', False, 'whether to restore training from checkpoint and log directory')
flags.DEFINE_integer('quant_bits', 8, 'number of bits for weight/activation quantization')
flags.DEFINE_integer('quant_delay', 101, 'when to start quantization during training')
flags.DEFINE_string('dataset', "/local/scratch/a/aankit/tensorflow/approx_memristor/cifar100/dataset/", 'what is the path to dataset')


# API for taining a dnn model
def train():

    # dataloader for validation accuracy computation  -dataloader for training data is embedded in model
    loader = Loader(FLAGS.batch_size, FLAGS.dataset)
    val_iterator = loader.get_dataset(train=False).get_next()

    # load model
    model = Model(FLAGS.batch_size)
    logits = model.logits
    labels = model.y_placeholder

    # create training op - add fake nodes to simulate quantization in inference
    # notice the qunat_delay of num_epochs+1, just adds fake nodes to be used later in inference
    softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="softmax_cross_entropy")
    loss = tf.reduce_mean(softmax)

    #loss_split = [tf.reduce_mean]
    #tf.contrib.quantize.experimental_create_training_graph(weight_bits=FLAGS.quant_bits,
    #                                                       activation_bits=FLAGS.quant_bits,
    #                                                       quant_delay=FLAGS.quant_delay)

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

    # create ops for validation and training accuracy
    outputs = tf.nn.softmax(logits)
    equality = tf.nn.in_top_k(outputs, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    tf.summary.scalar("Loss", loss)
    tf.summary.scalar("Training accuracy", accuracy)

    # set a saver for checkpointing
    saver = tf.train.Saver()

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

            merge = tf.summary.merge_all()
            print ('counter: ', counter)
            print ('Quantization bits: %d    delay: %d ' % (FLAGS.quant_bits, FLAGS.quant_delay))

            # train network for user-defined epochs
            num_batch_per_epoch_train = math.ceil(loader.num_training_examples / FLAGS.batch_size)

            while (counter < FLAGS.epochs*num_batch_per_epoch_train):
                counter += 1
                # a batch of training
                start_time = time.time()
                _, summary = sess.run([train_op, merge],feed_dict={})
                duration = time.time() - start_time
                print("Step: %d \t Training time (1 batch): %0.4f" % (counter, duration))
                train_writer.add_summary(summary, global_step=counter)

                # compute validation accuracy every epoch
                if (counter % num_batch_per_epoch_train == 0):
                    num_batch_per_epoch_val = math.ceil(loader.num_testing_examples / FLAGS.batch_size)

                    val_counter = 0
                    true_count = 0

                    while (val_counter < num_batch_per_epoch_val):
                        val_counter += 1
                        # a batch of validation
                        val_x,val_y = sess.run(val_iterator)
                        val_equality = sess.run([equality],feed_dict={model.x_placeholder:val_x,model.y_placeholder:val_y,model.training:False})
                        true_count += np.sum(val_equality)

                    val_accuracy = true_count / (FLAGS.batch_size * num_batch_per_epoch_val)
                    accuracy_summary = tf.Summary()
                    accuracy_summary.value.add(tag='Validation Accuracy',simple_value=val_accuracy)
                    train_writer.add_summary(accuracy_summary, global_step=counter)
                    print ('Validation accuracy %.4f' % val_accuracy)

                if (counter%(FLAGS.chpk_freq*num_batch_per_epoch_train) == 0):
                    # Save model
                    print("Periodically saving model...")
                    save_path = saver.save(sess, "./" + FLAGS.log_dir + "/model.ckpt")

        except KeyboardInterrupt:
            print("Interupted... saving model.")
            save_path = saver.save(sess, "./" + FLAGS.log_dir + "/model.ckpt-" + str(counter))


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
