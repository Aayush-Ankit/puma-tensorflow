import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

# Specify the model and dataloader
from data_loader import Loader
from vgg16 import Model

# add flags for user defined parameters
from absl import flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# define command line parameters
flags.DEFINE_integer('batch_size', 64, 'batch size used in training')
flags.DEFINE_string('log_dir', 'cifar100_test', 'checkpoint directory where model and logs will be saved')
flags.DEFINE_string('chpk_dir', 'cifar100_train', 'checkpoint directory where model and logs will be saved')
flags.DEFINE_integer('quant_bits', 8, 'number of bits for weight/activation quantization')


# API for taining a dnn model
def test():

    # dataloader for testing accuracy computation  -dataloader for training data is embedded in model
    loader = Loader(FLAGS.batch_size)
    test_iterator = loader.get_dataset(train=False).get_next()

    # load model
    model = Model(FLAGS.batch_size)
    logits = model.logits
    labels = model.y_placeholder

    # create testing op - add fake nodes to simulate quantization in inference
    tf.contrib.quantize.experimental_create_eval_graph(weight_bits=FLAGS.quant_bits,
                                                       activation_bits=FLAGS.quant_bits)
    outputs = tf.nn.softmax(logits)
    test_op = tf.nn.in_top_k(outputs, labels, 1)
    acc_op = tf.reduce_mean(tf.cast(test_op, tf.float32))

    # set a saver for checkpointing
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:

        # setup logfile for this training session
        test_writer = tf.summary.FileWriter(logdir="./"+FLAGS.log_dir, graph=sess.graph)

        assert (tf.gfile.Exists(FLAGS.chpk_dir)), 'Chpk file doesn\'t contain a trained model/checkpoint ...'
        saver.restore(sess, tf.train.latest_checkpoint("./"+FLAGS.chpk_dir))

        num_batch_per_epoch_test = math.ceil(loader.num_testing_examples / FLAGS.batch_size)

        print ('Quantization bits: %d    Checkpoint: %s    Log:%s' % (FLAGS.quant_bits, FLAGS.chpk_dir, FLAGS.log_dir))
        counter = 0
        true_count = 0

        while (counter < num_batch_per_epoch_test):
            counter += 1

            # a batch of testing
            test_x,test_y = sess.run(test_iterator)
            test_equality, batch_accuracy  = sess.run([test_op, acc_op],feed_dict={model.x_placeholder:test_x,model.y_placeholder:test_y,model.training:False})
            true_count += np.sum(test_equality)

            # add batch accuracy to summary
            batch_accuracy_summary = tf.Summary()
            batch_accuracy_summary.value.add(tag='Batch Accuracy',simple_value=batch_accuracy)
            test_writer.add_summary(batch_accuracy_summary, global_step=counter)

        test_accuracy = true_count / (FLAGS.batch_size * num_batch_per_epoch_test)
        print ('Testing accuracy %.4f' % test_accuracy)

        # add test accuracy to summary
        accuracy_summary = tf.Summary()
        accuracy_summary.value.add(tag='Testing Accuracy',simple_value=test_accuracy)
        test_writer.add_summary(accuracy_summary, global_step=counter)


def main(argv=None):
    test()


if __name__ == "__main__":
    tf.app.run()

