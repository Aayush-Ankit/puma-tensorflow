# API to train models using PANTHER's OPA technique (includes non-ideality models within outer-product class)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import time

# Specify the model and dataloader
from data_loader import Loader
from vgg16_puma import Model
import puma_models as puma

# add flags for user defined parameters
from absl import flags
flags = tf.app.flags
FLAGS = flags.FLAGS

# add debugger
from tensorflow.python import debug as tf_debug

# define command line parameters
flags.DEFINE_string('optimizer', 'adam', 'spcify the optimizer to use - adam/vanilla/momentum/nestrov')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to run the training')
flags.DEFINE_integer('chpk_freq', 10, 'How often models are checkpointed during training')
flags.DEFINE_integer('batch_size', 64, 'batch size used in training')
flags.DEFINE_float('lr', 0.0001, 'Initial learning rate')
flags.DEFINE_string('logdir', 'test/sample_train_run', 'checkpoint directory where model and logs will be saved')
flags.DEFINE_string('dataset', "/home/glau/puma/puma-tensorflow/cifar100/dataset/", 'what is the path to dataset')

# flags for restoring a previous run from a specific start counter
flags.DEFINE_boolean('restore', False, 'whether to restore training from checkpoint and log directory')
flags.DEFINE_integer('start_counter', 0, 'start counter for restore training, used when chkpt has no counter')

# flags for quantization (homogenous for the entire model) - NOTE: quantization support is not present in this script
flags.DEFINE_integer('quant_bits', 8, 'number of bits for weight/activation quantization')
flags.DEFINE_integer('quant_delay', 101, 'when to start quantization during training')

# flags to set non-ideality bounds for reram
flags.DEFINE_float('puma_sigma', 0.0, 'nonideality-write-noise-sigma')
flags.DEFINE_float('puma_alpha', 0.0, 'nonideality-write-nonlinearity-alpha')

# flag for panther weight updates (non-ideality, bit-slicing, heterogenous config, and crs)
flags.DEFINE_boolean('ifpanther', False, 'Chose whether to do panther-based training or typical')
flags.DEFINE_boolean('ifmixed', False, 'Chose whether to do mixed precision training or not')
flags.DEFINE_integer('slice_bits', 6, 'number of bits per outer-product slice')
flags.DEFINE_list('slice_bits_list', [6,6,6,6,5,5,5,5] , 'Specify slice_bits as comma-sepearted values (starting from LSB to MSB slice)')
flags.DEFINE_integer('crs_freq', 4, 'How often carry resolution occurs during training - batch granularity')


# API for taining a dnn model
def train():

    # for PANTHER-mode print slice precision
    if (FLAGS.ifpanther):
        if (FLAGS.ifmixed):
            assert (len(FLAGS.slice_bits_list) == 8), "list length should be 8"
            print ("PUMA sliced_bits_list", FLAGS.slice_bits_list)
        else:
            print ("PUMA slice bits: ", FLAGS.slice_bits)

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

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    if (FLAGS.ifpanther):
        ## NOTE: If not using puma-outerproduct; directly use nonideality class without going through outer_product class
        # outer_product is built on example-wise gradients and is slow [To Try for speedup - see Goodfeli blog - https://github.com/tensorflow/tensorflow/issues/4897]
        #nonideality = puma.nonideality(sigma=FLAGS.puma_sigma, alpha=FLAGS.puma_alpha)
        puma_op = puma.outer_product(var_list=var_list, sigma=FLAGS.puma_sigma, alpha=FLAGS.puma_alpha, \
                slice_bits=FLAGS.slice_bits, ifmixed=FLAGS.ifmixed, slice_bits_list=[int(x) for x in FLAGS.slice_bits_list])

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

        # Layer gradient computation
        grad = opt.compute_gradients(loss)

        # Weight gradient computation and update
        if (FLAGS.ifpanther):
            print ('Adding PANTHER ops')
            ## NOTE: If not using puma-outerproduct; directly use nonideality class without going through outer_product class
            #grad_n = nonideality.apply(grad)
            grad_n = puma_op.apply_batch(grad)
            train_op = opt.apply_gradients(zip(grad_n[0], var_list))
        else:
            grad_n = grad # added this placeholder for grad_n for consistency among different run types
            train_op = opt.apply_gradients(grad)

    # create ops for validation and training accuracy
    outputs = tf.nn.softmax(logits)
    equality = tf.nn.in_top_k(outputs, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    # create ops for top-5 accuracy
    equality5 = tf.nn.in_top_k(outputs, labels, 5)
    accuracy5 = tf.reduce_mean(tf.cast(equality5, tf.float32))

    tf.summary.scalar("Loss", loss)
    tf.summary.scalar("Training accuracy - Top-1", accuracy)
    tf.summary.scalar("Training accuracy - Top-5", accuracy5)

    # capture slice-wise saturation statistics
    if (FLAGS.ifpanther):
        puma_sat_stats  = grad_n[1]
        tf.summary.scalar("PUMA Parallel Write Saturation", puma_sat_stats[1])
        for i in range(puma_sat_stats[0].get_shape()[0]):
            tf.summary.scalar("PUMA Parallel Write Saturation-"+"Slice "+str(i), puma_sat_stats[0][i])

    # puma crs_sync
    if (FLAGS.ifpanther):
        crs_op = puma_op.crs_sync(var_list)

    # set a saver for checkpointing
    saver = tf.train.Saver()

    # run training within a session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        try:
            # setup logfile for this training session
            train_writer = tf.summary.FileWriter(logdir="./"+FLAGS.logdir, graph=sess.graph)

            # setup counter, summary writes and model based on if restore required
            # keep track of progress during training with counter - for correct restoring from last checkpoint
            if FLAGS.restore:
                assert (tf.gfile.Exists(FLAGS.logdir)), 'Restore requires log file from previous run, set restore to False and run...'
                print ('restoring train from: %s' % FLAGS.logdir)
                saver.restore(sess, tf.train.latest_checkpoint("./"+FLAGS.logdir))
                ckpt_name = tf.train.get_checkpoint_state(FLAGS.logdir).model_checkpoint_path # extract the latest checkpoint
                ## patch for case where model gets saved naturally, not because it stopped due to interrupt
                # counter = int(ckpt_name.split('-')[1]) # extract the counter from checkpoint
                temp_l = ckpt_name.split('-')
                if (len(temp_l) > 1):
                    counter = int(ckpt_name.split('-')[1]) # extract the counter from checkpoint
                else:
                    print ("Restoring from counter:", FLAGS.start_counter)
                    counter = FLAGS.start_counter
            else:
                counter = 0
                sess.run(tf.global_variables_initializer())

            merge = tf.summary.merge_all()
            print ('counter: ', counter)
            # NOTE: keep below commented until quantization support is not enabled for training
            #print ('Quantization bits: %d    delay: %d ' % (FLAGS.quant_bits, FLAGS.quant_delay))

            # train network for user-defined epochs
            num_batch_per_epoch_train = math.ceil(loader.num_training_examples / FLAGS.batch_size)
            print ('number of batches per epoch: ', num_batch_per_epoch_train)

            while (counter < FLAGS.epochs*num_batch_per_epoch_train):
                counter += 1

                # puma carry resolution step
                if (FLAGS.ifpanther and (counter % FLAGS.crs_freq == 0)):
                    print("Performing puma crs.....")
                    sess.run(crs_op)

                # a batch of training
                start_time = time.time()

                ## Uncomment the lines below if running detailed traces - for runtime compute and mmeory data
                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) #Uncomment these to get run-time compute/memory utilization
                #run_metadata = tf.RunMetadata()
                #_, _, summary = sess.run([grad_n, train_op, merge],feed_dict={}, options=run_options, run_metadata=run_metadata)
                _, _, summary = sess.run([grad_n, train_op, merge],feed_dict={})

                duration = time.time() - start_time
                print("Step: %d \t Training time (1 batch): %0.4f" % (counter, duration))

                ## Uncomment the lines below if running detailed traces - for runtime compute and mmeory data
                #train_writer.add_run_metadata(run_metadata, 'step%d' % counter)
                train_writer.add_summary(summary, global_step=counter)

                # compute validation accuracy every epoch
                if (counter % num_batch_per_epoch_train == 0):
                    num_batch_per_epoch_val = math.ceil(loader.num_testing_examples / FLAGS.batch_size)

                    val_counter = 0
                    true_count = 0
                    true_count5 = 0

                    while (val_counter < num_batch_per_epoch_val):
                        val_counter += 1
                        # a batch of validation
                        val_x,val_y = sess.run(val_iterator)
                        val_equality, val_equality5 = sess.run([equality, equality5],feed_dict={model.x_placeholder:val_x,model.y_placeholder:val_y,model.training:False})
                        true_count += np.sum(val_equality)
                        true_count5 += np.sum(val_equality5)

                    val_accuracy = true_count / (FLAGS.batch_size * num_batch_per_epoch_val)
                    val_accuracy5 = true_count5 / (FLAGS.batch_size * num_batch_per_epoch_val)
                    accuracy_summary = tf.Summary()
                    accuracy_summary.value.add(tag='Validation Accuracy - Top-1',simple_value=val_accuracy)
                    accuracy_summary.value.add(tag='Validation Accuracy - Top-5',simple_value=val_accuracy5)
                    train_writer.add_summary(accuracy_summary, global_step=counter)
                    print ('Validation accuracy: Top-1 %.4f \t Top-5 %.4f' % (val_accuracy, val_accuracy5))

                if (counter%(FLAGS.chpk_freq*num_batch_per_epoch_train) == 0):
                    # Save model
                    print("Periodically saving model...")
                    save_path = saver.save(sess, "./" + FLAGS.logdir + "/model.ckpt")

        except KeyboardInterrupt:
            print("Interupted... saving model.")
            save_path = saver.save(sess, "./" + FLAGS.logdir + "/model.ckpt-" + str(counter))


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
