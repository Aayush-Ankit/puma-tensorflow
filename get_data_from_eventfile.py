import tensorflow as tf

#purdue sim
path = '/local/scratch/a/aankit/tensorflow/approx_memristor/cifar100/vgg16_memristor_nonideality/puma_vgg16_slice_mixed_55554444_crsfreq_4/'
file_name = 'events.out.tfevents.1543592396.cbric-gpu3.ecn.purdue.edu'

# hpe sim
#path = '/local/scratch/a/aankit/tensorflow/approx_memristor/cifar100/vgg16_memristor_nonideality/simulationdata/puma_vgg16_slice_3bits_crsfreq_256/'
#file_name = 'events.out.tfevents.1543363584.deepsim-10.labs.hpecorp.net'

events_file_path = path + file_name

for e in tf.train.summary_iterator(events_file_path):
    for v in e.summary.value:
        if v.tag == 'Loss':
            print(v.simple_value)
