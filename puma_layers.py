### PUMA Dense and conv layers for tiling simulation

import keras
import keras.backend as K
import tensorflow as tf
import numpy as np

### puma_dense definition
"""
    puma_dense(units, ..., input_slices=[[input1_begin, input1_size], [input2_begin, input2_size], ...], size=[], ...)
    input_slices: array of input slices [[start_slice_1, size_slice_1], [start_slice_2, size_slice_2], ...]
    kernel_slices: array of kernel slices [[start_slice_1, size_slice_1], [start_slice_2, size_slice_2], ...]
        
    All input slices and kernel slices must be specified. These must be informed in order.. meaning that input_slice_1 will be multiplied by kernel_slice_1.
    Latter, products using the same input slice are concatenated and te ones using different input slices are added.
    This is checked one by one with its adjacent. So, the order of inputs (and kernels) matters. 
    Products are done this way:
        prod1 = I1 X K1
        prod2 = I2 X K2
        prod2 = I3 X K3    
           
"""

class puma_dense(keras.layers.Dense):

    def __init__(self, input_slices=None, kernel_slices=None, concat_axis=1, **kwargs):
        super(puma_dense, self).__init__(**kwargs)
        self.i_slices = input_slices
        self.k_slices = kernel_slices
        self.concat_axis = concat_axis

    def build (self, input_shape):
        super(puma_dense, self).build(input_shape)

    def concat_inputs(self, slice_1, slice_2):

        if slice_1[0] > slice_2[0] :
            begin = slice_1[0]
        else:
            begin = slice_2[0]

        if slice_1[1] > slice_2[1]:
            size = slice_1[1]
        else:
            size = slice_2[1]

        return [begin, size]

    def call(self, inputs):
        n_slices = np.shape(self.i_slices)[0]
        result = []
        for i in range(n_slices):
            inputs_ = K.slice(inputs, self.i_slices[i][0], self.i_slices[i][1])
            kernel_ = K.slice(self.kernel, self.k_slices[i][0], self.k_slices[i][1])
            result.append(K.dot(inputs_, kernel_))

        inputs = self.i_slices
        while len(inputs) > 1:
            inputs_ = []
            result_ = []
            # print(inputs)
            i = 0
            while i <= (len(inputs) - 1):
                #print("i: {}".format(i))
                if i == len(inputs) - 1:
                    # print("Appending {}".format(i))
                    result_.append(result[i])
                    inputs_.append(inputs[i])
                else:
                    if inputs[i] != inputs[i + 1]:
                        # print("Adding {}".format(i))
                        result_.append(result[i] + result[i + 1])
                        inputs_.append(self.concat_inputs(slice_1=inputs[i], slice_2=inputs[i + 1]))
                    else:
                        # print("Concatenating {}".format(i))
                        result_.append(K.concatenate((result[i], result[i + 1]), axis=self.concat_axis))
                        inputs_.append(inputs[i])
                i = i + 2
            result = result_
            inputs = inputs_
        return K.bias_add(result[0], self.bias)
