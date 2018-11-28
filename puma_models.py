#### APIs to model different non-idealities in PUMA forward/backward pass
# tf.cast and tf.stack take up too much memory

import tensorflow as tf
import numpy as np

### Non-ideality models and computation involved in memristor writes
class nonideality ():
    # non-ideality models
    def __init__ (self, sigma, alpha):
        self.write_noise_sigma = sigma
        self.write_nonlinearlity_alpha = alpha
        print("nonideality sigma: " + str(sigma))
        print("nonideality alpha: " + str(alpha))

    # model write noise
    def _compute_noise (self, grad_ideal, weights, name=None):
        # name_scope groups vars and ops  within a scope for easy visualization in tensorboard
        with tf.name_scope(name, "comp_noise"):
            # find stddev of noise based on range of weights
            weight_range = tf.reduce_max(weights) - tf.reduce_min(weights)
            noise_stddev = (self.write_noise_sigma * tf.sqrt(weight_range)) * tf.sqrt(tf.abs(grad_ideal)) #only interested in magnitude of updates

            # generate noise
            write_noise = tf.random_normal(tf.shape(grad_ideal), mean=0, stddev=noise_stddev, name="write_noise_gen")
            write_noise = write_noise * tf.cast((grad_ideal!=0),tf.float32) # noise is zero for zero grad
            return write_noise

    # model write asymmetric non_linearity
    def _compute_nonlinearity (self, grad_ideal, weights, name=None):
        # name_scope groups vars and ops  within a scope for easy visualization in tensorboard
        with tf.name_scope(name, "comp_nonlin"):
            # compute delta
            w_max_temp = tf.reduce_max(weights)
            w_min = tf.reduce_min(weights)
            w_max = tf.cond(tf.equal(w_min,w_max_temp), lambda: (w_min+1.0), lambda: (w_max_temp))
            weight_range = w_max-w_min
            delta = grad_ideal / weight_range

            w0 = weight_range / (1.0 - tf.exp(-self.write_nonlinearlity_alpha))

            # compute positive updates
            temp = (-self.write_nonlinearlity_alpha) * delta
            temp = 1.0 - tf.exp(temp)
            update_pos = (w0+w_min) - weights
            update_pos *= temp
            update_pos = update_pos * tf.cast((delta >= 0.0), tf.float32) # cast to zero for negative delta values

            # compute negative weights
            temp = (self.write_nonlinearlity_alpha) * delta
            temp = 1.0 - tf.exp(temp)
            update_neg = (w_max-w0) - weights
            update_neg *= temp
            update_neg = update_neg * tf.cast((delta < 0.0), tf.float32) # cast to zero for positive delta values

            # merge positive and negative updates
            grad_nonideal = update_pos + update_neg
            return [grad_nonideal, weight_range]

    # input is a list of tuples (grad, var) as returned by tf.optimizer.compute_gradient()
    # returns updated grad (delta_W with non-ideality) - 1. non-linearity, 2. assymettry, 3. write noise
    def apply (self, grads, name=None):
        # name_scope groups vars and ops  within a scope for easy visualization in tensorboard
        with tf.name_scope(name, "puma_non_ideality"):
            grads_nonideal = []
            weight_range = []
            for pair in grads: # grads represents (gradient, weight)
                gradient_ideal = pair[0]
                weights = pair[1]

                # apply non-ideality
                gradient_nonideal1 = self._compute_nonlinearity (gradient_ideal, weights)
                gradient_noise = self._compute_noise (gradient_nonideal1[0], weights)
                gradient_nonideal = gradient_ideal + gradient_noise

                # pack grad, weights
                pair_nonideal = (gradient_nonideal, weights)
                grads_nonideal.append (pair_nonideal)
                weight_range.append(gradient_nonideal1[1])

            # compute summary (difference in ideal and non-ideal normalized with weight_range)
            grad_ideal_list = [grad for grad, var in grads]
            grad_nonideal_list = [grad for grad, var in grads_nonideal]
            grad_diff = [((tf.abs(grad_ideal_list[i]-grad_nonideal_list[i]))/weight_range[i]) for i in range(len(weight_range))]
            grad_diff_mean = tf.stack([tf.reduce_mean(grad_diff_tensor) for grad_diff_tensor in grad_diff])
            tf.summary.scalar("grad_diff_mean",tf.reduce_mean(grad_diff_mean,name="grad_diff_mean"))

            # log summary of weight_range and grad_ideal(normalized with weight_range)
            tf.summary.scalar("weight_range_mean",tf.reduce_mean(tf.stack(weight_range)))
            grad_norm = [(tf.abs(grad_ideal_list[i])/weight_range[i]) for i in range(len(weight_range))]
            grad_norm_mean = tf.stack([tf.reduce_mean(grad_norm_tensor) for grad_norm_tensor in grad_norm])
            tf.summary.scalar("grad_ideal_norm_mean",tf.reduce_mean(grad_norm_mean))

            return [grads_nonideal, grad_diff_mean]


### quantize an input tensor based on dynamic quantization
# for n-bit fixed point have 2^n levels of data from w_min to w_max
def _quantize (inp, precision, name=None):
    # name_scope groups vars and ops  within a scope for easy visualization in tensorboard
    with tf.name_scope(name, "quantize_op"):
        # compute variable sused in quantization loop body
        min_val = tf.reduce_min(inp)
        max_val = tf.reduce_max(inp)
        grad_range = tf.subtract (max_val, min_val, name="grad_range")
        num_levels = tf.subtract(tf.pow (2.0, precision), 1, name="num_levels")
        step_size = tf.divide (grad_range, num_levels, name="step_size") # min + (num_levels-1)*step_size = max

        # quantize inp tensor
        inp_fixed = tf.round((inp - min_val) / step_size) # tf supports broadcasting scalars (tensor op scalar)
        inp_quant = tf.add(min_val, inp_fixed*step_size, name="tensor_quantized")
        return inp_quant


### PUMA layers' class defnition based on 1) Fixed-Point 2) Weight-sliced and 3) Bit-streamed computation
class puma_layers ():
    def __init__ (self, name, precision=16):
        self.name = name + "_puma"
        self.precision = precision # number of bits use dfor fixed point representation
        # Under construction - wt_slicing, bit-slicing and ADC parameters
        # these make sense only when matrix is partitioned into crossbars
        # current impl. only considers fixed-point quantization of weights and inputs
        self.bit_per_device = 2
        self.dac_res = 1
        self.adc_res = 8

### puma-dense definition
class dense (puma_layers):
    def __init__ (self, inputs, units, name, precision=16):
        # compute with quantized weights and inputs
        super().__init__(name, precision)

        # define layer
        self.layer = tf.layers.Dense(name=self.name, units=units, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.layer.build([None, inputs.shape[1]]) #layer build sets up the variable inside class - kernel for eg.

    def apply (self, inputs):
        with tf.name_scope(None, "puma_dense"): # name modified dense layers as puma_dense
            # quantize input
            inp_quant = _quantize(inputs, self.precision, "inp_quant")

            # quantize kernel
            with tf.variable_scope(self.name, reuse=True):
                w = tf.get_variable('kernel') # "reuse=True" creates a reference/pointer to the kernel of dense layer
                kernel_quant = _quantize(w, self.precision, "wt_quant")
                kernel_quantize = tf.assign(w, kernel_quant) # op to quantize kernel

            # set control dep - output should be computed after kernel_qunatization
            with tf.control_dependencies([kernel_quantize]): #add in a list to make input_arg iterable
                return self.layer.apply(inp_quant)

    ## for DEBUG - validate output of apply function (Essentially, validates if control-dep above works correctly)
    def validate (self, inputs):
        # expected output - golden output
        inp_quant = _quantize(inputs, self.precision, "inp_quant")
        with tf.variable_scope(self.name, reuse=True):
            w = tf.get_variable('kernel')
            b = tf.get_variable('bias')
            kernel_quant = _quantize(w, self.precision, "wt_quant")

        out_full_prec = tf.add(tf.matmul(inp_quant, w), b) # output with full-precision kernel
        out_exp = tf.add(tf.matmul(inp_quant, kernel_quant), b) # output with manual impl. of kernel_quant
        out_actual = self.apply(inputs) # output from apply function

        # see these for validation:
        # out_exp and out_actual should be same
        # w1 and kernel_quant should be same (Note: assignment should have happened until this point due to control dep.)
        with tf.variable_scope(self.name, reuse=True):
            w1 = tf.get_variable('kernel') # reference/pointer to the kernel of dense layer
        return [out_full_prec, out_exp, out_actual, w1, kernel_quant]


### puma-conv2d definition
class conv2d (puma_layers):
    # creates an op to cpmpute with quantized weights and inputs
    def __init__ (self, inputs, filters, kernel_size, name, precision=16):
        super().__init__(name, precision)

        # define layer
        self.layer = tf.layers.Conv2D(name=self.name, filters=filters, kernel_size=kernel_size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME', use_bias=True)
        self.layer.build([None, None, None, inputs.shape[3]]) #layer build sets up the variable inside class - kernel for eg.

    def apply (self, inputs):
        with tf.name_scope(None, "puma_conv2d"): # name modified conv2d layers as puma_conv2d
            # quantize input
            inp_quant = _quantize(inputs, self.precision, "inp_quant")

            # quantize kernel
            with tf.variable_scope(self.name, reuse=True):
                w = tf.get_variable('kernel')
                kernel_quant = _quantize(w, self.precision, "wt_quant")
                kernel_quantize = tf.assign(w, kernel_quant) # op to quantize kernel

            # set control dep - output should be computed after kernel_qunatization
            with tf.control_dependencies([kernel_quantize]): #add in a list to make input_arg iterable
                return self.layer.apply(inp_quant)

    ## for DEBUG - validate output of apply function (Essentially, validates if control-dep ablove works correctly)
    def validate (self, inputs):
        assert False , "validate function for conv2d deosn't exist"


### convert from float to sliced-representation (NOTE: sliced part represents fixed-part of float data)
def _slice (data, num_slice, step_size, min_val, precision):
    with tf.name_scope(None, "slice_op"):
        if (min_val != None): # for matrix
            data_fixed = tf.cast(tf.round((data - min_val) / step_size), dtype=tf.int32) # fixed_point repr. (unsigned)
        else:
            # for grad (grad represents change, hence represented with same step size as matrix)
            # clip the gradeints to contain within precision (NOTE: this isimportant, as step_size is set by weights and not grad)
            data_fixed_tmp = tf.cast(tf.round(data / step_size), dtype=tf.int32) # fixed_point repr. (can be signed)
            data_fixed = tf.clip_by_value(data_fixed_tmp, 0, tf.pow(2, precision)-1)

        # list of slices starting from least significant slice
        data_sliced = tf.stack([tf.bitwise.right_shift(data_fixed,2*i)-(tf.bitwise.right_shift(data_fixed, 2*(i+1))*4) for i in range(num_slice)])
        #data_sliced_temp = tf.stack([tf.bitwise.right_shift(data_fixed,2*i) for i in range(num_slice+1)])
        #data_sliced = tf.stack([(data_sliced_temp[i]-data_sliced_temp[i+1]*4) for i in range(num_slice)])
        return data_sliced


### convert from sliced form to float
def _unslice (data_sliced, num_slice, step_size, min_val): #data_sliced should be list of slices
    with tf.name_scope(None, "unslice_op"):
        data_val_tensor = tf.stack([data_sliced[i]*tf.pow(2,2*i) for i in range(num_slice)])

        if (min_val != None): # for matrix
            data_float = min_val + step_size * tf.cast(tf.reduce_sum(data_val_tensor, axis=0), dtype=tf.float32)
        else:
            data_float = step_size * tf.cast(tf.reduce_sum(data_val_tensor, axis=0), dtype=tf.float32)
        return data_float


### define object for bit-sliced representation of delta-xbar data
class sliced_data ():
    # default state - each slice has 2-bits of value, 4-bits of carry
    # matrix in specifies the shape of sliced storage (self.data_sliced)
    def __init__ (self, matrix_in, name, precision=16, num_slice=8, slice_bits=6):
        #with tf.device("/device:GPU:0"):
        with tf.variable_scope(None, "Sliced_Data"):
            # state refers to min_val, max_val, step_size
            self.state = tf.get_variable ("State_"+name, [3], trainable=False, dtype=tf.float32)

            data_shape = matrix_in.get_shape()
            sliced_data_shape = [num_slice] + [data_shape[i].value for i in range(len(data_shape))]
            self.data_sliced = tf.get_variable("Sliced_"+name, sliced_data_shape, trainable=False, dtype=tf.int32)

    # write data - initialize at crs-sync points
    def write_data (self, matrix_in, precision, num_slice):
        with tf.name_scope(None, "write_slice"):
            # update min_val, step-size based on current data stored after crs-sync
            # add +1/-1 to max_val/min_val to allow a dynamic range for const. weight/bias matrix
            min_val_temp = tf.reduce_min(matrix_in) # min_val of matrix stored since last sync
            max_val_temp = tf.reduce_max(matrix_in) # max_val of matrix stored since last sync

            # adding case for constant data (otherwise step_size becomes 0, leading to divide by 0)
            min_val = tf.cond (tf.equal(max_val_temp, min_val_temp), lambda: (min_val_temp-1.0), lambda: (min_val_temp))
            max_val = tf.cond (tf.equal(max_val_temp, min_val_temp), lambda: (max_val_temp+1.0), lambda: (max_val_temp))
            step_size = tf.divide (max_val-min_val, (tf.pow(2.0,precision)-1)) # step-size of matrix stored since last sync
            curr_state = self.state.assign(tf.stack([min_val, max_val, step_size]))

            # update sliced storage after crs-sync
            data_sliced_tmp = _slice(matrix_in, num_slice, curr_state[2], curr_state[0], None)
            data_sliced_updated = self.data_sliced.assign(data_sliced_tmp)
            return data_sliced_updated # returns a write op for assigning sliced_data

    # update data_sliced with grad computed during back-prop and return data_loss factor for grad
    # precisely updating stored weight - not doing now - NEED TO HACK APPLY_GRADIENT
    # data_loss per slice increases with increasing (grad_slice+data_slice-slice_max)
    def update_data (self, grad_in, precision, num_slice, slice_min, slice_max):
        with tf.name_scope(None, "update_slice"):
            grad_sliced = _slice(tf.abs(grad_in), num_slice, self.state[2], None, precision)

            sign_map = 1*tf.cast(grad_in >= 0, tf.int32) + -1*tf.cast(grad_in < 0, tf.int32) # grad_sliced has signed-magnitude repr. (sign_map is per slice, all slices have same sign)
            data_updated = grad_sliced*sign_map + self.data_sliced # this is tensor (sign_map will get broadcast across all slices)

            ## compute loss_factor per slice
            ## positive grads
            #loss_factor_pos = (data_updated - slice_max)
            #loss_factor_pos = loss_factor_pos * tf.cast((loss_factor_pos >= 0), tf.int32) # cast to zero for non-saturating updates
            #loss_factor_pos = loss_factor_pos * tf.cast((sign_map > 0), tf.int32) # cast to zero for non-positive grads
            ## negative grads
            #loss_factor_neg = (data_updated - slice_min)
            #loss_factor_neg = loss_factor_neg * tf.cast((loss_factor_neg <= 0), tf.int32) # cast to zero for non-saturating updates
            #loss_factor_neg = loss_factor_neg * tf.cast((sign_map <= 0), tf.int32) # cast to zero for non-negative grads

            ## compute overall loss (positive and negative grads) and return updated grad
            #loss_factor = loss_factor_pos + loss_factor_neg
            #grad_updated_sliced = grad_sliced * sign_map - loss_factor
            #grad_updated = _unslice(grad_updated_sliced, num_slice, self.state[2], None)

            # clip based on slice_bits - update data_sliced and return grad_updated
            data_updated_clipped = tf.clip_by_value(data_updated, slice_min, slice_max)
            grad_updated_sliced = data_updated_clipped - self.data_sliced
            grad_updated = _unslice(grad_updated_sliced, num_slice, self.state[2], None)
            data_sliced_updated = self.data_sliced.assign(data_updated_clipped)

            # compute grad_updated

            return [data_sliced_updated, grad_updated]

    # for DEBUG - reconstruct the data from sliced_data
    def read_data (self, num_slice):
        with tf.name_scope(None, "read_slice"):
            data_out = _unslice(self.data_sliced, num_slice, self.state[2], self.state[0])
            return data_out


## PUMA outer-product storage (models device saturation errors, adds nonideality on quantized example-wise gradients)
class outer_product ():
    def __init__ (self, var_list, num_slice=8, precision=16, slice_bits=6, lr=0.01, sigma=0.01, alpha=0.01):
        with tf.variable_scope(None, "puma_sliced_data"):
            self.lr = lr
            self.num_slice = num_slice # number of slices used to store w/delta_w (depends on number of bits in mvm/vmm xbars)
            self.slice_bits = slice_bits # number of slice bits in delta xbars
            self.precision = precision
            self.slice_min = -1*tf.pow(2, self.slice_bits-1)
            self.slice_max = tf.pow(2, self.slice_bits-1)-1

            # instance of nonideality object for delta xbar
            self.nonideality = nonideality(sigma=sigma, alpha=alpha)

            # instance of sliced data objects for all trainable variables in model
            self.num_var = len(var_list)
            self.var_sliced_list = [sliced_data(matrix_in=var_list[i], name=(var_list[i].name).split(":")[0]) \
                    for i in range(self.num_var)] # list of trainable variables in sliced format

    # reset sliced storage of all trainable avriables at crs-sync (resets all carries)
    def crs_sync (self, var_list): # var_list is list of trainable variables
        with tf.name_scope(None, "puma_crs"):
            write_op = [self.var_sliced_list[i].write_data(matrix_in=var_list[i], \
                    precision=self.precision, num_slice=self.num_slice) for i in range(self.num_var)]
            return write_op #list of tensors

    # update the slice with grad_in
    def update (self, grad_in):
        with tf.name_scope(None, "op_comp"):
            grad_updated = [self.var_sliced_list[i].update_data(grad_in=grad_in[i], precision=self.precision, \
                    num_slice=self.num_slice, slice_min=self.slice_min, slice_max=self.slice_max) for i in range(self.num_var)]
            return grad_updated #list of list of tensors

    # for DEBUG - read all slices
    def read (self):
        with tf.name_scope(None, "read_puma_OP"):
            data_out =[self.var_sliced_list[i].read_data(num_slice=self.num_slice) for i in range(self.num_var)]
            return data_out #list of tensors

    # function to model gradient storage on crossbar
    #def _store_gradient (self, grad_ideal, grad_stored, batch_size):
    #    # input: grad_ideal - current gradeint (from 1 example) that needs to be stored on crossbar (scale, quantzie, non-ideal, model saturation)
    #    # input: grad_stored- previous state of crossbar

    #    # downscale gradients by batch_size and learning rate
    #    num_grad = len(grad_ideal)
    #    grad_scaledown = [(grad_ideal[i]*self.lr)/batch_size for i in range(num_grad)]

    #    # quantize gradients - NOT REQUIRED: update module already quantizes gradients based on weight range
    #    # grad_quant = [_quantize(grad_scaledown[i], self.precision, "grad_quant") for i in range(batch_size)]

    #    # model non-ideality
    #    grad_var_pair = zip(grad_scaledown, grad_stored)
    #    grad_var_nonideal = self.nonideality.apply(grad_var_pair)
    #    grad_nonideal = [grad for grad,var in grad_var_nonideal]

    #    # model losses in grad due to saturation in storage
    #    #grad_nonideal_updated = self.update(grad_nonideal)

    #    # add new gradient to previous accumulation over batch
    #    #out = [(grad_nonideal_updated[i]+grad_stored[i]) for i in range(num_grad)]
    #    out = [(grad_nonideal[i]+grad_stored[i]) for i in range(num_grad)]
    #    return out

    #def apply (self, loss_list, var_list, update_ops, batch_size):
    #    # input: loss_list - list of example-wise loss for a batch
    #    # input: var_list - list of trainable variables in graph
    #    # input: update_ops to specify control dependancy on gradient computation

    #    ## compute gradient_list
    #    with tf.control_dependencies(update_ops):
    #        grads_list = [tf.gradients(loss_list[i], var_list) for i in range(batch_size)]

    #    ## model device storage (non-ideality and saturation)
    #    grad_tmp = [tf.zeros(tf.shape(grads_list[0][i])) for i in range(len(grads_list[0]))]
    #    for i in range (batch_size):
    #        grad_tmp = self._store_gradient(grads_list[i], grad_tmp, batch_size)

    #    return zip(grad_tmp, var_list) #return var_val which can be directly used with apply_gradients

    # input is a list of tuples (grad, var) as returned by tf.optimizer.compute_gradient()
    # returns updated grad (delta_W with non-ideality) - 1. non-linearity, 2. assymettry, 3. write noise
    def apply_batch (self, grads, name=None):
        # name_scope groups vars and ops  within a scope for easy visualization in tensorboard
        with tf.name_scope(name, "puma_parallel_write"):
            #grad_var_nonideal = self.nonideality.apply(grads)

            #grad_nonideal = [grad for grad,var in grad_var_nonideal] #extract grads
            grad_nonideal = [grad for grad,var in grads] #extract grads
            grad_updated = self.update(grad_nonideal)

            var_sliced_out = [grad_var[0] for grad_var in grad_updated]
            grad_out = [grad_var[1] for grad_var in grad_updated]

            #var_in = [var for grad,var in grads]
            #grad_var_out = zip(grad_out, var_in)

            # compute saturation stats
            sat_stat = _get_saturation_stats_list (var_sliced_out, self.slice_max, self.slice_min)
            return [grad_out, sat_stat]


def _get_saturation_stats_var (var_sliced, slice_max, slice_min):
    with tf.name_scope(None, "puma_stats_var"):
        # find number of values at slice_min and slice_max
        sat_identifier_tensor = tf.cast(tf.logical_or(tf.equal(var_sliced, slice_max), tf.equal(var_sliced, slice_min)), dtype=tf.uint8)
        #return tf.count_nonzero(input_tensor=sat_identifier_tensor, dtype=tf.float16)/tf.size(input=sat_identifier_tensor, out_type=tf.float16)
        # return slice-wise stats
        return tf.stack([tf.count_nonzero(input_tensor=sat_identifier_tensor[i], dtype=tf.float16)/tf.size(input=sat_identifier_tensor[i], out_type=tf.float16) \
                for i in range(var_sliced.get_shape()[0])])


## NOTE: this function is required to make sure var_sliced_out propagates somewhere [else tensorflow deson't update daat_sliced]
## get saturation stats for slices
def _get_saturation_stats_list (var_sliced_list, slice_max, slice_min):
    with tf.name_scope(None, "puma_stats_list"):
        sat_identifier_tensor = tf.stack([_get_saturation_stats_var(var, slice_max, slice_min) for var in var_sliced_list])
        #return tf.reduce_mean(tf.stack(sat_identifier_list))
        mean_per_slice = tf.reduce_mean(input_tensor=sat_identifier_tensor, axis=0)
        global_mean = tf.reduce_mean(mean_per_slice)
        return [mean_per_slice, global_mean]


# ************************* Recycle-bin *****************************
# quantization appraoch with while-loop

# body of loop used in quantize
#def _body (inp_q, inp, min_val, step_size, i, iters):
#    x = inp[i:i+1]
#    print ("loop\n")
#    #x_q = tf.get_variable("quant_val", [1], trainable=False)
#    n = tf.round (tf.divide (tf.subtract (x, min_val), step_size))
#    x_q = tf.add (x, tf.multiply(n, step_size), name="quant_x")
#    return [tf.concat([inp_q, x_q], axis=0), inp, min_val, step_size, i+1, iters]
#
#def _cond (inp_q, inp, min_val, step_size, i, iters):
#    return i < iters

# condition of while loop - with lamda
#cond = lambda i, iters: tf.less(i, iters)

## quatize tensor
##inp_q = tf.get_variable(name="quant_variable", initializer=inp, trainable=False) # create a copy of original tensor
#inp_q = tf.get_variable(name="quant_variable", shape=[1], trainable=False) # create a copy of original tensor
#i = tf.constant(0)
#iters = tf.size(inp)
#body = lambda inp_q, inp, min_val, step_size, i, iters: _body (inp_q, inp, min_val, step_size, i, iters)
#c = lambda inp_q, inp, min_val, step_size, i, iters: _cond (inp_q, inp, min_val, step_size, i, iters)
#quant_op = tf.while_loop (c, body, loop_vars=[inp_q, inp, min_val, step_size, i, iters]
#                                    ,shape_invariants=[tf.TensorShape([None]), inp.get_shape(), min_val.get_shape(), step_size.get_shape(), i.get_shape(), iters.get_shape()])
#return inp_q

## pack together example-wise gradients without modifying for memristor nonideality or carry saturation
#grad_tmp_list=[]
#for i in range(len(var_list)):
#    temp_tensor = tf.reduce_mean(tf.stack([grad[i] for grad in grads_list]), axis=0)
#    grad_tmp_list.append(temp_tensor)
#return zip(grad_tmp_list, var_list)
