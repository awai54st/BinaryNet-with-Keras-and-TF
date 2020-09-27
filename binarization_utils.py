import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import functools

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D,Cropping2D
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.engine.topology import Layer
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.layers.normalization import BatchNormalization
from tensorflow.python.framework import ops
#from multi_gpu import make_parallel


def FXP_quantize(
        value, point, width):
    point = K.cast(point, 'float32')
    width = K.cast(width, 'float32')
    maximum_value = K.cast(2 ** (width - 1), 'float32')

    # x << (width - point)
    shift = 2.0 ** (K.round(width) - K.round(point))
    value_shifted = value * shift
    # quantize
    value_shifted = K.round(value_shifted)
    value_shifted = tf.clip_by_value(value_shifted, -maximum_value, maximum_value - 1)
    # revert bit-shift earlier
    return value_shifted / shift, point, width

def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def LOG_quantize(
        value, width):
    sign = K.sign(value + 1e-45) # Adding a small bias to remove zeros
    value = log2(K.abs(value))
    # quantize
    value, __, __ = FXP_quantize(
        value, width, width)
    # represent
    zero = tf.constant(0, dtype=tf.float32)
    return sign * (2 ** value)

def binarize_weight(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,-1,1)
    rounded = K.sign(clipped)
    return clipped + K.stop_gradient(rounded - clipped)

def binarize_activation(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,-1,1)
    rounded = K.sign(clipped)
    return clipped + K.stop_gradient(rounded - clipped)

def tf_custom_gradient_method(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, '_tf_custom_gradient_wrappers'):
            self._tf_custom_gradient_wrappers = {}
        if f not in self._tf_custom_gradient_wrappers:
            self._tf_custom_gradient_wrappers[f] = tf.custom_gradient(lambda *a, **kw: f(self, *a, **kw))
        return self._tf_custom_gradient_wrappers[f](*args, **kwargs)
    return wrapped

class l1_batch_norm_mod_conv(Layer):
    def __init__(self,batch_size,width_in,ch_in,width,mu_point,var_point,momentum, **kwargs):
        super(l1_batch_norm_mod_conv, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.width_in = width_in
        self.ch_in = ch_in
        self.width = width
        self.mu_point = mu_point
        self.var_point = var_point
        self.momentum = momentum

    def build(self, input_shape):
        super(l1_batch_norm_mod_conv, self).build(input_shape)  # Be sure to call this at the end
        beta = np.zeros([self.ch_in])*1.0
        self.beta=K.variable(beta)
        self.trainable_weights=[self.beta]
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=[1,1,1,self.ch_in],
            initializer=tf.zeros_initializer(),
            trainable=False)
        self.moving_var = self.add_weight(
            name='moving_var',
            shape=[1,1,1,self.ch_in],
            initializer=tf.initializers.ones(),
            trainable=False)

    def call(self, x):
        # Check if training or inference
        training = K.learning_phase()

        N = self.batch_size*self.width_in*self.width_in
        self.mu = 1./N * K.sum(x, axis = [0,1,2])
        self.mu = K.reshape(self.mu,[1,1,1,-1])
        xmu = x - self.mu
        self.var = 1./N * K.sum(K.abs(xmu), axis = [0,1,2])
        self.var = K.reshape(self.var,[1,1,1,-1])

        mu_fp16 = tf.cast(self.mu, tf.float16)
        self.mu = tf.cast(mu_fp16, tf.float32)
        var_fp16 = tf.cast(self.var, tf.float16)
        self.var = tf.cast(var_fp16, tf.float32)

        mean_update = tf.cond(training,
            lambda:K.moving_average_update(self.moving_mean,
            self.mu, self.momentum),
            lambda:self.moving_mean)
        var_update = tf.cond(training,
            lambda:K.moving_average_update(self.moving_var,
            self.var, self.momentum),
            lambda:self.moving_var)
        self.add_update([mean_update, var_update])

        beta_fp16 = tf.cast(self.beta, tf.float16)
        self.beta = tf.cast(beta_fp16, tf.float32)

        return self.quantise_gradient_op(x) + K.reshape(self.beta, [1,1,1,-1])


    @tf_custom_gradient_method
    def quantise_gradient_op(self, x):

        # Check if training or inference
        training = K.learning_phase()

        if training in {0, False}:
            self.mu = self.moving_mean
            self.var = self.moving_var

        xmu = x - self.mu

        # quantise var
        #var = LOG_quantize(var, 8.0)

        ivar = 1./self.var

        result = xmu * ivar


        def custom_grad(dy):

            dy_fp16 = tf.cast(dy, tf.float16)
            dy = tf.cast(dy_fp16, tf.float32)

            N = self.batch_size*self.width_in*self.width_in

            dy_norm_x = dy * ivar
            term_1 = dy_norm_x - K.reshape(1.0/N * K.sum(dy_norm_x, axis=[0,1,2]), [1,1,1,-1])
            term_2 = K.sign(result)
            #term_3 = 1.0/N * K.sum(dy_norm_x * result, axis=[0,1,2])
            term_3 = 1.0/N * K.sum(dy_norm_x * K.sign(result) * K.reshape(1.0/N * K.sum(K.abs(result), axis=[0,1,2]), [1,1,1,-1]), axis=[0,1,2])
            term_3 = K.reshape(term_3, [1,1,1,-1])
            dx = term_1 - term_2 * term_3

            return dx

        return result, custom_grad

    def get_output_shape_for(self,input_shape):
        return input_shape
    def compute_output_shape(self,input_shape):
        return input_shape

class l1_batch_norm_mod_dense(Layer):
    def __init__(self,batch_size,ch_in,width,mu_point,var_point,momentum, **kwargs):
        super(l1_batch_norm_mod_dense, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.ch_in = ch_in
        self.width = width
        self.mu_point = mu_point
        self.var_point = var_point
        self.momentum = momentum

    def build(self, input_shape):
        super(l1_batch_norm_mod_dense, self).build(input_shape)  # Be sure to call this at the end
        beta = np.zeros([self.ch_in])*1.0
        self.beta=K.variable(beta)
        self.trainable_weights=[self.beta]
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=[1,self.ch_in],
            initializer=tf.zeros_initializer(),
            trainable=False)
        self.moving_var = self.add_weight(
            name='moving_var',
            shape=[1,self.ch_in],
            initializer=tf.initializers.ones(),
            trainable=False)

    def call(self, x):

        # Check if training or inference
        training = K.learning_phase()

        N = self.batch_size
        self.mu = 1./N * K.sum(x, axis = 0)
        self.mu = K.reshape(self.mu,[1,-1])
        xmu = x - self.mu
        self.var = 1./N * K.sum(K.abs(xmu), axis = 0)
        self.var = K.reshape(self.var,[1,-1])

        mu_fp16 = tf.cast(self.mu, tf.float16)
        self.mu = tf.cast(mu_fp16, tf.float32)

        var_fp16 = tf.cast(self.var, tf.float16)
        self.var = tf.cast(var_fp16, tf.float32)

        mean_update = tf.cond(training,
            lambda:K.moving_average_update(self.moving_mean,
            self.mu, self.momentum),
            lambda:self.moving_mean)
        var_update = tf.cond(training,
            lambda:K.moving_average_update(self.moving_var,
            self.var, self.momentum),
            lambda:self.moving_var)
        self.add_update([mean_update, var_update])

        beta_fp16 = tf.cast(self.beta, tf.float16)
        self.beta = tf.cast(beta_fp16, tf.float32)

        return self.quantise_gradient_op(x) + K.reshape(self.beta, [1,-1])

    @tf_custom_gradient_method
    def quantise_gradient_op(self, x):

        # Check if training or inference
        training = K.learning_phase()

        if training in {0, False}:
            self.mu = self.moving_mean
            self.var = self.moving_var

        xmu = x - self.mu

        # quantise var
        #var = LOG_quantize(var, 8.0)

        ivar = 1./self.var

        result = xmu * ivar

        def custom_grad(dy):

            dy_fp16 = tf.cast(dy, tf.float16)
            dy = tf.cast(dy_fp16, tf.float32)

            N = self.batch_size

            dy_norm_x = dy * ivar
            term_1 = dy_norm_x - K.reshape(1.0/N * K.sum(dy_norm_x, axis=0), [1,-1])
            term_2 = K.sign(result)
            #term_3 = 1.0/N * K.sum(dy_norm_x * result, axis=0)
            term_3 = 1.0/N * K.sum(dy_norm_x * K.sign(result) * K.reshape(1.0/N * K.sum(K.abs(result), axis=0), [1,-1]), axis=0)
            term_3 = K.reshape(term_3, [1,-1])
            dx = term_1 - term_2 * term_3

            return dx

        return result, custom_grad

    def get_output_shape_for(self,input_shape):
        return input_shape
    def compute_output_shape(self,input_shape):
        return input_shape

class Residual_sign(Layer):
    def __init__(self, gamma=1,**kwargs):
        self.width = 16.0
        self.point = -2.0
        self.gamma=gamma
        super(Residual_sign, self).__init__(**kwargs)
    def call(self, x, mask=None):
        out_bin = binarize_activation(x)*K.abs(self.gamma)
        return out_bin

    def get_output_shape_for(self,input_shape):
        return input_shape
    def compute_output_shape(self,input_shape):
        return input_shape


def switch(condition, t, e):
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.where(condition, t, e)
    elif K.backend() == 'theano':
        import theano.tensor as tt
        return tt.switch(condition, t, e)

#stochastic_matmul_grad_gpu_so = '/home/ew913/BNN-FXP-TRAINING-XNOR-COPY/training-software/MNIST-CIFAR-SVHN/stochastic_matmul_grad_gpu.so'
#stochastic_matmul_grad_gpu_module = tf.load_op_library(stochastic_matmul_grad_gpu_so)

class binary_conv(Layer):
	def __init__(self,nfilters,ch_in,k,padding,gamma,width,point,strides=(1,1),first_layer=False,**kwargs):
		self.nfilters=nfilters
		self.ch_in=ch_in
		self.k=k
		self.padding=padding
		self.strides=strides
		self.width = width
		self.point = point
		self.gamma=gamma
		self.first_layer = first_layer
		super(binary_conv,self).__init__(**kwargs)
	def build(self, input_shape):
		stdv=1/np.sqrt(self.k*self.k*self.ch_in)
		w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
		if keras.backend._backend=="mxnet":
			w=w.transpose(3,2,0,1)
		self.w=K.variable(w)
		self.trainable_weights=[self.w]


	def call(self, x,mask=None):
		# Quantise gradients to float16
		self.out = self.xnor_wg_conv_op(x,binarize_weight(self.fp16_grad(self.w)))

		#self.out=K.conv2d(x, kernel=binarize_weight(self.w), padding=self.padding,strides=self.strides )
		#self.out = self.xnor_wg_conv_op(x,binarize_weight(self.w))
		self.output_dim=self.out.get_shape()
		return self.out

	@tf_custom_gradient_method
	def xnor_wg_conv_op(self, x, w):
		result=K.conv2d(x, kernel=w, padding=self.padding,strides=self.strides )
		def custom_grad(dy):

			w_reversed = K.reverse(w, [0,1])
			w_reversed = tf.transpose(w_reversed, [0,1,3,2])

			# Valid
			dy_pad_number = tf.constant([[0, 0,], [2, 2,],[2, 2,], [0, 0]])
			dy_padded = tf.pad(dy, dy_pad_number)
			# if po2 dx
			dy_padded_max = K.max(K.abs(dy_padded))
			dy_padded_bias = -K.round(log2(dy_padded_max)) + 8 # 8 = 2^4/2
			dy_padded = dy_padded * (2**(dy_padded_bias))
			dy_padded = LOG_quantize(dy_padded, 4.0)
			dy_padded = dy_padded * (2**(-dy_padded_bias))

			dx = K.conv2d(dy_padded, kernel=w_reversed, padding="valid",strides=self.strides )

			if self.padding == "same":
				# Undo padding
				dx = Cropping2D(cropping=((1, 1), (1, 1)))(dx)
				# Pad x
				x_pad_number = tf.constant([[0, 0,], [1, 1,],[1, 1,], [0, 0]])
				x_trans = tf.transpose(tf.pad(x, x_pad_number), [3,1,2,0])
			elif self.padding == "valid":
				x_trans = tf.transpose(x, [3,1,2,0])
			dy_trans = tf.transpose(dy, [1,2,0,3])

			# Shift
			dy_trans_max = K.max(K.abs(dy_trans))
			dy_trans_bias = -K.round(log2(dy_trans_max)) + 8
			dy_trans = dy_trans * (2**(dy_trans_bias))
			dy_trans = LOG_quantize(dy_trans, 4.0)
			dy_trans = dy_trans * (2**(-dy_trans_bias))

			# Ternary dy
			ones = K.ones_like(dy_trans)
			zeros = K.zeros_like(dy_trans)

			if self.first_layer:
				dw = K.conv2d(x_trans, kernel=dy_trans, padding="valid",strides=self.strides )
			else:
				x_trans = K.sign(x_trans + 1e-16) # Adding bias 1e-8 to sign function
				x_patches = tf.extract_image_patches(x_trans,
					[1, self.output_dim[1], self.output_dim[2], 1],
					[1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1],
					padding="VALID")
				# CUDA impl
				#dw = stochastic_matmul_grad_gpu_module.stochastic_matmul_grad_gpu(tf.reshape(x_patches, [tf.shape(w)[2]*tf.shape(w)[0]*tf.shape(w)[1], tf.shape(dy)[0]*tf.shape(dy)[1]*tf.shape(dy)[2]]), tf.reshape(dy_trans, [-1, self.nfilters]))
				#dw = tf.reshape(dw, [tf.shape(w)[2], tf.shape(w)[0], tf.shape(w)[1], tf.shape(w)[3]])

				# Keras conv impl
				dw = K.conv2d(x_trans, kernel=dy_trans, padding="valid",strides=self.strides )
			dw = tf.transpose(dw, [1,2,0,3])

			N = self.k*self.k*self.ch_in*self.nfilters
			dw = 1./np.sqrt(N) * K.sign(dw + 1e-16) # Adding bias 1e-8 to sign function
		
			return (dx, dw)
	
		return result, custom_grad

	@tf_custom_gradient_method
	def fp16_grad(self, w):
		result= w
		def custom_grad(dy):
			dy_fp16 = tf.cast(dy, tf.float16)
			dy_fp32 = tf.cast(dy_fp16, tf.float32)
			return dy_fp32
	
		return result, custom_grad


	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])

#def stochastic_po2_element_wise_add(x, y):
#
#	# swap x and y elementwise, such that x is "bigger_operand" and y is "smaller_operand"
#	
#	bigger_operand = tf.where(K.abs(x) >= K.abs(y), x, y)
#	smaller_operand = tf.where(K.abs(x) >= K.abs(y), y, x)
#	
#	# flip a coin between z = -zmin and z = +zmin
#	z = K.sign(tf.random_uniform(shape=tf.shape(x), minval=-1., maxval=1.)) * (2**(-64))
#	
#	rng_uniform = tf.random_uniform(shape=tf.shape(x), minval=0., maxval=1.)
#	
#	# case 1: x > 0, y > 0, return z = 2*x with probability y/x and z = x with probability 1-y/x.
#	
#	case_1_cond = tf.logical_and(tf.logical_and(bigger_operand > 0, smaller_operand > 0), tf.abs(bigger_operand) != tf.abs(smaller_operand))
#	z = tf.where(case_1_cond, bigger_operand + bigger_operand * (tf.where((smaller_operand/bigger_operand) > rng_uniform, tf.ones_like(x), tf.zeros_like(x))), z)
#	
#	# case 2: x > 0, y < 0, return z = x/2 with probability -2y/x and z = x with probability 1+2y/x
#	
#	case_2_cond = tf.logical_and(tf.logical_and(bigger_operand > 0, smaller_operand < 0), tf.abs(bigger_operand) != tf.abs(smaller_operand))
#	z = tf.where(case_2_cond, bigger_operand - (bigger_operand/2) * (tf.where((-2.0*smaller_operand/bigger_operand) > rng_uniform, tf.ones_like(x), tf.zeros_like(x))), z)
#	
#	# case 3: x < 0, y < 0, same with case 1 and flip z sign
#	
#	case_3_cond = tf.logical_and(tf.logical_and(bigger_operand < 0, smaller_operand < 0), tf.abs(bigger_operand) != tf.abs(smaller_operand))
#	z = tf.where(case_3_cond, bigger_operand + bigger_operand * (tf.where((smaller_operand/bigger_operand) > rng_uniform, tf.ones_like(x), tf.zeros_like(x))), z)
#	
#	# case 4: x < 0, y > 0, return z = x/2 with probability -2y/x and z = x with probability 1+2y/x
#	
#	case_4_cond = tf.logical_and(tf.logical_and(bigger_operand < 0, smaller_operand > 0), tf.abs(bigger_operand) != tf.abs(smaller_operand))
#	z = tf.where(case_4_cond, bigger_operand - (bigger_operand/2) * (tf.where((-2.0*smaller_operand/bigger_operand) > rng_uniform, tf.ones_like(x), tf.zeros_like(x))), z)
#	
#	# case 5: x = y, then z = 2x
#	
#	case_5_cond = bigger_operand == smaller_operand
#	z = tf.where(case_5_cond, 2 * bigger_operand, z)
#	
#	# case 6: x = -y, as then you want z = 0 which is unrepresentable. So perhaps flip a coin between z = -zmin and z = +zmin
#
#
#	return z
#
#def stochastic_po2_dot(a, b, unstack_num):
#
#	a_shape = tf.shape(a)
#	a_transposed = tf.reshape(a, [a_shape[0], a_shape[1], 1])
#	a_transposed = tf.transpose(a_transposed, [0, 2, 1])
#
#	b_shape = tf.shape(b)
#	b_transposed = tf.reshape(b, [b_shape[0], b_shape[1], 1])
#	b_transposed = tf.transpose(b_transposed, [2, 1, 0])
#
#	tmp = tf.multiply(a_transposed, b_transposed)
#
#	tmp_unpacked = tf.unstack(tmp, axis=2, num=unstack_num) # defaults to axis 0, returns a list of tensors
#
#	c = K.sign(tf.random_uniform(shape=tf.shape(tmp_unpacked[0]), minval=-1., maxval=1.)) * (2**(-128))
#
#	for tmp_slice in tmp_unpacked:
#		c = stochastic_po2_element_wise_add(c,tmp_slice)
#
#	return c

class binary_dense(Layer):
	def __init__(self,n_in,n_out,gamma,width,point,first_layer=False,**kwargs):
		self.n_in=n_in
		self.n_out=n_out
		self.width = width
		self.point = point
		self.gamma=gamma
		self.first_layer = first_layer
		super(binary_dense,self).__init__(**kwargs)
	def build(self, input_shape):
		stdv=1/np.sqrt(self.n_in)
		w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
		self.w=K.variable(w)
		self.trainable_weights=[self.w]

	def call(self, x,mask=None):
		# Cast weights to float16
		self.out = self.xnor_wg_dense_op(x,binarize_weight(self.fp16_grad(self.w)))

		#self.out = self.xnor_wg_dense_op(x,binarize_weight(self.w))
		#self.out = K.dot(x,binarize_weight(self.w))
		return self.out

	@tf_custom_gradient_method
	def xnor_wg_dense_op(self, x, w):
		result=K.dot(x, w)
		def custom_grad(dy):

			# Shift
			dy_max = K.max(K.abs(dy))
			dy_bias = -K.round(log2(dy_max)) + 8
			dy = dy * (2**(dy_bias))
			dy = LOG_quantize(dy, 4.0)
			dy = dy * (2**(-dy_bias))

			dx = K.dot(dy, K.transpose(w))

			# Stochastic GPU impl
			if self.first_layer == True:
				#dw = stochastic_matmul_grad_gpu_module.stochastic_matmul_grad_gpu(K.transpose(x), dy)
				dw = K.dot(K.transpose(x), dy)
			else:
				#dw = stochastic_matmul_grad_gpu_module.stochastic_matmul_grad_gpu(K.transpose(K.sign(x + 1e-16)), dy)
				dw = K.dot(K.transpose(K.sign(x + 1e-16)), dy) # Adding bias 1e-8 to sign function
			
			N = self.n_in*self.n_out
			dw = 1./np.sqrt(N) * K.sign(dw + 1e-16) # Adding bias 1e-8 to sign function

			return (dx, dw)
	
		return result, custom_grad

	@tf_custom_gradient_method
	def fp16_grad(self, w):
		result= w
		def custom_grad(dy):
			dy_fp16 = tf.cast(dy, tf.float16)
			dy_fp32 = tf.cast(dy_fp16, tf.float32)
			return dy_fp32
	
		return result, custom_grad

	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.n_out)
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.n_out)

class my_flat(Layer):
	def __init__(self,**kwargs):
		super(my_flat,self).__init__(**kwargs)
	def build(self, input_shape):
		return

	def call(self, x, mask=None):
		self.out=tf.reshape(x,[-1,np.prod(x.get_shape().as_list()[1:])])
		return self.out
	def  compute_output_shape(self,input_shape):
		shpe=(input_shape[0],int(np.prod(input_shape[1:])))
		return shpe
