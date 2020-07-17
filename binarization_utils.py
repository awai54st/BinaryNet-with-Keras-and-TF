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

    sign = K.sign(value)
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


class Quantise_grad_layer(Layer):
    def __init__(self,width,point):
        super(Quantise_grad_layer, self).__init__()
        self.width = width
        self.point = point

    def build(self, input_shape):
        super(Quantise_grad_layer, self).build(input_shape)  # Be sure to call this at the end
        beta = np.zeros([5])*1.0
        self.beta=K.variable(beta)

    def call(self, x):
        return self.quantise_gradient_op(x)

    @tf_custom_gradient_method
    def quantise_gradient_op(self, x):
        result = x # do forward computation
        def custom_grad(dy):
            grad, self.point, self.width = FXP_quantize(dy, self.point, self.width)
            return grad
        return result, custom_grad


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
		self.out=K.conv2d(x, kernel=binarize_weight(self.w), padding=self.padding,strides=self.strides )
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
			#dy_padded = dy_padded * (2**(16))
			dy_padded = LOG_quantize(dy_padded, 8.0)
			#dy_padded = dy_padded * (2**(-16))
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
			#dy_trans = dy_trans * (2**(16))
			dy_trans = LOG_quantize(dy_trans, 8.0)
			#dy_trans = dy_trans * (2**(-16))

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


	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])

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
		#self.out = self.xnor_wg_dense_op(x,binarize_weight(self.w))
		self.out = K.dot(x,binarize_weight(self.w))
		return self.out

	@tf_custom_gradient_method
	def xnor_wg_dense_op(self, x, w):
		result=K.dot(x, w)
		def custom_grad(dy):

			# Shift
			#dy = dy * (2**(16))
			dy = LOG_quantize(dy, 8.0)
			#dy = dy * (2**(-16))

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
