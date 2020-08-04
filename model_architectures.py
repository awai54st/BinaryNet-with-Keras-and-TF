
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from tensorflow.python.framework import ops
from binarization_utils import *

batch_norm_eps=1e-4
batch_norm_alpha=0.1#(this is same as momentum)

def get_model(dataset,resid_levels):
	if dataset=='MNIST':
		model=Sequential()
		model.add(binary_dense(n_in=784,n_out=256,gamma=1.0,input_shape=[784],width=16.0,point=0.0,first_layer=True))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=256,gamma=1.0,width=16.0,point=0.0))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=256,gamma=1.0,width=16.0,point=0.0))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=256,gamma=1.0,width=16.0,point=0.0))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=10,gamma=1.0,width=16.0,point=0.0))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Activation('softmax'))

	elif dataset=="CIFAR-10" or dataset=="SVHN":
		model=Sequential()
		model.add(binary_conv(nfilters=64,ch_in=3,k=3,gamma=1.0,width=8.0,point=10.0,padding='valid',input_shape=[32,32,3],first_layer=True))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_conv(nfilters=64,ch_in=64,k=3,gamma=1.0,width=8.0,point=12.0,padding='valid'))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

		model.add(binary_conv(nfilters=128,ch_in=64,k=3,gamma=1.0,width=8.0,point=10.0,padding='valid'))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_conv(nfilters=128,ch_in=128,k=3,gamma=1.0,width=8.0,point=10.0,padding='valid'))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

		model.add(binary_conv(nfilters=256,ch_in=128,k=3,gamma=1.0,width=8.0,point=8.0,padding='valid'))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_conv(nfilters=256,ch_in=256,k=3,gamma=1.0,width=8.0,point=6.0,padding='valid'))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))

		model.add(my_flat())

		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=512,gamma=1.0,width=16.0,point=5.0))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=512,gamma=1.0,width=16.0,point=0.0))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=10,gamma=1.0,width=16.0,point=0.0))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Activation('softmax'))

	elif dataset=="binarynet":
		model=Sequential()
		model.add(binary_conv(nfilters=128,ch_in=3,k=3,gamma=1.0,width=8.0,point=10.0,padding='same',input_shape=[32,32,3],first_layer=True))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_conv(batch_size=100,width_in=32,ch_in=128,width=16.0,mu_point=8.0,var_point=8.0))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_conv(nfilters=128,ch_in=128,k=3,gamma=1.0,width=8.0,point=12.0,padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_conv(batch_size=100,width_in=16,ch_in=128,width=16.0,mu_point=8.0,var_point=8.0))
		model.add(Residual_sign(gamma=1.0))

		model.add(binary_conv(nfilters=256,ch_in=128,k=3,gamma=1.0,width=8.0,point=10.0,padding='same'))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_conv(batch_size=100,width_in=16,ch_in=256,width=16.0,mu_point=8.0,var_point=8.0))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_conv(nfilters=256,ch_in=256,k=3,gamma=1.0,width=8.0,point=10.0,padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_conv(batch_size=100,width_in=8,ch_in=256,width=16.0,mu_point=8.0,var_point=8.0))
		model.add(Residual_sign(gamma=1.0))

		model.add(binary_conv(nfilters=512,ch_in=256,k=3,gamma=1.0,width=8.0,point=8.0,padding='same'))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_conv(batch_size=100,width_in=8,ch_in=512,width=16.0,mu_point=8.0,var_point=8.0))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_conv(nfilters=512,ch_in=512,k=3,gamma=1.0,width=8.0,point=6.0,padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_conv(batch_size=100,width_in=4,ch_in=512,width=16.0,mu_point=8.0,var_point=8.0))
		model.add(Residual_sign(gamma=1.0))

		model.add(my_flat())

		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=1024,gamma=1.0,width=16.0,point=5.0))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_dense(batch_size=100,ch_in=1024,width=16.0,mu_point=8.0,var_point=4.0))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=1024,gamma=1.0,width=16.0,point=0.0))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_dense(batch_size=100,ch_in=1024,width=16.0,mu_point=8.0,var_point=4.0))
		model.add(Residual_sign(gamma=1.0))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=10,gamma=1.0,width=16.0,point=0.0))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(l1_batch_norm_mod_dense(batch_size=100,ch_in=10,width=16.0,mu_point=8.0,var_point=4.0))
		model.add(Activation('softmax'))
	elif dataset=="Imagenet":
		model=Sequential()

		model.add(binary_conv(nfilters=64,ch_in=3,k=11,strides=(4,4),padding='valid',input_shape=[3,224,224]))
		model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
		model.add(BatchNormalization(axis=1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels))

		model.add(binary_conv(nfilters=192,ch_in=64,k=5,padding='valid'))
		model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
		model.add(BatchNormalization(axis=1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels))


		model.add(binary_conv(nfilters=384,ch_in=192,k=3,padding='same'))
		model.add(BatchNormalization(axis=1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels))

		model.add(binary_conv(nfilters=256,ch_in=384,k=3,padding='same'))
		model.add(BatchNormalization(axis=1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels))

		model.add(binary_conv(nfilters=256,ch_in=256,k=3,padding='same'))
		model.add(BatchNormalization(axis=1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels))


		model.add(Flatten())

		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=4096))
		model.add(BatchNormalization(axis=1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels))
		#model.add(Dropout(0.5))
		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=4096))
		model.add(BatchNormalization(axis=1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(levels=resid_levels))
		#model.add(Dropout(0.5))

		model.add(binary_dense(n_in=int(model.output.get_shape()[1]),n_out=1000))
		model.add(BatchNormalization(axis=1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		#model.add(Dropout(0.5))

		model.add(Activation('softmax'))
	else:
		raise("dataset should be one of the following list: [MNIST, CIFAR-10, SVHN, Imagenet].")
	return model
