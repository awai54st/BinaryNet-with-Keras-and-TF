import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import math
import keras
from keras.datasets import cifar10,mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
sys.path.insert(0, '..')
from binarization_utils import *
from model_architectures import get_model

dataset='MNIST'
Train=True
Evaluate=True
Inspect=False
Inspect_fromscratch=False
batch_size=100
epochs=1000

def load_svhn(path_to_dataset):
	import scipy.io as sio
	train=sio.loadmat(path_to_dataset+'/train.mat')
	test=sio.loadmat(path_to_dataset+'/test.mat')
	extra=sio.loadmat(path_to_dataset+'/extra.mat')
	X_train=np.transpose(train['X'],[3,0,1,2])
	y_train=train['y']-1

	X_test=np.transpose(test['X'],[3,0,1,2])
	y_test=test['y']-1

	X_extra=np.transpose(extra['X'],[3,0,1,2])
	y_extra=extra['y']-1

	X_train=np.concatenate((X_train,X_extra),axis=0)
	y_train=np.concatenate((y_train,y_extra),axis=0)

	return (X_train,y_train),(X_test,y_test)

if dataset=="MNIST":
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# convert class vectors to binary class matrices
	X_train = X_train.reshape(-1,784)
	X_test = X_test.reshape(-1,784)
	use_generator=False
elif dataset=="CIFAR-10" or dataset=="binarynet":
	use_generator=True
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
elif dataset=="SVHN":
	use_generator=True
	(X_train, y_train), (X_test, y_test) = load_svhn('./svhn_data')
else:
	raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN].")

X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
X_train /= 255
X_test /= 255
X_train=2*X_train-1
X_test=2*X_test-1


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.025
	drop = 0.5
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

if Train:
	if not(os.path.exists('models')):
		os.mkdir('models')
	if not(os.path.exists('models/'+dataset)):
		os.mkdir('models/'+dataset)
	for resid_levels in range(1):
		print 'training with', resid_levels,'levels'
		sess=K.get_session()
		model=get_model(dataset,resid_levels,batch_size)
		#model.summary()

		#gather all binary dense and binary convolution layers:
		binary_layers=[]
		for l in model.layers:
			if isinstance(l,binary_dense) or isinstance(l,binary_conv):
				binary_layers.append(l)

		#gather all residual binary activation layers:
		resid_bin_layers=[]
		for l in model.layers:
			if isinstance(l,Residual_sign):
				resid_bin_layers.append(l)
		lr=0.001
		#opt = keras.optimizers.Adam(lr=lr,decay=1e-6)#SGD(lr=lr,momentum=0.9,decay=1e-5)
		opt = keras.optimizers.Adam(lr=lr)#SGD(lr=lr,momentum=0.9,decay=1e-5)
		#opt = keras.optimizers.SGD(lr=lr,momentum=0.9)
		#opt = keras.optimizers.SGD(lr=lr)
		#model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['sparse_categorical_accuracy'])
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])


		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		# learning rate scheduler callback
		#lrate = keras.callbacks.LearningRateScheduler(step_decay)
		lrate = keras.callbacks.ReduceLROnPlateau(
			monitor='val_acc', factor=0.5, patience=50, verbose=0, mode='auto',
			min_delta=0.0001, cooldown=0, min_lr=0
		)
		cback=keras.callbacks.ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True)
		if use_generator:
			if dataset=="CIFAR-10" or dataset=="binarynet":
				horizontal_flip=True
			if dataset=="SVHN":
				horizontal_flip=False
			datagen = ImageDataGenerator(
				width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=horizontal_flip)  # randomly flip images
			if keras.__version__[0]=='2':
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),steps_per_epoch=X_train.shape[0]/batch_size,
				nb_epoch=epochs,validation_data=(X_test, y_test),verbose=2,callbacks=[lrate, cback])
				#nb_epoch=epochs,validation_data=(X_test, y_test),verbose=2,callbacks=[cback])
			if keras.__version__[0]=='1':
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size), samples_per_epoch=X_train.shape[0], 
				nb_epoch=epochs, verbose=2,validation_data=(X_test,y_test),callbacks=[lrate, cback])
				#nb_epoch=epochs, verbose=2,validation_data=(X_test,y_test),callbacks=[cback])

		else:
			if keras.__version__[0]=='2':
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,epochs=epochs,callbacks=[lrate, cback])
				#history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,epochs=epochs,callbacks=[cback])
			if keras.__version__[0]=='1':
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,nb_epoch=epochs,callbacks=[lrate, cback])
				#history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,nb_epoch=epochs,callbacks=[cback])
		dic={'hard':history.history}
		foo=open('models/'+dataset+'/history_'+str(resid_levels)+'_residuals.pkl','wb')
		pickle.dump(dic,foo)
		foo.close()

if Evaluate:
	for resid_levels in range(1):
		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		model=get_model(dataset,resid_levels,batch_size)
		model.load_weights(weights_path)
		#lr=0.0001
		#opt = keras.optimizers.Adam(lr=lr,decay=1e-6)#SGD(lr=lr,momentum=0.9,decay=1e-5)
		opt = keras.optimizers.Adam()
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		#opt = keras.optimizers.Adam()
		#model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		#model.summary()
		score=model.evaluate(X_test,y_test,verbose=0, batch_size=batch_size)
		print "with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1])

if Inspect:
	#X_train = X_train[0:100,:,:,:]
	#y_train = y_train[0:100]
	X_test = X_test[0:100,:,:,:]
	y_test = y_test[0:100]

	for resid_levels in range(1):
		weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
		model=get_model(dataset,resid_levels,batch_size)
		model.load_weights(weights_path)

		lr=1.0
		#opt = keras.optimizers.Adam()
		opt = keras.optimizers.SGD(lr=lr,momentum=0.9)
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		#model.fit(X_train, y_train,batch_size=batch_size, verbose=2,epochs=epochs)
		#grads = K.gradients(model.total_loss, model.layers[1].output)
		sess = K.get_session()
		#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model.output, labels=tf.one_hot(y_test,10))
		cross_entropy = K.sparse_categorical_crossentropy(y_test,model.output)
		loss_operation = tf.reduce_mean(cross_entropy)

		##  Weight gradients
		#y_val_d_evaluated = sess.run(tf.gradients(loss_operation, model.layers[27].trainable_weights), feed_dict={model.input: X_test})
		#print("shape:")
		#print(np.shape(y_val_d_evaluated))
		#print("max:")
		#print(np.max(np.abs(y_val_d_evaluated)))
		#print("mean:")
		#print((np.sum(y_val_d_evaluated))/np.sum(np.shape(y_val_d_evaluated)))
		#print("std:")
		#print(np.std(y_val_d_evaluated, axis=(0,1)))
		##print(np.sqrt((np.sum(np.array(y_val_d_evaluated)**2,axis=(0,1,2,3)))/np.sum(np.shape(y_val_d_evaluated)[:-1])))
		
		#  Activation gradients
		y_val_d_evaluated = sess.run(tf.gradients(loss_operation, model.layers[14].output), feed_dict={model.input: X_test})
		print("shape:")
		print(np.shape(y_val_d_evaluated))
		print("max:")
		print(np.max(np.abs(y_val_d_evaluated)))
		print("mean:")
		print((np.sum(y_val_d_evaluated))/np.sum(np.shape(y_val_d_evaluated)))
		print("std:")
		print(np.std(y_val_d_evaluated, axis=(0,1,2,3)))

		##  Activation
		#y_val_d_evaluated = sess.run(model.layers[27].output, feed_dict={model.input: X_test})
		#print("shape:")
		#print(np.shape(y_val_d_evaluated))
		#print("max:")
		#print(np.max(np.abs(y_val_d_evaluated)))
		#print("mean:")
		#print((np.sum(y_val_d_evaluated))/np.sum(np.shape(y_val_d_evaluated)))
		#print("std:")
		##print(np.std(y_val_d_evaluated, axis=(0,1)))
		#print(np.sqrt((np.sum(np.array(y_val_d_evaluated)**2,axis=(0)))/np.sum(np.shape(y_val_d_evaluated)[:-1])))

		#get_intermediate_output = K.function([model.layers[0].input],
	#		[grads[0]])
#		layer_output = get_intermediate_output([X_test])[0]
#		print(np.shape(np.array(layer_output)))
	
		#print(np.shape(np.array(layer_grads)))
		#print(np.array(layer_grads))
		#get_intermediate_output = K.function([model.layers[0].input],
		#	[model.layers[3].output])
		#layer_output = get_intermediate_output([X_test])[0]
		#print(np.shape(np.array(layer_output)))

		
if Inspect_fromscratch:
	X_test = X_test[0:200,:,:,:]
	y_test = y_test[0:200]

	for resid_levels in range(1):
		model=get_model(dataset,resid_levels,batch_size)
		opt = keras.optimizers.Adam()
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		#model.fit(X_train, y_train,batch_size=batch_size, verbose=2,epochs=epochs)
		#grads = K.gradients(model.total_loss, model.layers[1].output)
		sess = K.get_session()
		#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model.output, labels=tf.one_hot(y_test,10))
		cross_entropy = K.sparse_categorical_crossentropy(y_test,model.output)
		loss_operation = tf.reduce_mean(cross_entropy)
		model.fit(X_test, y_test, batch_size=100,validation_data=(X_test, y_test), verbose=2,epochs=1,callbacks=[])

		#  Weight gradients
		y_val_d_evaluated = sess.run(tf.gradients(loss_operation, model.layers[14].trainable_weights), feed_dict={model.input: X_test})
		print("shape:")
		print(np.shape(y_val_d_evaluated))
		print("max:")
		print(np.max(np.abs(y_val_d_evaluated)))
		print("mean:")
		print((np.sum(y_val_d_evaluated))/np.sum(np.shape(y_val_d_evaluated)))
		print("std:")
		print(np.std(y_val_d_evaluated, axis=(0,1,2,3)))
		#print(np.sqrt((np.sum(np.array(y_val_d_evaluated)**2,axis=(0,1,2,3)))/np.sum(np.shape(y_val_d_evaluated)[:-1])))



