# This file is part of the dino package
#
# dino is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or any later version.
#
# dino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import numpy as np
import tensorflow as tf

from .dataUtilities import *

# Loss functions to be used in keras

def normalized_mse(y_true, y_pred):
		squared_difference = tf.square(y_true - y_pred)
		normalized_squared_difference = tf.reduce_mean(squared_difference,axis=-1)\
										/(tf.reduce_mean(tf.square(y_true),axis =-1))
		return tf.reduce_mean(normalized_squared_difference)

def normalized_mse_matrix(y_true, y_pred):
		squared_difference = tf.square(y_true - y_pred)
		# As far as I know, cannot specifcy axes as tuple axis = (1,2) like in numpy
		# Not the most elegant, but gets the job done...
		normalized_squared_difference = tf.reduce_sum(tf.reduce_sum(squared_difference,axis=-1),axis=-1)\
								/tf.reduce_sum(tf.reduce_sum(tf.square(y_true),axis =-1),axis = -1)
		return tf.reduce_mean(normalized_squared_difference)

def l2_accuracy(y_true, y_pred):
	squared_difference = tf.square(y_true - y_pred)
	normalized_squared_difference = tf.reduce_mean(squared_difference,axis=-1)\
									/tf.reduce_mean(tf.square(y_true),axis =-1)
	return 1. - tf.sqrt(tf.reduce_mean(normalized_squared_difference))

def f_accuracy_matrix(y_true, y_pred):
	squared_difference = tf.square(y_true - y_pred)
	# As far as I know, cannot specifcy axes as tuple axis = (1,2) like in numpy
	# Not the most elegant, but gets the job done...
	normalized_squared_difference = tf.reduce_sum(tf.reduce_sum(squared_difference,axis=-1),axis=-1)\
								/tf.reduce_sum(tf.reduce_sum(tf.square(y_true),axis =-1),axis = -1)
	return 1. - tf.sqrt(tf.reduce_mean(normalized_squared_difference))

def network_training_parameters():
	"""
	Settings for training
	"""
	opt_parameters = {}
	# How to weight the least squares losses [l2,h1 seminorm]
	opt_parameters['loss_weights'] = [1.0,1.0]
	opt_parameters['train_full_jacobian'] = True

	# Keras training parameters
	opt_parameters['train_keras'] = True
	opt_parameters['keras_epochs'] = 10
	opt_parameters['keras_batch_size'] = 32
	opt_parameters['keras_opt'] = 'adam' # choose from adam / SGD / whatever keras optimizers
	opt_parameters['keras_alpha'] = 1e-3
	opt_parameters['keras_verbose'] = False
	opt_parameters['keras_optimizer'] = None

	# Hessianlearn training parameters
	opt_parameters['train_hessianlearn'] = False
	opt_parameters['hessian_low_rank'] = 40
	opt_parameters['hess_alpha'] = 1e-4
	opt_parameters['hess_gbatch_size'] = 256
	opt_parameters['hess_batch_size'] = 32
	opt_parameters['hess_sweeps'] = 10
	opt_parameters['layer_weights'] = {}
	opt_parameters['printing_sweep_frequency'] = 0.1

	opt_parameters['callbacks'] = []
	
	return opt_parameters


def train_h1_network(network,train_dict,test_dict = None,opt_parameters = network_training_parameters(),verbose = True,logger = {}):
	"""
	h1 training routines.
	"""
	if opt_parameters['keras_optimizer'] is not None:
		optimizer = opt_parameters['keras_optimizer']
	else:
		if opt_parameters['keras_opt'] == 'adam':
			optimizer = tf.keras.optimizers.Adam(learning_rate = opt_parameters['keras_alpha'])
		elif opt_parameters['keras_opt'] == 'sgd':
			optimizer = tf.keras.optimizers.SGD(learning_rate = opt_parameters['keras_alpha'])
		else:
			raise 'Invalid choice of optimizer'

	assert len(opt_parameters['loss_weights']) == len(network.outputs)
	assert len(network.outputs) == 2
	# Specify losses and metrics
	losses = [normalized_mse]+[normalized_mse_matrix]
	metrics = [l2_accuracy]+[f_accuracy_matrix]

	network.compile(optimizer=optimizer,loss=losses,loss_weights = opt_parameters['loss_weights'],metrics=metrics)
	

	if opt_parameters['train_full_jacobian']:
		input_train = [train_dict['m_data']]
		output_train = [train_dict['q_data'],train_dict['J_data']]
		if test_dict is None:
			test_data = None
		else:
			input_test = [test_dict['m_data']]
			output_test = [test_dict['q_data'],test_dict['J_data']]
			test_data = (input_test,output_test)
	else:
		input_train = [train_dict['m_data'],train_dict['U_data'],train_dict['V_data']]
		output_train = [train_dict['q_data'],train_dict['sigma_data']]
		if test_dict is None:
			test_data = None
		else:
			input_test = [test_dict['m_data'],test_dict['U_data'],test_dict['V_data']]
			output_test = [test_dict['q_data'],test_dict['sigma_data']]
			test_data = (input_test,output_test)

	if verbose:
		eval_train = network.evaluate(input_train,output_train,verbose=2)
		eval_train_dict = {out: eval_train[i] for i, out in enumerate(network.metrics_names)}
		print('Before training: l2, h1 training accuracies = ', eval_train[3], eval_train[6])
		if test_dict is not None:
			eval_test = network.evaluate(input_test,output_test,verbose=2)
			eval_test_dict = {out: eval_test[i] for i, out in enumerate(network.metrics_names)}
			print('Before training: l2, h1 testing accuracies =  ', eval_test[3], eval_test[6])


	if opt_parameters['train_keras']:
		network.fit(input_train,output_train,
					validation_data = test_data,epochs = opt_parameters['keras_epochs'],\
										batch_size = opt_parameters['keras_batch_size'],verbose = opt_parameters['keras_verbose'],\
										callbacks = opt_parameters['callbacks'])

	if opt_parameters['train_hessianlearn']:
		import sys,os
		sys.path.append( os.environ.get('HESSIANLEARN_PATH'))
		import hessianlearn as hess
		KMWSettings = hess.KerasModelWrapperSettings()
		KMWSettings['max_sweeps'] = opt_parameters['hess_sweeps']
		KMWSettings['layer_weights'] = opt_parameters['layer_weights']
		KMWSettings['printing_sweep_frequency'] = opt_parameters['printing_sweep_frequency']
		KMW = hess.KerasModelWrapper(network,settings = KMWSettings)
		optimizer = hess.LowRankSaddleFreeNewton # The class constructor, not an instance
		hess_opt_parameters = hess.ParametersLowRankSaddleFreeNewton()
		hess_opt_parameters['hessian_low_rank'] = opt_parameters['hessian_low_rank']
		hess_opt_parameters['alpha'] = opt_parameters['hess_alpha']
		KMW.set_optimizer(optimizer,parameters = hess_opt_parameters)

		problem = KMW.problem
		if opt_parameters['train_full_jacobian']:
			hess_train_dict = {problem.x:input_train[0],\
									problem.y_true[0]:output_train[0],problem.y_true[1]:output_train[1]}
			if test_dict is not None:
				hess_val_dict = {problem.x[0]:input_test[0],\
										problem.y_true[0]:output_test[0],problem.y_true[1]:output_test[1]}

		else:
			hess_train_dict = {problem.x[0]:input_train[0],problem.x[1]:input_train[1],problem.x[2]:input_train[2],\
									problem.y_true[0]:output_train[0],problem.y_true[1]:output_train[1]}
			if test_dict is not None:
				hess_val_dict = {problem.x[0]:input_test[0],problem.x[1]:input_test[1],problem.x[2]:input_test[2],\
										problem.y_true[0]:output_test[0],problem.y_true[1]:output_test[1]}
			else:
				hess_val_dict = None
		data = hess.Data(hess_train_dict,opt_parameters['hess_gbatch_size'],\
			validation_data = hess_val_dict,hessian_batch_size = opt_parameters['hess_batch_size'])
		# And finally one can call fit!
		KMW.fit(data)

	if verbose:
		eval_train = network.evaluate(input_train,output_train,verbose=2)
		eval_train_dict = {out: eval_train[i] for i, out in enumerate(network.metrics_names)}
		print('After training: l2, h1 training accuracies = ', eval_train[3], eval_train[6])
		logger['l2_train'], logger['h1_train'] = eval_train[3],eval_train[6]
		if test_dict is not None:
			eval_test = network.evaluate(input_test,output_test,verbose=2)
			eval_test_dict = {out: eval_test[i] for i, out in enumerate(network.metrics_names)}
			print('After training: l2, h1 testing accuracies =  ', eval_test[3], eval_test[6])
			logger['l2_test'], logger['h1_test'] = eval_test[3],eval_test[6]

	return network

def train_l2_network(network,train_dict,test_dict = None,opt_parameters = network_training_parameters(),verbose = True):
	"""
	L2 training routines
	"""
	if opt_parameters['keras_opt'] == 'adam':
		optimizer = tf.keras.optimizers.Adam(learning_rate = opt_parameters['keras_alpha'])
	elif opt_parameters['keras_opt'] == 'sgd':
		optimizer = tf.keras.optimizers.SGD(learning_rate = opt_parameters['keras_alpha'])
	else:
		raise 'Invalid choice of optimizer'
	pass

	network.compile(optimizer=optimizer,loss=normalized_mse,metrics=[l2_accuracy])

	input_train = train_dict['m_data']
	output_train = train_dict['q_data']
	if test_dict is None:
		test_data = None
	else:
		input_test = test_dict['m_data']
		output_test = test_dict['q_data']
		test_data = (input_test,output_test)
	if verbose:
		l2_loss_train, l2_acc_train = network.evaluate(input_train,output_train,verbose=2)
		print('Before training: l2 accuracy = ', l2_acc_train)
		if test_dict is not None:
			l2_loss_test, l2_acc_test = network.evaluate(input_test,output_test,verbose=2)
			print('Before training: l2accuracy = ', l2_acc_test)

	network.fit(input_train,output_train,
				validation_data = test_data,epochs = opt_parameters['keras_epochs'],\
									batch_size = opt_parameters['keras_batch_size'],verbose = opt_parameters['keras_verbose'],\
									callbacks = opt_parameters['callbacks'])

	if verbose:
		l2_loss_train, l2_acc_train = network.evaluate(input_train,output_train,verbose=2)
		print('After training: l2 accuracy = ', l2_acc_train)
		if test_dict is not None:
			l2_loss_test, l2_acc_test = network.evaluate(input_test,output_test,verbose=2)
			print('After training: test l2 accuracy = ', l2_acc_test)

	return network




