import numpy as np
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
import tensorflow as tf
import pickle

import time

from .neuralNetworks import *
from .neuralNetworkUtilities import *
from .trainingUtilities import *
from .dataUtilities import load_data, remap_jacobian_data

from .jacobianConstruction import *


def get_the_data(settings, remapped_data = None, unflattened_data = None, verbose = True):
	################################################################################
	# Set up training and testing data.
	unflattened_train_dict = None

	if settings['data_dir'] is None:
		data_dir = '../data/'+settings['problem_settings']['formulation']+'_n_obs_'+str(settings['problem_settings']['ntargets'])+\
			'_g'+str(settings['problem_settings']['gamma'])+'_d'+str(settings['problem_settings']['delta'])+\
				'_nx'+str(settings['problem_settings']['nx'])+'/'
		settings['data_dir'] = data_dir
	else: 
		data_dir = settings['data_dir']

	assert os.path.isdir(data_dir), 'Directory does not exist'+data_dir
	if remapped_data is None:
		if unflattened_data is None:
			n_data = settings['train_data_size'] + settings['test_data_size']
			all_data = load_data(data_dir,rescale = False,n_data = n_data,derivatives = True)
			unflattened_train_dict, unflattened_test_dict = train_test_split(all_data,n_train = settings['train_data_size'])
		else:
			if len(unflattened_data) == 1:
				unflattened_train_dict = unflattened_data[0]
				unflattened_test_dict = None
			else:
				unflattened_train_dict, unflattened_test_dict = unflattened_data
		if settings['train_full_jacobian']:
			train_dict = remap_jacobian_data(unflattened_train_dict)
			if unflattened_test_dict is None:
				test_dict = None
			else:
				test_dict = remap_jacobian_data(unflattened_test_dict)
		else:
			train_dict = flatten_data(unflattened_train_dict,target_rank = settings['target_rank'],batch_rank = settings['batch_rank'])
			if unflattened_test_dict is None:
				test_dict = None
			else:
				test_dict = flatten_data(unflattened_test_dict,target_rank = settings['target_rank'],batch_rank = settings['batch_rank'])

	else:
		assert settings['train_full_jacobian']
		if len(remapped_data) == 1:
			train_dict = remapped_data[0]
			test_dict = None
		else:
			train_dict, test_dict = remapped_data

	# Check to make sure that the training and testing data loaded appropriately
	assert train_dict['m_data'].shape[0] != 0
	assert test_dict['m_data'].shape[0] != 0

	# If these assertions fail, then need to rethink the following logic
	assert len(train_dict['m_data'].shape) == 2
	assert len(train_dict['q_data'].shape) == 2
	n_train,dM = train_dict['m_data'].shape
	if test_dict is not None:
		n_test,dQ = test_dict['q_data'].shape
	else:
		n_test = 0
		dQ = train_dict['q_data'].shape[-1]
	settings['input_dim'] = dM
	settings['output_dim'] = dQ

	return {'train_dict': train_dict, 'test_dict': test_dict,'settings':settings,\
			'unflattened_train_dict':unflattened_train_dict}


def prune_the_data(projector_dict,train_dict,test_dict,\
	 input_pruning = True, output_pruning = True,verbose = False):
	'''
	'''
	assert input_pruning or output_pruning
	# Either way the J data gets pruned
	J_output_already_pruned = False
	if 'J_data' in train_dict.keys():
		J_train = train_dict['J_data']
		J_test = test_dict['J_data']
	elif 'PhiTJ_data' in train_dict.keys():
		J_train = train_dict['PhiTJ_data']
		J_test = test_dict['PhiTJ_data']
		J_output_already_pruned = True
		print('Assuming that Jacobian already pruned wrt outputs!')
	else:
		raise
	if verbose:
		print('J_train.shape = ',J_train.shape)
		print('J_test.shape = ',J_test.shape)
	# Save the full data for re-stitching post-process
	train_dict['J_full'] = J_train.copy()
	test_dict['J_full'] = J_test.copy()
	################################################################################
	# m data
	if input_pruning:
		m_train = train_dict['m_data']
		m_test = test_dict['m_data']
		# Save the full data for re-stitching post-process
		train_dict['m_full'] = m_train.copy()
		test_dict['m_full'] = m_test.copy()
		if verbose:
			print('m_train.shape = ',m_train.shape)
			print('m_test.shape = ',m_test.shape)
		input_projector = projector_dict['input']
		if verbose:
			print('input_projector.shape = ',input_projector.shape)
		assert input_projector.shape[0] == m_train.shape[1]

		m_train = np.einsum('ji,kj->ki',input_projector,m_train)
		m_test = np.einsum('ji,kj->ki',input_projector,m_test)

		if verbose:
			print('m_train.shape = ',m_train.shape)
			print('m_test.shape = ',m_test.shape)

			print('Pruning the input dimension on the J data')

		J_train = np.einsum('ji,klj->kli',input_projector,J_train)
		J_test = np.einsum('ji,klj->kli',input_projector,J_test)
		if verbose:
			print('J_train.shape = ',J_train.shape)
			print('J_test.shape = ',J_test.shape)

		train_dict['m_data'] = m_train
		test_dict['m_data'] = m_test

	if output_pruning:
		q_train = train_dict['q_data']
		q_test = test_dict['q_data']
		# Save the full data for re-stitching post-process
		train_dict['q_full'] = q_train.copy()
		test_dict['q_full'] = q_test.copy()
		if verbose:
			print('q_train.shape = ',q_train.shape)
			print('q_test.shape = ',q_test.shape)
		output_projector = projector_dict['output']
		last_layer_bias = projector_dict['last_layer_bias']
		if verbose:
			print('output_projector.shape = ',output_projector.shape)
		assert output_projector.shape[0] == q_train.shape[1]

		q_train = np.einsum('ji,kj->ki',output_projector,q_train - last_layer_bias)
		q_test = np.einsum('ji,kj->ki',output_projector,q_test- last_layer_bias)

		if J_output_already_pruned:
			assert J_train.shape[1] == output_projector.shape[1]
		else:
			J_train = np.einsum('ji,kjl->kil',output_projector,J_train)
			J_test = np.einsum('ji,kjl->kil',output_projector,J_test)
			if verbose:
				print('J_train.shape = ',J_train.shape)
				print('J_test.shape = ',J_test.shape)

		train_dict['q_data'] = q_train
		test_dict['q_data'] = q_test

	train_dict['J_data'] = J_train
	test_dict['J_data'] = J_test

	return train_dict, test_dict


def setup_the_dino(settings,train_dict,projector_dict = None,\
				 reduced_input_training = False,reduced_output_training = False,\
				 no_jacobian = False,reduced_output_Jacobian = False):
	################################################################################
	# Set up the neural networks
	regressor = choose_network(settings,projector_dict,reduced_input_training = reduced_input_training,\
														reduced_output_training = reduced_output_training)

	################################################################################
	# Initial guess choice
	if settings['initial_guess_path'] is not None:
		assert os.path.isfile(settings['initial_guess_path']), 'Trained weights may not exist as specified: '+str(settings['initial_guess_path'])
		import pickle
		regressor_weights = pickle.load(open(settings['initial_guess_path'],'rb'))
		for layer in regressor.layers:
			layer.set_weights(regressor_weights[layer.name])
		if settings['opt_parameters']['train_hessianlearn']:
			settings['opt_parameters']['layer_weights'] = regressor_weights

	################################################################################
	if not no_jacobian:
		# Tease out the derivatives
		if reduced_output_Jacobian:
			reduced_output_basis = projector_dict['output']
			regressor = equip_model_with_output_reduced_jacobian(regressor,reduced_output_basis,name_prefix = settings['name_prefix'])
		elif settings['train_full_jacobian']:
			print('Equipping Jacobian')
			settings['opt_parameters']['train_full_jacobian'] = settings['train_full_jacobian']
			regressor = equip_model_with_full_jacobian(regressor,name_prefix = settings['name_prefix'])
			
		else:
			regressor = equip_model_with_sketched_jacobian(regressor,settings['batch_rank'],name_prefix = settings['name_prefix'])
	
	return regressor


def train_dino(settings, regressor,train_dict,test_dict,unflattened_train_dict = None,verbose = True):
	################################################################################
	# Start the training
	print('Commencing training'.center(80))
	if settings['shuffle_every_epoch']:
		assert unflattened_train_dict is not None
		settings['opt_parameters']['keras_epochs'] = settings['inner_epochs']
		for epoch in range(settings['outer_epochs']):
			if verbose:
				print(('Running inner iteration '+str(epoch)).center(80))
			train_dict = flatten_data(unflattened_train_dict,target_rank = settings['target_rank'],batch_rank = settings['batch_rank'],order_random = True,burn_in = epoch)
			regressor = train_h1_network(regressor,train_dict,test_dict,opt_parameters = settings['opt_parameters'],verbose = True)
		pass
	else:
		regressor = train_h1_network(regressor,train_dict,test_dict,opt_parameters = settings['opt_parameters'])

	if settings['save_weights']:
		import pickle

		jacobian_weights = {}
		for layer in regressor.layers:
			jacobian_weights[layer.name] = layer.get_weights()

		os.makedirs(settings['weights_dir'],exist_ok = True)
		if settings['network_name'] is None:
			network_name = settings['name_prefix']+str(settings['architecture'])+'_depth'+str(settings['depth'])+\
										'_batch_rank_'+str(settings['batch_rank'])
		else:
			network_name = settings['network_name']

		jacobian_filename = settings['weights_dir']+network_name+'.pkl'
		with open(jacobian_filename,'wb+') as f_jacobian:
			pickle.dump(jacobian_weights,f_jacobian,pickle.HIGHEST_PROTOCOL)

	return regressor

def restitch_and_postprocess(reduced_regressor,settings,train_dict,\
							test_dict,projector_dict,l2_only = False,\
							reduced_output_Jacobian = False, verbose = False,\
							logger = {}):
	for i in range(5):
		print(80*'#')
	print('Re-stitching post processing')
	# Setup a full re-stiched network 
	regressor = setup_the_dino(settings,train_dict,projector_dict,\
				reduced_input_training = False,reduced_output_training = False,\
						no_jacobian = l2_only, reduced_output_Jacobian = reduced_output_Jacobian)

	big_network_layers = [layer.name for layer in regressor.layers]
	little_network_layers = [layer.name for layer in reduced_regressor.layers]

	for layer in reduced_regressor.layers:
		if layer.name in big_network_layers:
			regressor.get_layer(layer.name).set_weights(layer.get_weights())

	# Handle output and input layers here???

	opt_parameters = settings['opt_parameters']
	optimizer = tf.keras.optimizers.Adam(learning_rate = opt_parameters['keras_alpha'])
	if l2_only:
		assert len(regressor.outputs) == 1
		losses = normalized_mse
		metrics = [l2_accuracy]
		loss_weights = [1.0]
	else:
		losses = [normalized_mse]+[normalized_mse_matrix]
		metrics = [l2_accuracy]+[f_accuracy_matrix]
		loss_weights = opt_parameters['loss_weights']

	regressor.compile(optimizer=optimizer,loss=losses,loss_weights = loss_weights,metrics=metrics)

	input_train = [train_dict['m_full']]
	input_test = [test_dict['m_full']]

	if not ('q_full' in train_dict.keys()):
		train_dict['q_full'] = train_dict['q_data']
		test_dict['q_full'] = test_dict['q_data']

	if l2_only:
		output_train = [train_dict['q_full']]
		output_test = [test_dict['q_full']]
	else:
		output_train = [train_dict['q_full'],train_dict['J_full']]
		output_test = [test_dict['q_full'],test_dict['J_full']]

	eval_train = regressor.evaluate(input_train,output_train,verbose=2)

	if l2_only:
		l2_loss_train, l2_acc_train = regressor.evaluate(input_train,output_train,verbose=2)

		print('After training: l2 accuracy = ', l2_acc_train)
		l2_loss_test, l2_acc_test = regressor.evaluate(input_test,output_test,verbose=2)
		print('After training: test l2 accuracy = ', l2_acc_test)
		logger['l2_train'] = l2_acc_train
		logger['l2_test'] = l2_acc_test
	else:
		eval_train_dict = {out: eval_train[i] for i, out in enumerate(regressor.metrics_names)}
		print('After training: l2, h1 training accuracies = ', eval_train[3], eval_train[6])
		eval_test = regressor.evaluate(input_test,output_test,verbose=2)
		eval_test_dict = {out: eval_test[i] for i, out in enumerate(regressor.metrics_names)}
		if verbose:
			print('eval_test_dict = ',eval_test_dict)
		print('After training: l2, h1 testing accuracies =  ', eval_test[3], eval_test[6])
		logger['l2_train'], logger['h1_train'] = eval_train[3], eval_train[6]
		logger['l2_test'], logger['h1_test'] = eval_test[3], eval_test[6]

	return regressor
	


def jacobian_training_driver(settings, remapped_data = None, unflattened_data = None, verbose = True):
	'''
	'''
	for loss_weight in settings['opt_parameters']['loss_weights']:
		assert loss_weight >= 0
	################################################################################
	# Set up training and testing data.
	data_dict = get_the_data(settings, remapped_data = remapped_data, unflattened_data = unflattened_data, verbose = verbose)
	train_dict = data_dict['train_dict']
	test_dict = data_dict['test_dict']
	unflattened_train_dict = data_dict['unflattened_train_dict']

	################################################################################
	# Setup the reduced bases (if it applies)
	projector_dict = setup_reduced_bases(settings,train_dict)
	# Prune the data here if desired...
	if settings['reduced_input_training'] or settings['reduced_output_training']:
		assert settings['train_full_jacobian']
		# Need to pass the reduced input dimension in for network construction
		# The projector is assumed to have dims (dM,rM)
		assert len(projector_dict['input'].shape) == 2
		if settings['reduced_input_training']:
			settings['reduced_input_dim'] = projector_dict['input'].shape[1]
			print('reduced_input_dim = ',settings['reduced_input_dim'])
		if settings['reduced_output_training']:
			settings['reduced_output_dim'] = projector_dict['output'].shape[1]
		

		# Projector is also assumed to be orthonormal, this should maybe be checked / confirmed. 
		train_dict, test_dict = prune_the_data(projector_dict,train_dict,test_dict,input_pruning = settings['reduced_input_training'],\
																			output_pruning = settings['reduced_output_training'])

	################################################################################
	# Set up the neural networks
	regressor = setup_the_dino(settings,train_dict,projector_dict,reduced_input_training = settings['reduced_input_training'],\
																	reduced_output_training = settings['reduced_output_training'])

	################################################################################
	# Start the training
	regressor = train_dino(settings, regressor,train_dict,test_dict,unflattened_train_dict)

	################################################################################
	# Post-processing / re-stitching in the case of the reduced training.
	if settings['reduced_input_training'] or settings['reduced_output_training']:
		regressor = restitch_and_postprocess(regressor,settings,train_dict,test_dict,projector_dict)
	
	return regressor



