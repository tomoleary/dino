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


try:
	tf.random.set_seed(0)
except:
	tf.set_random_seed(0)


def jacobian_network_settings(problem_settings):
	'''
	'''
	jacobian_settings = {}

	# Jacobian loss settings
	jacobian_settings['batch_rank'] = 10
	jacobian_settings['target_rank'] = 50
	jacobian_settings['train_full_jacobian'] = False

	# Neural network architecture settings
	jacobian_settings['architecture'] = 'as_dense'
	jacobian_settings['depth'] = 6
	jacobian_settings['truncation_dimension'] = 50
	jacobian_settings['layer_rank'] = 8
	jacobian_settings['fixed_input_rank'] = 50
	jacobian_settings['fixed_output_rank'] = 50
	jacobian_settings['input_subspace'] = 'as'
	jacobian_settings['output_subspace'] = 'as'
	jacobian_settings['name_prefix'] = 'jacobian_'
	jacobian_settings['breadth_tolerance'] = 1e2
	jacobian_settings['max_breadth'] = 10

	jacobian_settings['input_dim'] = None
	jacobian_settings['output_dim'] = None


	# Training parameters
	jacobian_settings['opt_parameters'] = network_training_parameters()
	jacobian_settings['shuffle_every_epoch'] = False
	jacobian_settings['outer_epochs'] = 50
	jacobian_settings['inner_epochs'] = 1

	# Data settings
	jacobian_settings['test_data_size'] = 1024
	jacobian_settings['train_data_size'] = 1024
	jacobian_settings['batch_size'] = 256
	jacobian_settings['hess_batch_size'] = 32

	# Loading / saving settings
	jacobian_settings['save_weights'] = True
	jacobian_settings['weights_dir'] = 'trained_weights/'
	jacobian_settings['network_name'] = None
	jacobian_settings['initial_guess_path'] = None

	# Problem specific settings are passed in
	assert type(problem_settings) is dict, 'problem_settings passed in should have type dict'
	jacobian_settings['problem_settings'] = problem_settings

	return jacobian_settings



def jacobian_training_driver(settings,verbose = True):
	'''
	'''
	n_data = settings['train_data_size'] + settings['test_data_size']

	if settings['data_dir'] is None:
		data_dir = '../data/'+settings['problem_settings']['formulation']+'_n_obs_'+str(settings['problem_settings']['ntargets'])+\
			'_g'+str(settings['problem_settings']['gamma'])+'_d'+str(settings['problem_settings']['delta'])+\
				'_nx'+str(settings['problem_settings']['nx'])+'/'
	else: 
		data_dir = settings['data_dir']


	assert os.path.isdir(data_dir), 'Directory does not exist'+data_dir
	for loss_weight in settings['opt_parameters']['loss_weights']:
		assert loss_weight >= 0

	all_data = load_data(data_dir,rescale = False,n_data = n_data,derivatives = True)

	input_dim = all_data['m_data'].shape[-1]
	output_dim = all_data['q_data'].shape[-1]
	settings['input_dim'] = input_dim
	settings['output_dim'] = output_dim


	unflattened_train_dict, unflattened_test_dict = train_test_split(all_data,n_train = settings['train_data_size'])
	# If these assertions fail, then need to rethink the following logic
	assert len(unflattened_train_dict['m_data'].shape) == 2
	assert len(unflattened_train_dict['q_data'].shape) == 2
	n_train,dM = unflattened_train_dict['m_data'].shape
	n_test,dQ = unflattened_test_dict['q_data'].shape
	

	if settings['train_full_jacobian']:
		train_dict = remap_jacobian_data(unflattened_train_dict)
		test_dict = remap_jacobian_data(unflattened_test_dict)
	else:
		# Flatten training data
		train_dict = flatten_data(unflattened_train_dict,target_rank = settings['target_rank'],batch_rank = settings['batch_rank'])
		# Flatten testing data
		test_dict = flatten_data(unflattened_test_dict,target_rank = settings['target_rank'],batch_rank = settings['batch_rank'])

	if settings['architecture'] in ['as_resnet','as_dense']:
		data_dict_pod = {'input_train':train_dict['m_data'], 'output_train':train_dict['q_data']}
		last_layer_weights = build_POD_layer_arrays(data_dict_pod,truncation_dimension = settings['truncation_dimension'],\
										breadth_tolerance = settings['breadth_tolerance'],max_breadth = settings['max_breadth'])


		projectors = get_projectors(data_dir,fixed_input_rank = settings['fixed_input_rank'],fixed_output_rank = settings['fixed_output_rank'])

		input_projector,output_projector = modify_projectors(projectors,settings['input_subspace'],settings['output_subspace'])

		projector_dict = {}
		projector_dict['input'] = input_projector
		projector_dict['output'] = last_layer_weights
	else:
		projector_dict = None

	################################################################################

	regressor = choose_network(settings,projector_dict)

	if settings['initial_guess_path'] is not None:
		assert os.path.isfile(settings['initial_guess_path']), 'Trained weights may not exist as specified: '+str(settings['initial_guess_path'])
		import pickle
		regressor_weights = pickle.load(open(settings['initial_guess_path'],'rb'))
		for layer in regressor.layers:
			layer.set_weights(regressor_weights[layer.name])
		if settings['opt_parameters']['train_hessianlearn']:
			settings['opt_parameters']['layer_weights'] = regressor_weights
	if settings['train_full_jacobian']:
		print('Equipping Jacobian')
		settings['opt_parameters']['train_full_jacobian'] = settings['train_full_jacobian']
		regressor = equip_model_with_full_jacobian(regressor,name_prefix = settings['name_prefix'])
		
	else:
		regressor = equip_model_with_sketched_jacobian(regressor,settings['batch_rank'],name_prefix = settings['name_prefix'])


	print('Commencing training'.center(80))
	if settings['shuffle_every_epoch']:
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


def jacobian_network_loader(settings,file_name = None):
	"""
	"""
	if file_name is None:
		file_name = settings['weights_dir']+settings['name_prefix']+str(settings['architecture'])+\
			'_depth'+str(settings['depth'])+'_batch_rank_'+str(settings['batch_rank'])+'.pkl'

	assert os.path.isfile(file_name), 'Trained weights may not exist as specified: '+str(file_name)

	# jacobian_weights = pickle.load(open(file_name,'rb'))

	try:
		# import pickle
		jacobian_weights = pickle.load(open(file_name,'rb'))
	except:
		import pickle5
		jacobian_weights = pickle5.load(open(file_name,'rb'))

	try:
		projector_dict = {'input':jacobian_weights[settings['name_prefix']+'input_proj_layer'][0],\
							 'output':jacobian_weights[settings['name_prefix']+'output_layer']}
	except:
		projector_dict = None

	jacobian_network = choose_network(settings,projector_dict)

	for layer in jacobian_network.layers:
		layer.set_weights(jacobian_weights[layer.name])

	return jacobian_network



def equip_model_with_sketched_jacobian(model,batch_rank,name_prefix = ''):
	"""
	"""
	assert len(model.inputs) == 1
	assert len(model.outputs) == 1
	input_m = model.inputs[0]
	output_q = model.outputs[0]
	try:
		input_dim = input_m.shape[-1].value
		output_dim = output_q.shape[-1].value
	except:
		input_dim = input_m.shape[-1]
		output_dim = output_q.shape[-1]
	input_U = tf.keras.layers.Input(shape=(output_dim,batch_rank),name=name_prefix+'_input_U')
	input_V = tf.keras.layers.Input(shape=(input_dim,batch_rank),name=name_prefix+'_input_V')
	UTys = tf.einsum('ijk,ij->ik',input_U,output_q)
	unstacked_UTys = tf.unstack(UTys,axis = 1)
	unstacked_UTdydxs = [tf.gradients(UTy,input_m,stop_gradients=input_U,name = name_prefix+'UT_dydx'+str(i))[0] for i, UTy in enumerate(unstacked_UTys)]
	UTdydxs = tf.stack(unstacked_UTdydxs,axis = 1)
	UTdydxVs = tf.einsum('ijk,ikl->ijl',UTdydxs,input_V,name = name_prefix+'reduced_jacobian')
	output_sigma = UTdydxVs
	return tf.keras.models.Model([input_m,input_U,input_V], [output_q,output_sigma])


def equip_model_with_full_jacobian(model,name_prefix = ''):
	"""
	"""
	assert len(model.inputs) == 1
	assert len(model.outputs) == 1
	input_m = model.inputs[0]
	output_q = model.outputs[0]
	try:
		input_dim = input_m.shape[-1].value
		output_dim = output_q.shape[-1].value
	except:
		input_dim = input_m.shape[-1]
		output_dim = output_q.shape[-1]


	with tf.GradientTape(persistent = True) as tape:
		tape.watch(input_m)
		qout = model(input_m)
	# Full batched Jacobian
	fullJ = tape.batch_jacobian(qout,input_m)

	new_model = tf.keras.models.Model(input_m,[output_q,fullJ])

	return new_model


