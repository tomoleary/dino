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
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
import tensorflow as tf
import time


try:
	tf.random.set_seed(0)
except:
	tf.set_random_seed(0)

from .neuralNetworks import *
from .neuralNetworkUtilities import *
from .trainingUtilities import *
from .dataUtilities import load_data


def observable_network_settings(problem_settings):
	"""
	This defines network settings when used without Jacobians.
	"""
	settings = {}
	
	# Neural network architecture settings
	settings['architecture'] = 'rb_dense'
	settings['compat_layer'] = True
	settings['depth'] = 6
	settings['truncation_dimension'] = 50
	settings['layer_rank'] = 8
	settings['fixed_input_rank'] = 50
	settings['fixed_output_rank'] = 50
	settings['input_basis'] = 'as'
	settings['output_basis'] = 'jjt'
	settings['name_prefix'] = 'observable_'
	settings['compat_layer'] = True
	settings['breadth_tolerance'] = 1e2
	settings['max_breadth'] = 10

	settings['weights_dir'] = 'trained_weights/'

	# Training parameters
	settings['opt_parameters'] = network_training_parameters()
	# settings['opt_parameters']['loss_weights'] = 1.0
	settings['epochs'] = 50

	# Data settings
	settings['test_data_size'] = 1024
	settings['train_data_size'] = 15*1024
	settings['batch_size'] = 256
	settings['hess_batch_size'] = 32


	# Problem specific settings are passed in
	assert type(problem_settings) is dict, 'problem_settings passed in should have type dict'
	settings['problem_settings'] = problem_settings


	return settings


def observable_training_driver(settings,verbose = True):
	"""
	Driver for generic L2 training
	"""
	n_data = settings['train_data_size'] + settings['test_data_size']
	if settings['data_dir'] is None:
		data_dir = '../data/'+settings['problem_settings']['formulation']+'_n_obs_'+str(settings['problem_settings']['ntargets'])+\
			'_g'+str(settings['problem_settings']['gamma'])+'_d'+str(settings['problem_settings']['delta'])+\
				'_nx'+str(settings['problem_settings']['nx'])+'/'
		settings['data_dir'] = data_dir
	else:
		data_dir = settings['data_dir']

	assert os.path.isdir(data_dir), 'Directory does not exist'+data_dir
	for loss_weight in settings['opt_parameters']['loss_weights']:
		assert loss_weight >= 0

	all_data = load_data(data_dir,rescale = False,n_data = n_data,derivatives = False)

	input_dim = all_data['m_data'].shape[-1]
	output_dim = all_data['q_data'].shape[-1]

	train_dict, test_dict = train_test_split(all_data,n_train = settings['train_data_size'])

	if settings['architecture'].lower() in ['rb_resnet','rb_dense']:
		projector_dict = setup_reduced_bases(settings,train_dict)
	else:
		projector_dict = None

	################################################################################

	regressor = choose_network(settings,projector_dict)
	print('Commencing training'.center(80))
	regressor = train_l2_network(regressor,train_dict,test_dict,opt_parameters = settings['opt_parameters'])

	return regressor


def observable_network_loader(settings,file_name = None,verbose = False):
	"""
	Loader of the observable network after training is complete. 
	No Jacobians in this case.
	"""
	
	if file_name is None:
		file_name = settings['weights_dir']+settings['name_prefix']+str(settings['architecture'])+\
			'_depth'+str(settings['depth'])+'.pkl'


	assert os.path.isfile(file_name), 'Trained weights may not exist as specified: '+str(file_name)

	try:
		import pickle
		observable_weights = pickle.load(open(file_name,'rb'))
	except:
		import pickle5
		observable_weights = pickle5.load(open(file_name,'rb'))

	try:
		projector_dict = {'input':observable_weights[settings['name_prefix']+'input_proj_layer'][0],\
						 'output':observable_weights[settings['name_prefix']+'output_layer'][0].T,\
						 'last_layer_bias':observable_weights[settings['name_prefix']+'output_layer'][1]}
	except:
		projector_dict = {}

	use_compat_layer = False
	for weight_name in observable_weights:
		if 'compat_layer' in weight_name:
			use_compat_layer = True

	settings['compat_layer'] = use_compat_layer


	observable_network = choose_network(settings,projector_dict)

	for layer in observable_network.layers:
		if verbose:
			print('layer = ',layer.name)
			try:
				print('SHAPE = ',observable_weights[layer.name][0].shape)
			except:
				print('issue moving on')
		layer.set_weights(observable_weights[layer.name])

	return observable_network

