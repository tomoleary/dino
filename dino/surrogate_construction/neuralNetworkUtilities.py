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

from .neuralNetworks import *
from .dataUtilities import get_projectors, modify_projectors


# Define procedures for defining networks via POD spectral decay here

def build_POD_layer_arrays(data_dict,truncation_dimension = None, breadth_tolerance = 1e2, max_breadth = 10):
	"""
	"""
	U, s, _ = np.linalg.svd((data_dict['output_train']-np.mean(data_dict['output_train'],axis=0)).T,\
							full_matrices = False)
	if truncation_dimension is None:
		orders_reduction = np.array([s[0]/(si+1e-12) for si in s])
		# Absolute tolerance for breadth
		abs_tol_idx = np.where(orders_reduction>breadth_tolerance)[0][0]
	#     print('abs tol idx = ',abs_tol_idx)
		truncation_dimension = min(abs_tol_idx,max_breadth)
	else:
		assert type(truncation_dimension) is int

	U = U[:,:truncation_dimension]
	last_layer_weights = [U.T,np.mean(data_dict['output_train'], axis=0)]

	return last_layer_weights


def choose_network(settings,projector_dict = None):
	'''

	'''

	# Load the input and output projectors
	layer_weights = {}

	depth = settings['depth']
	truncation_dimension = settings['truncation_dimension']
	architecture = settings['architecture']
	if architecture in ['as_resnet','as_dense']:
		assert projector_dict is not None

	network_name_prefix = settings['name_prefix']

	if architecture == 'as_resnet':
		print(80*'#')
		print('Loading AS ResNet'.center(80))

		ranks = depth*[settings['layer_rank']]
		input_projector = projector_dict['input']
		layer_weights[network_name_prefix+'input_proj_layer'] = [input_projector]
		last_layer_weights = projector_dict['output']
		layer_weights[network_name_prefix+'output_layer'] = last_layer_weights
		regressor, last_layer_weights = construct_projected_resnet(input_projector, last_layer_weights, ranks,\
											  name_prefix = network_name_prefix)


		# regressor.summary()

	elif architecture == 'as_dense':
		print(80*'#')
		print('Loading AS Dense'.center(80))
		hidden_layer_dimensions = 2*[truncation_dimension]

		input_projector = projector_dict['input']
		layer_weights[network_name_prefix+'input_proj_layer'] = [input_projector]
		last_layer_weights = projector_dict['output']
		layer_weights[network_name_prefix+'output_layer'] = last_layer_weights
		regressor, last_layer_weights = construct_projected_dense(input_projector, last_layer_weights, depth,\
											 name_prefix = network_name_prefix)


	elif architecture == 'generic_dense':
		print(80*'#')
		print('Loading generic dense'.center(80))
		assert settings['input_dim'] is not None
		assert settings['output_dim'] is not None
		input_dim = settings['input_dim']
		output_dim = settings['output_dim']
		truncation_dimension = min(input_dim,output_dim)
		regressor = generic_dense(input_dim,output_dim,depth*[truncation_dimension])

	return regressor


def construct_projected_resnet(input_projector, last_layer_weights, ranks,  name_prefix = ''):
	"""
	"""
	# last_layer_weights = build_POD_layer_arrays(data_dict,truncation_dimension = truncation_dimension,\
	# 									breadth_tolerance = breadth_tolerance,max_breadth = max_breadth)

	pod_resnet = projected_resnet(input_projector,last_layer_weights,\
									ranks = ranks,name_prefix = name_prefix)

	return pod_resnet, last_layer_weights

def construct_projected_dense(input_projector, last_layer_weights, depth, name_prefix = ''):
	"""
	"""
	# last_layer_weights = build_POD_layer_arrays(data_dict,truncation_dimension = truncation_dimension,\
	# 									breadth_tolerance = breadth_tolerance,max_breadth = max_breadth)
	truncation_dimension = last_layer_weights[0].shape[0]
	pod_dense_network = projected_dense(input_projector,last_layer_weights,\
									hidden_layer_dimensions = depth*[truncation_dimension],name_prefix = name_prefix)

	return pod_dense_network, last_layer_weights


	