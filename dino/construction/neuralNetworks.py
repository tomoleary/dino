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

import tensorflow as tf


def projected_dense(input_projector = None,last_layer_weights = None,hidden_layer_dimensions = [],\
				compat_layer = True, first_layer_trainable = False, last_layer_trainable = False, name_prefix = '',\
				reduced_input_dim = None,reduced_output_dim = None):
	"""
	This function creates networks based on the reduced basis dense architecture
	"""
	# Input reduced basis stuff
	if input_projector is None:
		input_data = tf.keras.layers.Input(shape=(reduced_input_dim,), name = name_prefix+'input_data')
		input_proj_layer = input_data
	else:
		input_dim,reduced_input_dim = input_projector.shape
		input_data = tf.keras.layers.Input(shape=(input_dim,), name = name_prefix+'input_data')
		input_proj_layer = tf.keras.layers.Dense(reduced_input_dim,name = name_prefix+'input_proj_layer',use_bias = False)(input_data)

	# Output reduced basis stuff
	if last_layer_weights is not None:
		assert type(last_layer_weights) is list
		assert len(last_layer_weights) == 2
		reduced_output_dim, output_dim = last_layer_weights[0].shape
		# Check shape interface conditions
		assert len(last_layer_weights[1].shape) == 1
		assert last_layer_weights[1].shape[0] == output_dim
	else:
		assert reduced_output_dim is not None
	assert hidden_layer_dimensions[-1] == reduced_output_dim

		

	z =  tf.keras.layers.Dense(reduced_input_dim,activation = 'softplus',name=name_prefix + 'dense_reduction_layer')(input_proj_layer)
	for i,hidden_layer_dimension in enumerate(hidden_layer_dimensions):
		z = tf.keras.layers.Dense(hidden_layer_dimension,activation = 'softplus',name = name_prefix+'inner_layer_'+str(i))(z)

	if compat_layer:
		z = tf.keras.layers.Dense(reduced_output_dim,name = name_prefix+'output_compat_layer',use_bias = False)(z)

	if last_layer_weights is None:
		assert compat_layer, print('In this case we need the compat layer')
		output_layer = z
	else:
		output_layer = tf.keras.layers.Dense(output_dim,name = name_prefix + 'output_layer')(z)

	regressor = tf.keras.models.Model(input_data,output_layer)

	if input_projector is not None:
		regressor.get_layer(name_prefix+'input_proj_layer').trainable =  first_layer_trainable
		regressor.get_layer(name_prefix+'input_proj_layer').set_weights([input_projector])

	if last_layer_weights is not None:
		regressor.get_layer(name_prefix + 'output_layer').trainable =  last_layer_trainable
		regressor.get_layer(name_prefix + 'output_layer').set_weights(last_layer_weights)

	return regressor



def generic_dense(input_dim,output_dim,n_hidden_neurons, name_prefix = ''):
	"""
	This creates a generic dense architecture
	"""
	assert type(n_hidden_neurons) is list

	input_data = tf.keras.layers.Input(shape=(input_dim,), name = name_prefix+'input_data')
	z = input_data
	for i,n_hidden_neuron in enumerate(n_hidden_neurons):	
		z = tf.keras.layers.Dense(n_hidden_neuron, activation='softplus',name = name_prefix+'inner_layer_'+str(i))(z)
	output = tf.keras.layers.Dense(output_dim,name = name_prefix+'final_dense')(z)
	regressor = tf.keras.models.Model(input_data, output)
	return regressor



def low_rank_layer(input_x,rank = 8,activation = 'softplus',name_prefix = None,zeros = True):
	"""
	Low rank layer used in resnet construction
	"""
	output_shape = input_x.shape
	assert len(output_shape) == 2
	output_dim = output_shape[-1]
	if name_prefix is None:
		if zeros:
			intermediate = tf.keras.layers.Dense(rank,activation = activation)(input_x)
			return tf.keras.layers.Dense(output_dim,
										kernel_initializer = tf.keras.initializers.Zeros(),
										bias_initializer = tf.keras.initializers.Zeros())(intermediate)
		else:
			intermediate = tf.keras.layers.Dense(rank,activation = activation)(input_x)
			return tf.keras.layers.Dense(output_dim)(intermediate)
	else:
		if zeros:
			intermediate = tf.keras.layers.Dense(rank,activation = activation,name = name_prefix+'low_rank_residual_in')(input_x)
			return tf.keras.layers.Dense(output_dim,name = name_prefix+'low_rank_residual_out',
										kernel_initializer = tf.keras.initializers.Zeros(),
										bias_initializer = tf.keras.initializers.Zeros())(intermediate)
		else:
			intermediate = tf.keras.layers.Dense(rank,activation = activation,name = name_prefix+'low_rank_residual_in')(input_x)
			return tf.keras.layers.Dense(output_dim,name = name_prefix+'low_rank_residual_out')(intermediate)




def projected_resnet(input_projector,last_layer_weights,ranks = [],compat_layer = True,\
							first_layer_trainable = False, last_layer_trainable = True, name_prefix = '',\
						reduced_input_dim = None):
	"""
	This function creates a network based on the reduced basis ResNet architecture.
	"""
	if input_projector is None:
		input_data = tf.keras.layers.Input(shape=(reduced_input_dim,), name = name_prefix+'input_data')
		input_proj_layer = input_data
	else:
		input_dim,reduced_input_dim = input_projector.shape
		input_data = tf.keras.layers.Input(shape=(input_dim,), name = name_prefix+'input_data')
		input_proj_layer = tf.keras.layers.Dense(reduced_input_dim,name = name_prefix+'input_proj_layer',use_bias = False)(input_data)

	assert type(last_layer_weights) is list
	assert len(last_layer_weights) == 2
	reduced_output_dim, output_dim = last_layer_weights[0].shape
	# Check shape interface conditions
	assert len(last_layer_weights[1].shape) == 1
	assert last_layer_weights[1].shape[0] == output_dim
	assert hidden_layer_dimensions[-1] == reduced_output_dim

	z = input_proj_layer
	# The question is whether or not to handle a possible dimension mismatch
	# before or after the residual portion of the network
	if not reduced_input_dim == reduced_output_dim:
		# Meet the square ResNet requirements
		z = tf.keras.layers.Dense(reduced_output_dim,name = name_prefix+'input_resnet_interface',use_bias=False)(z)

	for i,rank in enumerate(ranks):
		z = tf.keras.layers.Add(name = name_prefix+'add'+str(i))([low_rank_layer(z,rank = rank,activation = 'softplus',\
																						name_prefix=name_prefix+str(i)),z])

	# If not handling a dimension mismatch before then do it here. 
	# z = tf.keras.layers.Dense(reduced_output_dim)(z)
	if compat_layer:
		z = tf.keras.layers.Dense(reduced_output_dim,name = name_prefix+'output_compat_layer',use_bias = False)(z)

	output_layer = tf.keras.layers.Dense(output_dim,name = name_prefix+'output_layer')(z)

	regressor = tf.keras.models.Model(input_data,output_layer)

	########################################################################
	# Modify input layer by setting weights and setting trainable boolean
	regressor.get_layer(name_prefix+'input_proj_layer').trainable =  first_layer_trainable
	regressor.get_layer(name_prefix+'input_proj_layer').set_weights([input_projector])

	regressor.get_layer(name_prefix + 'output_layer').trainable =  last_layer_trainable
	regressor.get_layer(name_prefix + 'output_layer').set_weights(last_layer_weights)

	return regressor








