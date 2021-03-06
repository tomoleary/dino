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

def low_rank_layer(input_x,rank = 8,activation = 'softplus',name_prefix = None,zeros = True):
	"""
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




def projected_resnet(input_projector,last_layer_weights,ranks = [],\
							trainable = False, name_prefix = ''):
	"""
	"""
	input_dim,reduced_input_dim = input_projector.shape
	assert type(last_layer_weights) is list
	assert len(last_layer_weights) == 2
	reduced_output_dim, output_dim = last_layer_weights[0].shape
	# Check shape interface conditions
	assert len(last_layer_weights[1].shape) == 1
	assert last_layer_weights[1].shape[0] == output_dim

	
	# input_dim, reduced_input_dim = input_projector.shape
	# output_dim, reduced_output_dim = output_projector.shape

	input_data = tf.keras.layers.Input(shape=(input_dim,),name = name_prefix+'network_input')

	input_proj_layer = tf.keras.layers.Dense(reduced_input_dim,name = name_prefix+'input_proj_layer',use_bias = False)(input_data)

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

	output_layer = tf.keras.layers.Dense(output_dim,name = name_prefix+'output_layer')(z)

	regressor = tf.keras.models.Model(input_data,output_layer)

	########################################################################
	# Modify input layer by setting weights and setting trainable boolean
	regressor.get_layer(name_prefix+'input_proj_layer').trainable =  trainable
	regressor.get_layer(name_prefix+'input_proj_layer').set_weights([input_projector])

	regressor.get_layer(name_prefix + 'output_layer').trainable =  True
	regressor.get_layer(name_prefix + 'output_layer').set_weights(last_layer_weights)

	return regressor




def projected_dense(input_projector,last_layer_weights,hidden_layer_dimensions = [],\
		trainable = False, name_prefix = ''):
	"""
	"""
	input_dim,reduced_input_dim = input_projector.shape
	assert type(last_layer_weights) is list
	assert len(last_layer_weights) == 2
	reduced_output_dim, output_dim = last_layer_weights[0].shape
	# Check shape interface conditions
	assert len(last_layer_weights[1].shape) == 1
	assert last_layer_weights[1].shape[0] == output_dim
	assert hidden_layer_dimensions[-1] == reduced_output_dim

	input_data = tf.keras.layers.Input(shape=(input_dim,), name = name_prefix+'input_data')

	input_proj_layer = tf.keras.layers.Dense(reduced_input_dim,name = name_prefix+'input_proj_layer',use_bias = False)(input_data)

	z =  tf.keras.layers.Dense(reduced_input_dim,activation = 'softplus',name=name_prefix + 'dense_reduction_layer')(input_proj_layer)
	for i,hidden_layer_dimension in enumerate(hidden_layer_dimensions):
		z = tf.keras.layers.Dense(hidden_layer_dimension,activation = 'softplus',name = name_prefix+'inner_layer_'+str(i))(z)
	output_layer = tf.keras.layers.Dense(output_dim,name = name_prefix + 'output_layer')(z)

	regressor = tf.keras.models.Model(input_data,output_layer,name = 'output_proj_layer')

	regressor.get_layer(name_prefix+'input_proj_layer').trainable =  trainable
	regressor.get_layer(name_prefix+'input_proj_layer').set_weights([input_projector])

	regressor.get_layer(name_prefix + 'output_layer').trainable =  True
	regressor.get_layer(name_prefix + 'output_layer').set_weights(last_layer_weights)

	return regressor



def generic_dense(input_dim,output_dim,n_hidden_neurons, name_prefix = ''):
	"""
	"""
	assert type(n_hidden_neurons) is list

	input_data = tf.keras.layers.Input(shape=(input_dim,), name = name_prefix+'input_data')
	z = input_data
	for i,n_hidden_neuron in enumerate(n_hidden_neurons):	
		z = tf.keras.layers.Dense(n_hidden_neuron, activation='softplus',name = name_prefix+'inner_layer_'+str(i))(z)
	output = tf.keras.layers.Dense(output_dim,name = name_prefix+'final_dense')(z)
	regressor = tf.keras.models.Model(input_data, output)
	return regressor



