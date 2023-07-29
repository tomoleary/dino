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
from ..construction.jacobianConstruction import  equip_model_with_sketched_jacobian,\
															 equip_model_with_full_jacobian

from ..construction.observableConstruction import observable_network_loader

from ..construction.trainingUtilities import normalized_mse, l2_accuracy, flatten_data, train_test_split,\
																normalized_mse_matrix, f_accuracy_matrix
from ..construction.dataUtilities import load_data, remap_jacobian_data, remap_jacobian_data_with_orth_V




def evaluateJacobianNetwork(settings,jacobian_network = None,file_name = None,split_data = None, data_dir = None):
	"""
	Post-processing of Jacobian accuracies
	"""
	if jacobian_network is None:
		if file_name is None and settings['network_name'] is not None:
			file_name  = settings['weights_dir']+settings['network_name']+'.pkl'
		jacobian_network = observable_network_loader(settings,file_name = file_name)

	print(jacobian_network.summary())


	if split_data is None:
		n_data = settings['train_data_size'] + settings['test_data_size']
		problem_settings = settings['problem_settings']
		if data_dir is None:
			data_dir = '../data/'+problem_settings['formulation']+'_n_obs_'+str(problem_settings['ntargets'])+\
					'_g'+str(problem_settings['gamma'])+'_d'+str(problem_settings['delta'])+'_nx'+str(problem_settings['nx'])+'/'

		all_data = load_data(data_dir,rescale = False,n_data = n_data,derivatives = True)
		unflattened_train_dict, unflattened_test_dict = train_test_split(all_data,n_train = settings['train_data_size'])
		if settings['full_JVVT']:
			train_dict = remap_jacobian_data_with_orth_V(unflattened_train_dict)
			test_dict = remap_jacobian_data_with_orth_V(unflattened_test_dict)

			print('shape of V_data = ',train_dict['V_data'].shape)

			input_train = [train_dict['m_data'],train_dict['V_data']]
			output_train = [train_dict['q_data'],train_dict['J_data'],train_dict['JVVT_data']]

			input_test = [test_dict['m_data'],test_dict['V_data']]
			output_test = [test_dict['q_data'],test_dict['J_data'],test_dict['JVVT_data']]

		elif settings['full_jacobian']:
			train_dict = remap_jacobian_data(unflattened_train_dict)
			test_dict = remap_jacobian_data(unflattened_test_dict)
			input_train = [train_dict['m_data']]
			output_train = [train_dict['q_data'],train_dict['J_data']]
			input_test = [test_dict['m_data']]
			output_test = [test_dict['q_data'],test_dict['J_data']]

		else:
			train_dict = flatten_data(unflattened_train_dict,target_rank = settings['target_rank'],batch_rank = settings['batch_rank'])
			test_dict = flatten_data(unflattened_test_dict,target_rank = settings['target_rank'],batch_rank = settings['batch_rank'])
			input_train = [train_dict['m_data'],train_dict['U_data'],train_dict['V_data']]
			output_train = [train_dict['q_data'],train_dict['sigma_data']]
			input_test = [test_dict['m_data'],test_dict['U_data'],test_dict['V_data']]
			output_test = [test_dict['q_data'],test_dict['sigma_data']]
	else:
		input_train = split_data['input_train']
		output_train = split_data['output_train']
		input_test = split_data['input_test']
		output_test = split_data['output_test']

	n_train,dM = input_train[0].shape
	n_test,dQ = output_test[0].shape


	if settings['full_JVVT']:
		for i in range(10):
			print('RUNNING FOR FULL JACOBIAN with VVT')
		settings['opt_parameters']['loss_weights'] = [1.,1.,1.]
		V_rank = train_dict['V_data'].shape[-1]
		jacobian_network = equip_model_with_full_jacobian_and_right_nullspace(jacobian_network,rank = V_rank,\
																			name_prefix = settings['name_prefix'])
	elif settings['full_jacobian']:
		for i in range(10):
			print('RUNNING FOR FULL JACOBIAN')
		jacobian_network = equip_model_with_full_jacobian(jacobian_network,name_prefix = settings['name_prefix'])

	else:
		jacobian_network = equip_model_with_sketched_jacobian(jacobian_network,settings['batch_rank'],\
												 name_prefix = settings['name_prefix'])


	if settings['full_JVVT']:
		losses = [normalized_mse] +(len(jacobian_network.outputs)-1)*[normalized_mse_matrix]
		metrics = [l2_accuracy]+[f_accuracy_matrix]
	else:
		losses = [normalized_mse] +[normalized_mse_matrix]
		metrics = [l2_accuracy]+[f_accuracy_matrix]

	jacobian_network.compile(optimizer='adam',loss=losses,\
						  loss_weights = settings['opt_parameters']['loss_weights'],metrics=metrics)

	print('output_train[0].shape = ',output_train[0].shape)
	print('output_train[1].shape = ',output_train[1].shape)

	eval_train = jacobian_network.evaluate(input_train,output_train,verbose=True)
	eval_dict_train = {out: eval_train[i] for i, out in enumerate(jacobian_network.metrics_names)}
	# print('eval train = ',eval_train)
	eval_test = jacobian_network.evaluate(input_test,output_test,verbose=True)
	eval_dict_test = {out: eval_test[i] for i, out in enumerate(jacobian_network.metrics_names)}

	print('eval test = ',eval_test)

	if settings['full_JVVT']:
		# 0: total_loss, 1: output1 loss, 2: output2loss, 3: output3loss 4: l2 acc output1 5: f acc output1 6: l2 acc output2 7: f acc output2
		# 8: l2 acc output 3 9: f acc output3

		print('After loading: l2, full h1, reduced h1 training accuracies = ', eval_train[4], eval_train[7],eval_train[9])
		print('After loading: l2, full h1, reduced h1 testing accuracies =  ', eval_test[4], eval_test[7],eval_test[9])	
		eval_logger = {'l2_train': eval_train[4],'h1_train': eval_train[7],'l2_test': eval_train[4],'h1_test': eval_train[7],\
							'reduced_h1_train':eval_train[9],'reduced_h1_test':eval_test[9]}
		eval_logger['train_eval'] = eval_dict_train
		eval_logger['test_eval'] = eval_dict_test
	else:
		# 0: total_loss, 1: output1 loss, 2: output2loss, 3: l2 acc out1, 4: f acc out1, 5: l2 acc out2, 6: f acc out2
		print('After loading: l2, h1 training accuracies = ', eval_train[3], eval_train[6])
		print('After loading: l2, h1 testing accuracies =  ', eval_test[3], eval_test[6])	
		eval_logger = {'l2_train': eval_train[3],'h1_train': eval_train[6],'l2_test': eval_train[3],'h1_test': eval_train[6]}
		eval_logger['train_eval'] = eval_dict_train
		eval_logger['test_eval'] = eval_dict_test


	return eval_logger


