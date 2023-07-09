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
import os

def load_data(data_dir,rescale = False,derivatives = False, n_data = None):
	assert os.path.isdir(data_dir)
	data_files = os.listdir(data_dir)
	data_files = [data_dir + file for file in data_files]

	m_files = []
	q_files = []
	mq_files = []
	for file in data_files:
		if 'ms_on_proc_' in file:
			m_files.append(file)
		if 'qs_on_proc_' in file:
			q_files.append(file)
		if 'mq_on_proc' in file:
			mq_files.append(file)

	if len(mq_files) == 0:
		ranks = [int(file.split(data_dir+'ms_on_proc_')[-1].split('.npy')[0]) for file in m_files]
	else:
		ranks = [int(file.split(data_dir+'mq_on_proc')[-1].split('.npz')[0]) for file in mq_files]
	max_rank = max(ranks)

	# Serially concatenate data
	if len(mq_files) == 0:
		m_data = np.load(data_dir+'ms_on_proc_0.npy')
		q_data = np.load(data_dir+'qs_on_proc_0.npy')
		for i in range(1,max_rank+1):
				appendage_m = np.load(data_dir+'ms_on_proc_'+str(i)+'.npy')
				m_data = np.concatenate((m_data,appendage_m))
				appendage_q = np.load(data_dir+'qs_on_proc_'+str(i)+'.npy')
				q_data = np.concatenate((q_data,appendage_q))
	else:
		npz_data = np.load(data_dir+'mq_on_proc0.npz')
		m_data = npz_data['m_data']
		q_data = npz_data['q_data']
		for i in range(1,max_rank+1):
			npz_data = np.load(data_dir+'mq_on_proc'+str(i)+'.npz')
			appendage_m = npz_data['m_data']
			appendage_q = npz_data['q_data']
			m_data = np.concatenate((m_data,appendage_m))
			q_data = np.concatenate((q_data,appendage_q))
      
	if n_data is not None:
		assert type(n_data) is int
		assert n_data <= m_data.shape[0], 'Requesting too much data, available number: '+str(m_data.shape[0])+', try again'
		m_data = m_data[:n_data]
		q_data = q_data[:n_data]
	if rescale:
		from sklearn import preprocessing
		m_data = preprocessing.scale(m_data)
		q_data = preprocessing.scale(q_data)

	data_dict = {'m_data': m_data,'q_data':q_data}

	if derivatives:
		U_files = []
		sigma_files = []
		V_files = []
		J_files = []
		for file in data_files:
			if 'Us_on_proc_' in file:
				U_files.append(file)
			if 'sigmas_on_proc_' in file:
				sigma_files.append(file)
			if 'Vs_on_proc_' in file:
				V_files.append(file)
			if 'J_on_proc' in file:
				J_files.append(file)

		if len(J_files) == 0:
			ranks = [int(file.split(data_dir+'sigmas_on_proc_')[-1].split('.npy')[0]) for file in sigma_files]
		else:
			ranks = [int(file.split(data_dir+'J_on_proc')[-1].split('.npz')[0]) for file in J_files]
		max_rank = max(ranks)

		if len(J_files) == 0:
			# Serially concatenate derivative data
			U_data = np.load(data_dir+'Us_on_proc_0.npy')
			sigma_data = np.load(data_dir+'sigmas_on_proc_0.npy')
			V_data = np.load(data_dir+'Vs_on_proc_0.npy')
			for i in range(1,max_rank+1):
				appendage_U = np.load(data_dir+'Us_on_proc_'+str(i)+'.npy')
				U_data = np.concatenate((U_data,appendage_U))
				appendage_sigma = np.load(data_dir+'sigmas_on_proc_'+str(i)+'.npy')
				sigma_data = np.concatenate((sigma_data,appendage_sigma))
				appendage_V = np.load(data_dir+'Vs_on_proc_'+str(i)+'.npy')
				V_data = np.concatenate((V_data,appendage_V))
		else:
			Jnpz_data = np.load(data_dir+'J_on_proc0.npz')
			U_data = Jnpz_data['U_data']
			sigma_data = Jnpz_data['sigma_data']
			V_data = Jnpz_data['V_data']
			for i in range(1,max_rank+1):
				Jnpz_data = np.load(data_dir+'J_on_proc'+str(i)+'.npz')
				appendage_U = Jnpz_data['U_data']
				appendage_sigma = Jnpz_data['sigma_data']
				appendage_V = Jnpz_data['V_data']
				U_data = np.concatenate((U_data,appendage_U))
				sigma_data = np.concatenate((sigma_data,appendage_sigma))
				V_data = np.concatenate((V_data,appendage_V))

		if n_data is not None:
			assert type(n_data) is int
			U_data = U_data[:n_data]
			sigma_data = sigma_data[:n_data]
			V_data = V_data[:n_data]

		if rescale:
			raise NotImplementedError('This needs to be thought out with care')

		data_dict['U_data'] = U_data
		data_dict['sigma_data'] = sigma_data
		data_dict['V_data'] = V_data
		

	return data_dict


def get_projectors(data_dir,as_input_tolerance=1e-4,as_output_tolerance=1e-4,\
					kle_tolerance = 1e-4,pod_tolerance = 1e-4,\
					 fixed_input_rank = 0, fixed_output_rank = 0, mixed_output = True, verbose = False):
	projector_dictionary = {}
	try:
		################################################################################
		# Derivative Informed Input Subspace
		AS_input_projector = np.load(data_dir+'AS_input_projector.npy')
		if verbose:
			print('AS input projector shape before truncation = ', AS_input_projector.shape)
		if fixed_input_rank > 0:
			AS_input_projector = AS_input_projector[:,:fixed_input_rank]
		else:
			AS_input_d = np.load(data_dir+'AS_d_GN.npy')
			AS_input_projector = AS_input_projector[:,np.where(AS_input_d>as_input_tolerance)[0]]
		if verbose:
			print('AS input projector shape after truncation = ', AS_input_projector.shape)
		projector_dictionary['AS_input'] = AS_input_projector
		################################################################################
		# Derivative Informed Output Subspace
		AS_output_projector = np.load(data_dir+'AS_output_projector.npy')
		
		if verbose:
			print('AS output projector shape before truncation = ', AS_output_projector.shape)
		if fixed_output_rank > 0:
			AS_output_projector = AS_output_projector[:,:fixed_output_rank]
		else:
			AS_output_d = np.load(data_dir+'AS_d_NG.npy')
			AS_output_projector = AS_output_projector[:,np.where(AS_output_d>as_output_tolerance)[0]]
		if verbose:
			print('AS output projector shape after truncation = ', AS_output_projector.shape)
		projector_dictionary['AS_output'] = AS_output_projector
	except:
		print('Active subspaces did not load')
	try:
		################################################################################
		# KLE Input Subspace
		KLE_projector = np.load(data_dir+'KLE_projector.npy')
		if verbose:
			print('KLE projector shape before truncation = ', KLE_projector.shape)
		if fixed_input_rank > 0:
			KLE_projector = KLE_projector[:,:fixed_input_rank]
		else:
			KLE_d = np.load(data_dir+'KLE_d.npy')
			KLE_projector = KLE_projector[:,np.where(KLE_d>kle_tolerance)[0]]
		if verbose:
			print('KLE projector shape after truncation = ', KLE_projector.shape)
		projector_dictionary['KLE'] = KLE_projector
	except:
		print('KLE did not load')
	try:
		################################################################################
		# POD Output Subspace
		POD_projector = np.load(data_dir+'POD_projector.npy')
		if verbose:
			print('POD projector shape before truncation = ', POD_projector.shape)
		if fixed_output_rank > 0:
			POD_projector = POD_projector[:,:fixed_output_rank]
		else:
			POD_d = np.load(data_dir+'POD_d.npy')
			POD_projector = POD_projector[:,np.where(POD_d>pod_tolerance)[0]]
		if verbose:
			print('POD projector shape after truncation = ', POD_projector.shape)
		projector_dictionary['POD'] = POD_projector
	except:
		print('Pre-computed POD did not load')
	return projector_dictionary


################################################################################
# Training related data functions


def shuffle_single_data(data_dictionary,train_size,val_size,test_size,seed = 0,copy = True,burn_in = 0):
	"""
	Shuffling from the same fixed data, so train + val + test should be strictly < total
	"""

	assert type(train_size) is int
	assert type(val_size) is int
	assert type(test_size) is int


	all_m_data = data_dictionary['m_data']
	all_q_data = data_dictionary['q_data']

	total_data_size = all_m_data.shape[0]

	assert train_size + val_size + test_size <= total_data_size

	# Partition the data into train, val and test here.
	random_state = np.random.RandomState(seed = 0)

	for burn in range(burn_in):
		_ = random_state.permutation(total_data_size)


	random_indices = random_state.permutation(total_data_size)
	# First extract the testing data
	test_indices = random_indices[:test_size]

	remaining_indices = random_indices[test_size:][: train_size + val_size]

	train_indices = remaining_indices[:train_size]
	validation_indices = remaining_indices[train_size:]


	shuffle_dict = {'input_train':all_m_data[train_indices],'output_train':all_q_data[train_indices],
					'input_val':all_m_data[validation_indices], 'output_val':all_q_data[validation_indices],
					'input_test':all_m_data[test_indices], 'output_test':all_q_data[test_indices]}

	return shuffle_dict




def shuffle_data(data_dictionary,train_size,val_size,test_size,n_shuffles = 10,seed = 0,copy = True,burn_in = 0):
	"""
	Shuffling from the same fixed data, so train + val + test should be strictly < total
	"""

	assert type(train_size) is int
	assert type(val_size) is int
	assert type(test_size) is int

	multi_shuffled_data = {}

	all_m_data = data_dictionary['m_data']
	all_q_data = data_dictionary['q_data']

	total_data_size = all_m_data.shape[0]

	assert train_size + val_size + test_size <= total_data_size

	if train_size + val_size + test_size == total_data_size:
		assert n_shuffles == 1 


	# Partition the data into train, val and test here.
	random_state = np.random.RandomState(seed = 0)

	for burn in range(burn_in):
		_ = random_state.permutation(total_data_size)

	for shuffle in range(n_shuffles):
		random_indices = random_state.permutation(total_data_size)
		# First extract the testing data
		test_indices = random_indices[-test_size:]

		remaining_indices = random_indices[test_size:][: train_size + val_size]

		train_indices = remaining_indices[:train_size]
		validation_indices = remaining_indices[train_size:]

		if copy:
			shuffle_dict = {'input_train':np.copy(all_m_data[train_indices]),'output_train':np.copy(all_q_data[train_indices]),
							'input_val':np.copy(all_m_data[validation_indices]), 'output_val':np.copy(all_q_data[validation_indices]),
							'input_test':np.copy(all_m_data[test_indices]), 'output_test':np.copy(all_q_data[test_indices])}
		else:
			shuffle_dict = {'input_train':all_m_data[train_indices],'output_train':all_q_data[train_indices],
							'input_val':all_m_data[validation_indices], 'output_val':all_q_data[validation_indices],
							'input_test':all_m_data[test_indices], 'output_test':all_q_data[test_indices]}

		multi_shuffled_data[shuffle] = shuffle_dict

	return multi_shuffled_data



def flatten_data(data_dict,target_rank = 80,batch_rank = 8,order_random = True,diagonalize_sigma = True,\
				key_map = {'m_data':'m_data','q_data':'q_data','U_data':'U_data','sigma_data':'sigma_data','V_data':'V_data'},\
					seed = 0,burn_in = 0,verbose = False, independent_sampling = False):
	assert target_rank%batch_rank == 0
	batch_factor = int(target_rank/batch_rank)
	# Load data
	m_data = data_dict[key_map['m_data']]
	q_data = data_dict[key_map['q_data']]
	U_data = data_dict[key_map['U_data']][:,:,:target_rank]
	sigma_data = data_dict[key_map['sigma_data']][:,:target_rank]
	V_data = data_dict[key_map['V_data']][:,:,:target_rank]
	if independent_sampling:
		# Diagonalize sigma here
		sigma_data_ = np.zeros(sigma_data_.shape + sigma_data_.shape[-1:])
		for i in range(sigma_data_.shape[0]):
		    sigma_data_[i] = np.diag(sigma_data[i])

		# Shuffle up the ranks?
		if order_random:
			random_state = np.random.RandomState(seed = seed)
			for burn in range(burn_in):
				_ = random_state.permutation(target_rank)
			indices_U = random_state.permutation(target_rank)
			indices_V = random_state.permutation(target_rank)

			U_data = U_data[:,:,indices_U]
			sigma_data_ = sigma_data[:,indices_U,:][:,:,indices_V]
			V_data = V_data[:,:,indices_V]
		
		# Infer dimensions
		n_data = m_data.shape[0]
		total_rank = sigma_data.shape[-1]
		dM = V_data.shape[1]
		dQ = U_data.shape[1]
		n_batch_data = batch_factor*n_data
		mnew = np.repeat(m_data,batch_factor,axis = 0)
		qnew = np.repeat(q_data,batch_factor,axis = 0)

		Unew = U_data.transpose((0,2,1)).reshape((n_batch_data,batch_rank,dQ)).transpose(0,2,1)

		sigmanew = np.zeros((n_batch_data,batch_rank,batch_rank))
		for i in range(n_data):
			for j in range(batch_factor):
				sigmanew[i*batch_factor+j,:,:] = sigma_data[i,j*batch_rank:(j+1)*batch_rank,j*batch_rank:(j+1)*batch_rank]

		Vnew = V_data.transpose((0,2,1)).reshape((n_batch_data,batch_rank,dM)).transpose(0,2,1)

	else:
		# Shuffle up the ranks?
		if order_random:
			random_state = np.random.RandomState(seed = seed)
			for burn in range(burn_in):
				_ = random_state.permutation(target_rank)
			indices = random_state.permutation(target_rank)
			U_data = U_data[:,:,indices]
			sigma_data = sigma_data[:,indices]
			V_data = V_data[:,:,indices]
		
		# Infer dimensions
		n_data = m_data.shape[0]
		total_rank = sigma_data.shape[-1]
		dM = V_data.shape[1]
		dQ = U_data.shape[1]
		n_batch_data = batch_factor*n_data
		if verbose:
			print('n_data = ',n_data)
			print('n_batch_data = ',n_batch_data)
			print('total_rank = ',total_rank)
			print('dM = ',dM)
			print('dQ = ',dQ)

		mnew = np.repeat(m_data,batch_factor,axis = 0)
		qnew = np.repeat(q_data,batch_factor,axis = 0)
		Unew = U_data.transpose((0,2,1)).reshape((n_batch_data,batch_rank,dQ)).transpose(0,2,1)

		sigmanew = sigma_data.reshape((n_batch_data,batch_rank))

		if diagonalize_sigma:
			sigma_new = np.zeros(sigmanew.shape + sigmanew.shape[-1:])
			for i in range(sigmanew.shape[0]):
			    sigma_new[i] = np.diag(sigmanew[i])
			sigmanew = sigma_new

		Vnew = V_data.transpose((0,2,1)).reshape((n_batch_data,batch_rank,dM)).transpose(0,2,1)

	if verbose:	
		print('mnew.shape = ',mnew.shape)
		print('qnew.shape = ',qnew.shape)
		print('Unew.shape = ',Unew.shape)
		print('sigmanew.shape = ',sigmanew.shape)
		print('Vnew.shape = ',Vnew.shape)
	
	return {key_map['m_data']:mnew,key_map['q_data']:qnew,\
			key_map['U_data']:Unew,key_map['sigma_data']:sigmanew,\
			key_map['V_data']:Vnew}


def train_test_split(data_dict,n_train,seed = 0):
	"""
	Assuming that the total is n_train + n_test
	"""
	m_data = data_dict['m_data']
	q_data = data_dict['q_data']
	n_data = m_data.shape[0]

	# Not using random for the time being in order to ensure the same test data are always used.
	# If random sampling of the training data is implemented later, make sure n_test is passed in
	# and the test data are set aside first before shuffling.
	random_state = np.random.RandomState(seed = seed)
	indices = random_state.permutation(n_data)

	m_train = m_data[-n_train:]
	q_train = q_data[-n_train:]

	m_test = m_data[:-n_train]
	q_test = q_data[:-n_train]


	if 'U_data' in data_dict.keys():
		U_data = data_dict['U_data']
		sigma_data = data_dict['sigma_data']
		V_data = data_dict['V_data']
	
		U_train = U_data[-n_train:]
		sigma_train = sigma_data[-n_train:]
		V_train = V_data[-n_train:]

		U_test = U_data[:-n_train]
		sigma_test = sigma_data[:-n_train]
		V_test = V_data[:-n_train]

		train_dict = {'m_data':m_train,'q_data':q_train,'U_data':U_train,'sigma_data':sigma_train,'V_data':V_train}
		test_dict = {'m_data':m_test,'q_data':q_test,'U_data':U_test,'sigma_data':sigma_test,'V_data':V_test}

	else:
		train_dict = {'m_data':m_train,'q_data':q_train}
		test_dict = {'m_data':m_test,'q_data':q_test}

	return train_dict, test_dict	


def remap_jacobian_data(data_dict):
    new_dict = {}
    new_dict['m_data'] = data_dict['m_data']
    new_dict['q_data'] = data_dict['q_data']
    
    U_data = data_dict['U_data']
    sigma_data = data_dict['sigma_data']
    V_data = data_dict['V_data']
    
    n_data, dQ,rank = U_data.shape
    _,dM,_ = V_data.shape
    J_data = np.zeros((n_data,dQ,dM))
    for i in range(n_data):
        J_data[i] = U_data[i]@(sigma_data[i]*V_data[i]).T
    new_dict['J_data'] = J_data
    return new_dict

def remap_jacobian_data_with_orth_V(data_dict):
    new_dict = {}
    V_data = data_dict['V_data']
    V_data_new = np.zeros_like(V_data)
    for i in range(V_data.shape[0]):
        # print(i)
        V_data_new[i,:,:] = np.linalg.qr(V_data[i,:,:])[0]
    new_dict['V_data'] = V_data_new
    
    new_dict['m_data'] = data_dict['m_data']
    new_dict['q_data'] = data_dict['q_data']
    
    U_data = data_dict['U_data']
    sigma_data = data_dict['sigma_data']
    V_data = data_dict['V_data']
    
    n_data, dQ,rank = U_data.shape
    _,dM,_ = V_data.shape
    J_data = np.zeros((n_data,dQ,dM))
    for i in range(n_data):
        J_data[i] = U_data[i]@(sigma_data[i]*V_data[i]).T
    new_dict['J_data'] = J_data
    # For orth conditions
    # Extra allocation for the sake of clarity with einsums
    ndata,dQ,dM = J_data.shape
    JV_data = np.zeros((ndata,dQ,dQ))
    JV_data[:,:,:] = np.einsum('ijk,ikl->ijl',J_data,V_data)

    JVVT_data = np.zeros_like(J_data)
    JVVT_data[:,:,:] = np.einsum('ijk,ilk->ijl',JV_data,V_data)

    new_dict['JVVT_data'] = JVVT_data
    
    return new_dict
