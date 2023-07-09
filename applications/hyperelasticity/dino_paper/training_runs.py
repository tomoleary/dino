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

import os

def default_settings():
	settings = {}
	settings['total_epochs'] = 100
	settings['total_epochs'] = 1

	# Architecture selection
	settings['architecture'] = 'as_dense'
	# Specific to reduced basis networks
	settings['fixed_input_rank'] = 100
	settings['fixed_output_rank'] = 50

	# Optimization formulation settings
	settings['train_full_jacobian'] = 1
	settings['l2_weight'] = 1.0
	settings['h1_weight'] = 1.0
	# truncated H1 w/ and w/o matrix subsampling
	settings['target_rank'] = 100
	settings['batch_rank'] = 100
	return settings

def build_command(settings,command = 'python dino_training.py'):
	command += ' '
	for key,value in settings.items():
		command += '-'+key
		command += ' '
		command += str(value)
		command += ' '
	return command

################################################################################
# Architectures: Generic Dense, DIPNet (100-50, 200-100)
# Formulations: L2, full H1, truncated H1, truncated H1 w/ matrix subsampling (MS)
# Amount of training data: ndatas = [16,64,256,1024,4096,8192-1024][:-1]

ndatas = [16,64,256,1024,4096,8192-1024][:-1]


################################################################################
# DIPNet training runs

rb_dims = [(100,50),(200,100)]

for ndata in ndatas:
	for rM, rQ in rb_dims:
		settings = default_settings()
		settings['architecture'] = 'as_dense'
		initial_name0 = 'DIPNet'
		# Specific to this loop
		settings['train_data_size'] = ndata
		settings['fixed_input_rank'] = rM
		settings['fixed_output_rank'] = rQ
		initial_name = initial_name0 + str(rM)+'-'+str(rQ)
		# L2 training
		settings['h1_weight'] = 0.0
		settings['network_name'] = initial_name + '_ndata'+str(ndata)
		settings['network_name'] += 'l2'
		print(80*'#')
		print(build_command(settings))
		print(80*'#')
		os.system(build_command(settings))
		# Full H1 training
		settings['h1_weight'] = 1.0
		settings['train_full_jacobian'] = 1
		settings['network_name'] = initial_name + '_ndata'+str(ndata)
		settings['network_name'] += 'fullh1'
		print(80*'#')
		print(build_command(settings))
		print(80*'#')
		os.system(build_command(settings, command = 'python fast_training.py'))
		# Truncated H1 training
		settings['h1_weight'] = 1.0
		settings['train_full_jacobian'] = 0
		settings['network_name'] = initial_name + '_ndata'+str(ndata)
		settings['network_name'] += 'th1'
		print(80*'#')
		print(build_command(settings))
		print(80*'#')
		os.system(build_command(settings))
		# Truncated H1 training w/ matrix-subsampling
		settings['h1_weight'] = 1.0
		settings['train_full_jacobian'] = 0
		settings['batch_rank'] = 10
		settings['network_name'] = initial_name + '_ndata'+str(ndata)
		settings['network_name'] += 'th1ms'
		print(80*'#')
		print(build_command(settings))
		print(80*'#')
		os.system(build_command(settings))



################################################################################
# Generic dense training runs

for ndata in ndatas:
	settings = default_settings()
	settings['architecture'] = 'generic_dense'
	initial_name = 'generic_dense'
	# Specific to this loop
	settings['train_data_size'] = ndata
	# L2 training
	settings['h1_weight'] = 0.0
	settings['network_name'] = initial_name + '_ndata'+str(ndata)
	settings['network_name'] += 'l2'
	print(80*'#')
	print(build_command(settings))
	print(80*'#')
	os.system(build_command(settings))
	# Full H1 training
	settings['h1_weight'] = 1.0
	settings['train_full_jacobian'] = 1
	settings['network_name'] = initial_name + '_ndata'+str(ndata)
	settings['network_name'] += 'fullh1'
	print(80*'#')
	print(build_command(settings))
	print(80*'#')
	os.system(build_command(settings))
	# Truncated H1 training
	settings['h1_weight'] = 1.0
	settings['train_full_jacobian'] = 0
	settings['network_name'] = initial_name + '_ndata'+str(ndata)
	settings['network_name'] += 'th1'
	print(80*'#')
	print(build_command(settings))
	print(80*'#')
	os.system(build_command(settings))
	# Truncated H1 training w/ matrix-subsampling
	settings['h1_weight'] = 1.0
	settings['train_full_jacobian'] = 0
	settings['batch_rank'] = 10
	settings['network_name'] = initial_name + '_ndata'+str(ndata)
	settings['network_name'] += 'th1ms'
	print(80*'#')
	print(build_command(settings))
	print(80*'#')
	os.system(build_command(settings))


