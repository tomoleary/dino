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
import numpy as np
import time, datetime
import pickle

from argparse import ArgumentParser

# Arguments to be parsed from the command line execution
parser = ArgumentParser(add_help=True)

parser.add_argument("-full_batch_jacobian", dest='full_batch_jacobian',required=False, default = 1,  help="execute full batch runs?",type=int)
parser.add_argument("-full_jacobian_training", dest='full_jacobian_training',required=False, default = 0,  help="execute full J training runs?",type=int)
parser.add_argument("-mini_batch_jacobian", dest='mini_batch_jacobian',required=False, default = 0,  help="execute mini batch runs?",type=int)
parser.add_argument("-nullspace_constraints", dest='nullspace_constraints',required=False, default = 0,  help="execute nullspace penalzing runs?",type=int)
parser.add_argument("-train_data_size", dest='train_data_size',required=False, default = 1024,  help="training data size",type=int)

args = parser.parse_args()

def default_settings():
	settings = {}

	settings['architecture'] = 'as_dense'
	settings['fixed_input_rank'] = 50
	settings['fixed_output_rank'] = 50
	settings['truncation_dimension'] = 50
	settings['total_epochs'] = 100
	settings['target_rank'] = 50
	settings['batch_rank'] = 50
	settings['l2_weight'] = 1.
	settings['h1_weight'] = 1.
	settings['left_weight'] = 1.
	settings['right_weight'] = 1.
	settings['train_data_size'] = args.train_data_size
	settings['test_data_size'] = 1024
	settings['constraint_sketching'] = 0
	settings['output_sketch_dim'] = 50
	settings['input_sketch_dim'] = 50
	settings['train_full_jacobian'] = 0

	return settings

def build_string(settings):
	command = 'python dino_training.py'
	command+=' '
	for key,value in settings.items():
		command += '-'+key
		command += ' '
		command += str(value)
		command += ' '
	return command

def run_just_fullJ_training(settings):
	initial_name = settings['network_name']
	initial_name += '_train_'+str(settings['train_data_size'])
	settings['left_weight'] = 0.
	settings['right_weight'] = 0.
	inner_timing_dict = {}
	settings['network_name'] = initial_name+'fullJ'
	settings['train_full_jacobian'] = 1
	t0 = time.time()
	os.system(build_string(settings))
	duration = time.time() - t0
	inner_timing_dict[settings['network_name']] = duration
	print('training of '+settings['network_name']+' took ',duration, 's')
	return inner_timing_dict


def run_h1_l2_andj_trains(settings, nullspace_constraints = False):
	assert settings['network_name'] is not None
	inner_timing_dict = {}
	initial_name = settings['network_name']
	initial_name += '_train_'+str(settings['train_data_size'])
	#H1 Run
	settings['network_name'] = initial_name+'h1'
	# Forget the constraints they take too long
	settings['left_weight'] = 0.
	settings['right_weight'] = 0.
	if settings['batch_rank'] < settings['target_rank']:
		settings['left_weight'] = 0.
		settings['right_weight'] = 0.
	t0 = time.time()
	os.system(build_string(settings))
	duration = time.time() - t0
	inner_timing_dict[settings['network_name']] = duration
	print('training of '+settings['network_name']+' took ',duration, 's')
	#L2 Run
	if settings['batch_rank'] == settings['target_rank']:
		settings['network_name'] = initial_name+'l2'
		settings['left_weight'] = 0.
		settings['right_weight'] = 0.
		settings['h1_weight'] = 0.
		t0 = time.time()
		os.system(build_string(settings))
		duration = time.time() - t0
		inner_timing_dict[settings['network_name']] = duration
		print('training of '+settings['network_name']+' took ',duration, 's')

	# Full Jacobian training
	if settings['batch_rank'] == settings['target_rank'] and settings['train_full_jacobian']:
		settings['network_name'] = initial_name+'fullJ'
		settings['train_full_jacobian'] = 1
		t0 = time.time()
		os.system(build_string(settings))
		duration = time.time() - t0
		inner_timing_dict[settings['network_name']] = duration
		print('training of '+settings['network_name']+' took ',duration, 's')


	if settings['batch_rank'] == settings['target_rank'] and nullspace_constraints:
		#H1 Run with constraints
		settings['network_name'] = initial_name+'h1_nullspace_constraints'
		# Forget the constraints they take too long
		settings['left_weight'] = 1.
		settings['right_weight'] = 1.
		t0 = time.time()
		os.system(build_string(settings))
		duration = time.time() - t0
		inner_timing_dict[settings['network_name']] = duration
		print('training of '+settings['network_name']+' took ',duration, 's')


		# Run the sketched constraint here
		settings['network_name'] = initial_name+'h1_constraint_sketch_in'+str(settings['output_sketch_dim'])+'out_'+str(settings['input_sketch_dim'])
		settings['l2_weight'] = 1.
		settings['left_weight'] = 1.
		settings['right_weight'] = 1.
		settings['h1_weight'] = 1.
		settings['constraint_sketching'] = 1
		t0 = time.time()
		os.system(build_string(settings))
		duration = time.time() - t0
		inner_timing_dict[settings['network_name']] = duration
		print('training of '+settings['network_name']+' took ',time.time() - t0, 's')

	return inner_timing_dict

all_timings = {}

################################################################################
# Active subspace networks
# 50-50 network 
if args.full_batch_jacobian:
	# Full batch training
	settings5050 = default_settings()
	settings5050['fixed_input_rank'] = 50
	settings5050['fixed_output_rank'] = 50
	settings5050['architecture'] = 'as_dense'
	settings5050['network_name'] = 'as_dense5050'

	all_timings.update(run_h1_l2_andj_trains(settings5050,nullspace_constraints = args.nullspace_constraints))

if args.full_jacobian_training:
	# full Jacobian training
	settings5050 = default_settings()
	settings5050['fixed_input_rank'] = 50
	settings5050['fixed_output_rank'] = 50
	settings5050['architecture'] = 'as_dense'
	settings5050['network_name'] = 'as_dense5050'

	all_timings.update(run_just_fullJ_training(settings5050))

if args.mini_batch_jacobian:
	# Batch rank 10
	settings5050 = default_settings()
	settings5050['fixed_input_rank'] = 50
	settings5050['fixed_output_rank'] = 50
	settings5050['architecture'] = 'as_dense'
	settings5050['network_name'] = 'as_dense5050_batch10'
	settings5050['batch_rank'] = 10

	all_timings.update(run_h1_l2_andj_trains(settings5050,nullspace_constraints = args.nullspace_constraints))

# 100-50 networks
if args.full_batch_jacobian:
	# Full batch training
	settings10050 = default_settings()
	settings10050['fixed_input_rank'] = 100
	settings10050['fixed_output_rank'] = 50
	settings10050['architecture'] = 'as_dense'
	settings10050['network_name'] = 'as_dense10050'

	all_timings.update(run_h1_l2_andj_trains(settings10050,nullspace_constraints = args.nullspace_constraints))

if args.full_jacobian_training:
	# full Jacobian training
	settings10050 = default_settings()
	settings10050['fixed_input_rank'] = 100
	settings10050['fixed_output_rank'] = 50
	settings10050['architecture'] = 'as_dense'
	settings10050['network_name'] = 'as_dense10050'

	all_timings.update(run_just_fullJ_training(settings10050))

if args.mini_batch_jacobian:
	# Batch rank 10
	settings10050 = default_settings()
	settings10050['fixed_input_rank'] = 100
	settings10050['fixed_output_rank'] = 50
	settings10050['architecture'] = 'as_dense'
	settings10050['network_name'] = 'as_dense10050_batch10'
	settings10050['batch_rank'] = 10

	all_timings.update(run_h1_l2_andj_trains(settings10050,nullspace_constraints = args.nullspace_constraints))

################################################################################
if args.full_batch_jacobian:
	# Full batch
	settingsgeneric = default_settings()
	settingsgeneric['architecture'] = 'generic_dense'
	settingsgeneric['network_name'] = 'generic_dense'
	all_timings.update(run_h1_l2_andj_trains(settingsgeneric,nullspace_constraints = args.nullspace_constraints))

if args.full_jacobian_training:
	# full Jacobian training
	settingsgeneric = default_settings()
	settingsgeneric['architecture'] = 'generic_dense'
	settingsgeneric['network_name'] = 'generic_dense'

	all_timings.update(run_just_fullJ_training(settingsgeneric))

if args.mini_batch_jacobian:
	# Batch rank 10
	settingsgeneric = default_settings()
	settingsgeneric['batch_rank'] = 10
	settingsgeneric['architecture'] = 'generic_dense'
	settingsgeneric['network_name'] = 'generic_dense_batch10'

	all_timings.update(run_h1_l2_andj_trains(settingsgeneric,nullspace_constraints = args.nullspace_constraints))


# Save the timings dictionary:
os.makedirs('training_timings/',exist_ok=True)
timings_filename = 'training_timings/n_data'+str(settingsgeneric['train_data_size'])+str(datetime.date.today())+'.pkl'
with open(timings_filename,'wb+') as f_timings:
	pickle.dump(all_timings,f_timings,pickle.HIGHEST_PROTOCOL)





