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

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
import numpy as np
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    
import time
import pickle

sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib

sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow


# Import dino inference module
sys.path.append(os.environ.get('DINO_PATH'))
from dino import *

from dino.evaluation.surrogatePostProcessing import evaluateJacobianNetwork

# Import reaction diffusion problem specifics
sys.path.append('../../')
from rdiffModelSettings import rdiff_problem_settings

try:
	tf.random.set_seed(0)
except:
	tf.set_random_seed(0)

from argparse import ArgumentParser
# Arguments to be parsed from the command line execution
parser = ArgumentParser(add_help=True)
# Weights directory
parser.add_argument("-weights_dir", dest='weights_dir', required=True, help="Weights directory",type=str)
# parser.add_argument("-ndata", dest='ndata', required=True, help="ndata",type=str)
parser.add_argument("-input_dim", dest = 'input_dim',required=False,default = 4225, help = "input dim",type = int)
parser.add_argument("-logging_dir", dest = 'logging_dir',required=False,default = 'postproc/accuracies/', help = "input dim",type = str)
args = parser.parse_args()

problem_settings = rdiff_problem_settings()

weights_dir = args.weights_dir+'/'


weights_files = os.listdir(weights_dir)

n_obs = 50
gamma = 0.1
delta = 1.0
nx = 64
data_dir = '../../data/rdiff_nobs_'+str(n_obs)+'_g'+str(gamma)+'_d'+str(delta)+'_nx'+str(nx)+'/'

print(os.path.isdir(data_dir))


for weights_name in weights_files[:]:
	print('weights_name = ',weights_name)
	t0 = time.time()
	####
	evaluate_network = False
	settings = jacobian_network_settings(problem_settings)
	settings['nullspace_constraints'] = False
	settings['opt_parameters']['loss_weights'] = [1.0,1.0]
	settings['depth'] = 6
	settings['fixed_input_rank'] = 50
	settings['full_jacobian'] = True
	settings['full_JVVT'] = False
	####

	if ('as_dense' in weights_name.lower()) or ('dipnet' in weights_name.lower()):
		settings['architecture'] = 'rb_dense'
		if ('10050' in weights_name) or ('100-50' in weights_name):
			print('100')
			settings['fixed_input_rank'] = 100

		evaluate_network = True

	elif 'generic_dense' in weights_name:
		settings['architecture'] = 'generic_dense'
		# What is a better way in general to set the input and output dimensions.
		settings['input_dim'] = args.input_dim
		settings['output_dim'] = 50
		evaluate_network = True
	else:
		print('Not implemented, passing for now')
		pass

	if evaluate_network:
		file_name = weights_dir+weights_name
		jacobian_network = observable_network_loader(settings, file_name = file_name)	
		for i in range(2):
			print(80*'#')
		print('Running for :'.center(80))
		print(weights_name.center(80))
		for i in range(2):
			print(80*'#')
		results = evaluateJacobianNetwork(settings,jacobian_network = jacobian_network,data_dir = data_dir)
		logging_dir = args.logging_dir
		logger_name = weights_name.split(weights_dir)[-1].split('.pkl')[0]+'_accuracies.pkl'

		os.makedirs(logging_dir,exist_ok = True)
		import pickle

		with open(logging_dir+logger_name, 'wb+') as f:
		    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

	print(' Time = ',time.time() - t0,'s')





