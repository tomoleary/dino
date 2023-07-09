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


sys.path.append( os.environ.get('DINO_PATH'))
from dino import *

# sys.path.append('../../../dino/')
# from surrogate_construction import *

# Import hyperelasticity problem specifics
sys.path.append('../')
from hyperelasticityModelSettings import hyperelasticity_problem_settings

try:
	tf.random.set_seed(0)
except:
	tf.set_random_seed(0)

from argparse import ArgumentParser

# Arguments to be parsed from the command line execution
parser = ArgumentParser(add_help=True)
# Architectural parameters
parser.add_argument("-architecture", dest='architecture',required=False, default = 'rb_dense', help="architecture type: as_dense or generic_dense",type=str)
parser.add_argument("-input_basis", dest='input_basis',required=False, default = 'as',  help="input basis: as or kle",type=str)
parser.add_argument("-output_basis", dest='output_basis',required=False, default = 'jjt',  help="output basis: pod or jjt",type=str)
parser.add_argument("-fixed_input_rank", dest='fixed_input_rank',required=False, default = 200, help="rank for input of AS network",type=int)
parser.add_argument("-fixed_output_rank", dest='fixed_output_rank',required=False, default = 50, help="rank for output of AS network",type=int)
parser.add_argument("-truncation_dimension", dest='truncation_dimension',required=False, default = 200, help="truncation dimension for low rank networks",type=int)
parser.add_argument("-network_name", dest='network_name',required=True,  help="out name for the saved weights",type=str)

# Optimization parameters
parser.add_argument("-total_epochs", dest='total_epochs',required=False, default = 1,  help="total epochs for training",type=int)

# Loss function parameters
parser.add_argument("-target_rank", dest='target_rank',required=False, default = 50,  help="target rank to be learned for Jacobian information",type=int)
parser.add_argument("-batch_rank", dest='batch_rank',required=False, default = 50,  help="batch rank parameter used in sketching of Jacobian information",type=int)
parser.add_argument("-l2_weight", dest='l2_weight',required=False, default = 1.,  help="weight for l2 term",type=float)
parser.add_argument("-h1_weight", dest='h1_weight',required=False, default = 10.,  help="weight for h1 term",type=float)

# Full J training
parser.add_argument("-train_full_jacobian", dest='train_full_jacobian',required=False, default = 1,  help="full J",type=int)


parser.add_argument("-train_data_size", dest='train_data_size',required=False, default = 7*1024,  help="training data size",type=int)
parser.add_argument("-test_data_size", dest='test_data_size',required=False, default = 1024,  help="testing data size",type=int)

args = parser.parse_args()

# jacobian_network = None
problem_settings = hyperelasticity_problem_settings()


settings = jacobian_network_settings(problem_settings)

n_obs = 50
correlation_length = 0.3
nx = 64
settings['data_dir'] = '../data/hyperelasticity_nobs_'+str(n_obs)+'_correlation_length'+str(correlation_length)+'_nx'+str(nx)+'/'

settings['target_rank'] = args.target_rank
settings['batch_rank'] = args.batch_rank

settings['train_data_size'] = args.train_data_size
settings['test_data_size'] = args.test_data_size

settings['architecture'] = args.architecture
settings['depth'] = 6

settings['fixed_input_rank'] = args.fixed_input_rank
settings['fixed_output_rank'] = args.fixed_output_rank
settings['truncation_dimension'] = args.truncation_dimension
settings['input_basis'] = args.input_basis
settings['output_basis'] = args.output_basis

settings['train_full_jacobian'] = args.train_full_jacobian
settings['opt_parameters']['train_full_jacobian'] = args.train_full_jacobian

settings['reduced_input_training'] = True
settings['reduced_output_training'] = False


if (settings['batch_rank'] == settings['target_rank']):
	settings['outer_epochs'] = 1
	settings['opt_parameters']['keras_epochs'] = args.total_epochs
else:
	settings['shuffle_every_epoch'] = True
	settings['outer_epochs'] = args.total_epochs
	settings['opt_parameters']['keras_epochs'] = 1
settings['opt_parameters']['keras_verbose'] = True

settings['opt_parameters']['loss_weights'] = [args.l2_weight,args.h1_weight]

settings['network_name'] = args.network_name

if args.l2_weight != 1.0:
	settings['network_name'] += 'l2_weight_'+str(args.l2_weight)


jacobian_network = jacobian_training_driver(settings)



