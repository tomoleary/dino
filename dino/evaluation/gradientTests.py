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

import dolfin as dl
import numpy as np
import hippylib as hp
import time


def gradient_error_test(oracle_dict,n_samples = 100):
	"""
	"""
	surrogate_names = list(oracle_dict.keys())

	required_settings = {'use_regularization': False,'gradient_method':'NN',\
							'misfit_method': 'NN'}
	reset_settings = {}

	for surrogate in surrogate_names:
		reset_settings[surrogate] = {}
		for required_key,required_target in required_settings.items():
			reset_settings[required_key] = oracle_dict[surrogate].settings[required_key]
			oracle_dict[surrogate].settings[required_key] = required_target

	modelwrapper = oracle_dict[surrogate_names[0]].modelwrapper


	# Log the gradient as well as the prior preconditioned Riesz mapped gradient
	g_rel_errors = {}
	g_total_errors = {}

	mg_rel_errors = {}
	mg_total_errors = {}

	for surrogate in surrogate_names:
		g_rel_errors[surrogate] = np.zeros(n_samples)
		g_total_errors[surrogate] = np.zeros(n_samples)

		mg_rel_errors[surrogate] = np.zeros(n_samples)
		mg_total_errors[surrogate] = np.zeros(n_samples)

	# Allocation of vectors
	true_g = modelwrapper.model.generate_vector(hp.PARAMETER)
	surrogate_g = modelwrapper.model.generate_vector(hp.PARAMETER)

	for i in range(n_samples):
		t0 = time.time()
		# Sample point to evaluate gradient at
		m = modelwrapper.samplePrior()
		true_mg = modelwrapper.evalVariationalGradient([None,m,None])
		true_mg_norm = np.linalg.norm(true_mg.get_local())
		# true_reg = modelwrapper.evalRegularizationGradient([None,m,None])
		true_g.zero()
		modelwrapper.invertRegularization(true_g,true_mg)
		true_g_norm = np.linalg.norm(true_g.get_local())

		for surrogate in surrogate_names:
			surrogate_mg = oracle_dict[surrogate].variational_gradient(m)
			error_mg_i = true_mg.get_local() - surrogate_mg.get_local()
			mg_total_error_i = np.linalg.norm(error_mg_i)
			mg_rel_error_i = mg_total_error_i/true_mg_norm

			mg_total_errors[surrogate][i] = mg_total_error_i
			mg_rel_errors[surrogate][i] = mg_rel_error_i

			surrogate_g.zero()
			modelwrapper.invertRegularization(surrogate_g,surrogate_mg)
			error_g_i = true_g.get_local() - surrogate_g.get_local()
			g_total_error_i = np.linalg.norm(error_g_i)
			g_rel_error_i = g_total_error_i/true_g_norm

			g_total_errors[surrogate][i] = g_total_error_i
			g_rel_errors[surrogate][i] = g_rel_error_i

			# Free up memory that is continually allocated? Fix this?
			del(surrogate_mg)
		print('Sample ',i,'took ',time.time() - t0,'s')


	for surrogate in surrogate_names:
		reset_settings[surrogate] = {}
		for reset_key,reset_target in reset_settings.items():
			oracle_dict[surrogate].settings[reset_key] = reset_target

	return {'g_rel_errors':g_rel_errors, 'g_total_errors':g_total_errors,\
			'mg_rel_errors':mg_rel_errors, 'mg_total_errors':mg_total_errors}


def gradient_timing_test(oracle_dict,n_samples = 100):
	"""
	"""
	surrogate_names = list(oracle_dict.keys())

	required_settings = {'use_regularization': False,'gradient_method':'NN',\
							'misfit_method': 'NN'}
	reset_settings = {}

	for surrogate in surrogate_names:
		reset_settings[surrogate] = {}
		for required_key,required_target in required_settings.items():
			reset_settings[required_key] = oracle_dict[surrogate].settings[required_key]
			oracle_dict[surrogate].settings[required_key] = required_target

	modelwrapper = oracle_dict[surrogate_names[0]].modelwrapper






