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
import dolfin as dl
import numpy as np
import time
import hippylib as hp
import hippyflow as hf

def reduce_dict(master_logger,reduction = 'mean'):
	"""
	Helper function for dictionary reduction
	"""
	reduced_logger = {}
	for file in master_logger.keys():
		reduced_logger[file] = {}
		for key in master_logger[file].keys():
			if reduction == 'mean':
				reduced_logger[file][key] = np.mean(master_logger[file][key])
	return reduced_logger

def compute_jacobian_errors(modelwrapper, oracle_dictionary, n_samples = 100, rank = 50,p = 5):
	"""
	Computation of Jacobian and GN-Hessian errors.
	"""

	master_logger = {}
	for file in oracle_dictionary.keys():
		master_logger[file] = {}
		master_logger[file]['Jerror'] = []
		master_logger[file]['JTJerror'] = []
		master_logger[file]['JJTerror'] = []
		master_logger[file]['rJTJerror'] = []
		master_logger[file]['rJJTerror'] = []
		master_logger[file]['reducedJerror'] = []
		master_logger[file]['nullspace_error'] = []
		master_logger[file]['JTJtraceerror'] = []
		master_logger[file]['JTJlogdeterror'] = []
		master_logger[file]['geigerror'] = []
		master_logger[file]['gtraceerror'] = []
		master_logger[file]['gevperror'] = []
		master_logger[file]['reducedgevperror'] = []


	reducedH = hp.ReducedHessian(modelwrapper.model,misfit_only = True)
	reducedH.gauss_newton_approx = True

	jhelp_dir = 'jacobian_help/'
	os.makedirs(jhelp_dir,exist_ok=True)

	for data_choice in range(n_samples):
		t0 = time.time()
		# Attempt to load everything from jacobian_help/
		try:
			m_help_np = np.load(jhelp_dir+'m_help_np_'+str(data_choice))
			J_full = np.load(jhelp_dir+'J_full_'+str(data_choice))
			lmbda_true = np.load(jhelp_dir+'lmbda_true_'+str(data_choice))
			V_true = np.load(jhelp_dir+'V_true_'+str(data_choice))

		except:
			# Else compute it
			m_help = modelwrapper.samplePrior()
			m_help_np = m_help.get_local()
			np.save(jhelp_dir+'m_help_np_'+str(data_choice),m_help_np)
			# Full true Jacobian
			J_full = modelwrapper.evalJacobian([None,m_help,None])
			np.save(jhelp_dir+'J_full_'+str(data_choice),J_full)
			if (modelwrapper.model.misfit.noise_variance is not None) and False:
				# Set up GEVP
				Omega = hp.MultiVector(m_help,rank + p)
				hp.parRandom.normal(1.,Omega)


				GN_true = hf.npToDolfinOperator(\
												J_full.T@J_full/modelwrapper.model.misfit.noise_variance)
				lmbda_true, V_true = hp.doublePassG(GN_true,modelwrapper.model.prior.R,\
													modelwrapper.model.prior.Rsolver,Omega,rank)

				V_true = hf.mv_to_dense(V_true)
				np.save(jhelp_dir+'lmbda_true_'+str(data_choice),lmbda_true)
				np.save(jhelp_dir+'V_true_'+str(data_choice),V_true)
			print('Full Jacobian compute took',time.time() - t0,'s')

		for file,oracle in oracle_dictionary.items():
			t1 = time.time()
			J_full_pred = oracle.J(m_help_np)
			print('Tensorflow prediction took ',time.time() - t1,'s')
			# J error
			J_error = np.linalg.norm(J_full_pred - J_full)/np.linalg.norm(J_full)
			master_logger[file]['Jerror'].append(J_error)
			# JTJ error
			GN_error = np.linalg.norm(J_full_pred.T@J_full_pred - J_full.T@J_full)/np.linalg.norm(J_full.T@J_full)
			master_logger[file]['JTJerror'].append(GN_error)
			# JJT error
			NG_error = np.linalg.norm(J_full_pred@J_full_pred.T - J_full@J_full.T)/np.linalg.norm(J_full@J_full.T)
			master_logger[file]['JJTerror'].append(NG_error)
			# Reduced SVD error
			U,sigma, VT = np.linalg.svd(J_full,full_matrices = False)
			reduced_J_error = np.linalg.norm(U.T@J_full_pred@VT.T - U.T@J_full@VT.T)/np.linalg.norm(U.T@J_full@VT.T)
			master_logger[file]['reducedJerror'].append(reduced_J_error)
			# Nullspace Error
			nullspace_error = np.linalg.norm(J_full_pred - J_full_pred@VT.T@VT)/np.linalg.norm(J_full)
			master_logger[file]['nullspace_error'].append(nullspace_error)

			# Reduced JTJ error
			rGN_error = np.linalg.norm(VT@J_full_pred.T@J_full_pred@VT.T - VT@J_full.T@J_full@VT.T)/np.linalg.norm(VT@J_full.T@J_full@VT.T)
			master_logger[file]['rJTJerror'].append(rGN_error)

			# Reduced JJT error
			rNG_error = np.linalg.norm(U.T@J_full_pred@J_full_pred.T@U - U.T@J_full@J_full.T@U)/np.linalg.norm(U.T@J_full@J_full.T@U)
			master_logger[file]['rJJTerror'].append(rNG_error)
			
		print('Running sample ',data_choice, ' took ',time.time() - t0, 's')
	return master_logger

