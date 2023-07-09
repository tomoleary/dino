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
import numpy as np
import warnings

import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH', "../") )
import hippylib as hp


def inferenceOracleSettings(settings = {}):
	"""
	"""

	# These options are either 'NN' or 'PDE'
	settings['misfit_method'] = 'NN'
	settings['cost_method'] = 'NN'
	settings['gradient_method'] = 'NN'

	settings['use_regularization'] = True

	# Choose from 'regularization', 'mass_matrix', or None (in which case not inf-dim consistent)
	settings['riesz_operator'] = 'regularization'


	return settings


class InferenceOracle:
	"""
	"""

	def __init__(self, hpModelWrapper, neural_operator, settings = inferenceOracleSettings()):
		"""
		"""

		self.modelwrapper = hpModelWrapper

		self.neural_operator = neural_operator
		self.neural_operator_weights = None

		self.settings = settings


		self.observable_shape = tuple(self.neural_operator.outputs[0].get_shape().as_list()[1:])
		self.dQ = int(np.prod(self.observable_shape))
		self.parameter_shape = tuple(self.neural_operator.inputs[0].get_shape().as_list()[1:])
		self.dM = int(np.prod(self.parameter_shape))

		assert neural_operator is not None

		# Define Jtproduct network here.
		input_m = neural_operator.inputs[0]
		# input_dim = 
		input_dim = self.dM
		output_q = neural_operator.outputs[0]
		# output_dim = output_q.shape[-1].value
		output_dim = self.dQ

		# Efficient(?) implementation of Jacobian-transpose vector products
		qhat = tf.keras.layers.Input(shape = (output_dim))
		qqhat = tf.einsum('ij,ij->i',output_q,qhat)
		Jtqhat = tf.gradients(qqhat,input_m,stop_gradients = qhat,name='Jtqhat')[0]
		self.Jtprod = tf.keras.Model([input_m,qhat],Jtqhat)

		# full Jacobian
		self.input_m = input_m
		Jfull_model = tf.keras.Model(input_m,output_q)

		with tf.GradientTape(persistent = True) as tape:
			tape.watch(self.input_m)
			jmodel = Jfull_model(self.input_m)
		# sign issue
		self.fullJ = tape.batch_jacobian(jmodel,self.input_m)

		# Jacobian vector products 
		# (via contraction w/ full Jacobian, fwd mode was difficult to get to work in tf)
		mhat = tf.keras.layers.Input(shape = (input_dim))
		Jmhat = tf.einsum('ikj,ij->ik',self.fullJ,mhat)
		self.Jprod = tf.keras.Model([self.input_m,mhat],Jmhat)

		# Inverse problem data:
		self.d = self.modelwrapper.model.misfit.d.get_local()
		self.m_help = self.modelwrapper.model.generate_vector(hp.PARAMETER)


	def Jtvp(self,mi,qhat):
		"""
		"""
		# The logic here needs to be better thought out to accomodate action on matrices
		if mi.shape[0] != 1 or len(mi.shape) == 1:
			mi = np.expand_dims(mi,axis = 0)
		if qhat.shape[0] != 1 or len(mi.shape) == 1:
			qhat = np.expand_dims(qhat,axis = 0)

		return self.Jtprod.predict([mi,qhat])[0]

	def Jvp(self,mi,mhat,sess = None):
		"""
		"""
		# The logic here needs to be better thought out to accomodate action on matrices
		if mi.shape[0] != 1 or len(mi.shape) == 1:
			mi = np.expand_dims(mi,axis = 0)
		if mhat.shape[0] != 1 or len(mi.shape) == 1:
			mhat = np.expand_dims(mhat,axis = 0)

		return self.Jprod.predict([mi,mhat])[0]

	def J(self,mi):
		"""
		"""
		if mi.shape[0] != 1 or len(mi.shape) == 1:
			mi = np.expand_dims(mi,axis = 0)
		if int(tf.__version__[0]) > 1:
			sess = tf.compat.v1.keras.backend.get_session()
		else:
			sess = tf.keras.backend.get_session()
		if mi.shape[0] == 1:
			return sess.run(self.fullJ,feed_dict={self.input_m:mi})[0]
		else:
			return sess.run(self.fullJ,feed_dict={self.input_m:mi})


	def cost(self,m):
		"""
		"""
		if self.settings['cost_method'] == 'PDE':
			if type(m) is np.ndarray:
				self.m_help.zero()
				self.m_help.set_local(m)
			cost = self.modelwrapper.evalMisfitCost(self.m_help)
		elif self.settings['cost_method'] == 'NN':
			q_NN = self.neural_operator.predict(np.expand_dims(m,axis = 0))[0]
			misfit = q_NN - self.d
			cost = (0.5/self.modelwrapper.model.misfit.noise_variance)*np.linalg.norm(misfit)
		
		return cost

	def misfit(self,m):
		"""
		"""
		if self.settings['misfit_method'] == 'PDE':
			if type(m) is np.ndarray:
				self.m_help.zero()
				self.m_help.set_local(m)
			misfit = self.modelwrapper.evalMisfit(self.m_help).get_local()
		elif self.settings['misfit_method'] == 'NN':
			q_NN = self.neural_operator.predict(np.expand_dims(m,axis = 0))[0]
			misfit = (1./self.modelwrapper.model.misfit.noise_variance)*(q_NN - self.d)
			
		return misfit

	def variational_gradient(self,m,u = None,p = None):
		"""
		Optional parameters u and p can be passed in, in the case that the PDE variational
		gradient is computed, PDE solves can be reused
		"""
		if self.settings['gradient_method'] == 'PDE':
			mg_var = self.modelwrapper.evalVariationalGradient([u,m,p])
		elif self.settings['gradient_method'] == 'NN':
			mg_var = self.modelwrapper.model.generate_vector(hp.PARAMETER)
			misfit  = self.misfit(m)
			mg_var.set_local(self.Jtvp(m.get_local(),misfit))

		if self.settings['use_regularization']:
			mg_reg = self.modelwrapper.evalRegularizationGradient([u,m,p])
			mg_var.axpy(1.0,mg_reg)

		return mg_var

	def gradient(self,m,u = None, p = None):
		"""
		"""
		mg_var = self.variational_gradient(m,u=u,p=p)

		gradient = self.modelwrapper.model.generate_vector(hp.PARAMETER)
		if self.settings['riesz_operator'] == 'regularization':
			self.modelwrapper.invertRegularization(gradient,mg_var)
		elif self.settings['riesz_operator'] == 'mass_matrix':
			try:
				self.modelwrapper.invertMassMatrix(gradient,mg_var)
			except:
				print('Mass matrix solution failed, using regulzation operator instead'.center(80))
				self.modelwrapper.invertRegularization(gradient,mg_var)
		else:
			gradient = mg_var

		return gradient.get_local()

