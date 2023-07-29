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
import dolfin as dl
import ufl
import numpy as np
import tensorflow as tf
import time


# Keeping these generic as operators to accomodate
# more general matrix structures than scalar multiple of identity later on 
class noiseCovariance:
	"""
	Generic noise covariance class (assuming scalar multiple of identity)
	"""
	
	def __init__(self, noise_variance):

		self.noise_variance = noise_variance


	def __mult__(self,x):
		return self.noise_variance*x


class noisePrecision:
	"""
	Inverse of covariance (based on scalar multiple of identity)
	"""
	
	def __init__(self, noise_variance):

		self.noise_variance = noise_variance


	def __mult__(self,x):
		return x*(1/self.noise_variance)

class JTJfromData:
	"""
	Computation of active subspace efficient from Jacobian training data samples.
	"""
	def __init__(self, J, noise_precision = None, init_vector_lambda = None):
		"""
		"""

		self._J = J

		self.dQ, self.dM = self.J.shape

		self._noise_precision = noise_precision

		self.init_vector_lambda = init_vector_lambda # For GEVP 

	@property
	def J(self):
		return self._J

	@property
	def noise_precision(self):
		return self._noise_precision

	def set_J(self,J):
		self._J = J
	
	def init_vector(self,x,dim):
		"""
		Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`A`.
		"""
		assert init_vector_lambda is not None
		self.init_vector_lambda(x,dim)

	def mult(self,x,y):
		"""
		Compute :math:`y = mean(JTJ)x `
		"""
		"""
		"""
		x_np = x.get_local()
		assert x_np.shape[0] == self.dM
		Jx_np = np.einsum('jk,k->j',self.J,x_np)

		if self.noise_precision is not None:
			Jx_np = self.noise_precision@Jx_np

		# print('PhiTJX_np.shape = ',JX_np.shape)
		JTJx_np = np.einsum('jk,j->k',self.J,Jx_np)
		# print('JTPhiPhiTJX_np.shape = ',JTJX_np.shape)
		y.set_local(JTJx_np)

	def transpmult(self,x,y):
		"""
		Compute :math:`y = mean(JTJ)x `
		"""
		return self.mult(x,y)




class JTGamma_invJ:
	"""
	This class implements the operator :math:`J^TJ` given a Jacobian :math:`J`
	"""
	def __init__(self,J,noise_variance):
		"""
		Constructor
			- :code:`J` - Jacobian object, assumed to be of of type :code:`hippyflow.modeling.Jacobian`
			- :code:`noise_variance` - float, assuming scalar multiple of identity noise covariance
		"""
		self.J = J
		self.vector_help = dl.Vector(self.J.mpi_comm())
		self.J.init_vector(self.vector_help,0)
		self.noise_variance = noise_variance

	def mult(self,x,y):
		"""
		Compute :math:`y = J^TJ x `
		"""
		self.J.mult(x,self.vector_help)
		# The Jacobian implements the B and BT
		# So we need to manually handle the other aspect of
		# applyWuu
		# e.g., the scaling by noise variance
		# https://github.com/hippylib/hippylib/blob/master/hippylib/modeling/misfit.py#LL121C27-L121C27
		self.vector_help *= (1./self.noise_variance)
		self.J.transpmult(self.vector_help,y)

	def init_vector(self,x,dim=None):
		"""
		Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`JTJ`.
		"""
		self.J.init_vector(x,1)



