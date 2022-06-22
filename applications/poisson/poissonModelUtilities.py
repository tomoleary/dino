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
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_PATH', "../") )
import hippylib as hl

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

np.random.seed(seed=0)

from poissonModelSettings import poisson_problem_settings

def poisson_model_wrapper(settings = poisson_problem_settings()):
	# Set up the mesh, finite element spaces, and PDE forward problem
	ndim = settings['ndim']
	nx = settings['nx']
	ny = settings['ny']
	mesh = dl.UnitSquareMesh(nx, ny)
	Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
		Vh[hl.STATE].dim(), Vh[hl.PARAMETER].dim(), Vh[hl.ADJOINT].dim()) )

	# Now for the PDE formulation


	def u_boundary(x, on_boundary):
		return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

	u_bdr = dl.Expression("x[1]", degree=1)
	u_bdr0 = dl.Constant(0.0)
	bc = dl.DirichletBC(Vh[hl.STATE], u_bdr, u_boundary)
	bc0 = dl.DirichletBC(Vh[hl.STATE], u_bdr0, u_boundary)

	f = dl.Constant(0.0)
		
	def pde_varf(u,m,p):
		return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx
		
	pde = hl.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

	# Set up the prior

	gamma = settings['gamma']
	delta = settings['delta']
		
	theta0 = settings['prior_theta0']
	theta1 = settings['prior_theta1']
	alpha  = settings['prior_alpha']
		
	anis_diff = dl.CompiledExpression(hl.ExpressionModule.AnisTensor2D(), degree = 1)
	anis_diff.set(theta0, theta1, alpha)

	# Set up the prior
	prior = hl.BiLaplacianPrior(Vh[hl.PARAMETER], gamma, delta, anis_diff, robin_bc=True)
	print("Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2))    

	# Set up the observation and misfits
	nx_targets = settings['nx_targets']
	ny_targets = settings['ny_targets']
	ntargets = nx_targets*ny_targets
	assert ntargets == settings['ntargets'], 'Inconsistent target dimensions in settings'

	#Targets only on the bottom
	x_targets = np.linspace(0.1,0.9,nx_targets)
	y_targets = np.linspace(0.1,0.5,ny_targets)
	targets = []
	for xi in x_targets:
		for yi in y_targets:
			targets.append((xi,yi))
	targets = np.array(targets)

	print( "Number of observation points: {0}".format(ntargets) )

	misfit = hl.PointwiseStateObservation(Vh[hl.STATE], targets)

	model = hl.Model(pde, prior, misfit)

	mw_settings = hf.hippylibModelWrapperSettings()
	mw_settings['rel_noise'] = settings['rel_noise']
	mw_settings['seed'] = settings['seed']

	modelwrapper = hf.hippylibModelWrapper(model,settings = mw_settings)
	modelwrapper.targets = targets
	# Setup the inverse problem
	modelwrapper.setUpInverseProblem()
	
	return modelwrapper

