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
import hippylib as hp

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

np.random.seed(seed=0)

from rdiffModelSettings import *


def rdiff_model_wrapper(settings = rdiff_problem_settings()):
########################################################################
	#####Set up the mesh and finite element spaces#########################
	ndim = settings['ndim']
	nx = settings['nx']
	ny = settings['ny']
	mesh = dl.UnitSquareMesh(nx, ny)
	Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	para_dim = Vh[hp.PARAMETER].dim()

	print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()) )

	########################################################################
	#####Set up the forward problem#########################
	def u_boundary(x, on_boundary):
		return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

	u_bdr = dl.Expression("x[1]", degree=1)
	# u_bdr = dl.Constant(0.0)
	u_bdr0 = dl.Constant(0.0)
	bc = dl.DirichletBC(Vh[hp.STATE], u_bdr, u_boundary)
	bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)

	# f = dl.Constant(0.0)
	c = dl.Constant(1.0)
	# f = dl.interpolate( dl.Expression('max(0.5,exp(-25*(pow(x[0]-0.5,2) +  pow(x[1]-0.5,2))))',degree=5), Vh[hp.STATE])

	# def pde_varf(u,m,p):
	#     return dl.inner(dl.exp(m)*dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx +c*u*u*u*p*dl.dx- f*p*dl.dx


	# 3. Set up rhs
	N_WELLS_PER_SIDE = 5
	LOC_LOWER = 0.25
	LOC_UPPER = 0.75
	WELL_WIDTH = 0.1
	STRENGTH_UPPER = 10.0
	STRENGTH_LOWER = -10.0

	well_grid = np.linspace(LOC_LOWER, LOC_UPPER, N_WELLS_PER_SIDE)
	well_grid_x, well_grid_y = np.meshgrid(well_grid, well_grid)
	mollifier_list = [] 

	for i in range(N_WELLS_PER_SIDE):
		for j in range(N_WELLS_PER_SIDE):
			mollifier_list.append(
					dl.Expression("a*exp(-(pow(x[0]-xi,2)+pow(x[1]-yj,2))/(pow(b,2)))", 
						xi=well_grid[i], 
						yj=well_grid[j], 
						a=1/(2*math.pi*WELL_WIDTH**2),
						b=WELL_WIDTH,
						degree=2)
					)

	mollifiers = dl.as_vector(mollifier_list)

	n_wells = len(mollifier_list)
	Vh_RHS = dl.VectorFunctionSpace(mesh, "R", degree=0, dim=n_wells)

	rhs_fun = dl.Function(Vh_RHS)

	# rhs_vec = sample_well_strength(n_wells, STRENGTH_LOWER, STRENGTH_UPPER)
	strength  = 0.25
	rhs_vec = strength*np.ones(n_wells)
	rhs_vec[::2] = -strength

	rhs_fun.vector().set_local(rhs_vec)


	def pde_varf(u,m,p):
		return dl.inner(dl.exp(m)*dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx +c*u*u*u*p*dl.dx- dl.inner(mollifiers, rhs_fun)*p*dl.dx

	pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

	# Set up the prior

	gamma = settings['gamma']
	delta = settings['delta']
		
	theta0 = settings['prior_theta0']
	theta1 = settings['prior_theta1']
	alpha  = settings['prior_alpha']
		
	anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
	anis_diff.set(theta0, theta1, alpha)

	# Set up the prior
	m_mean_fun = dl.Function(Vh[hp.PARAMETER])
	m_mean_fun.interpolate(dl.Constant(-1.0))
	prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff,mean = m_mean_fun.vector(), robin_bc=True)
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

	misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)

	model = hp.Model(pde, prior, misfit)

	mw_settings = hf.hippylibModelWrapperSettings()
	mw_settings['rel_noise'] = settings['rel_noise']
	mw_settings['seed'] = settings['seed']

	modelwrapper = hf.hippylibModelWrapper(model,settings = mw_settings)
	modelwrapper.targets = targets
	# Setup the inverse problem
	modelwrapper.setUpInverseProblem()
	
	return modelwrapper

