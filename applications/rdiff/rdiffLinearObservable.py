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
import ufl
import math

path_to_hippylib = '../../hippylib/'
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH',path_to_hippylib))
from hippylib import *

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import LinearStateObservable


def rdiff_linear_observable(mesh,nx_targets= 10, ny_targets = 5,output_folder ='rdiff_setup/',\
									 verbose = False,seed = 0):

	########################################################################
	#####Set up the mesh and finite element spaces#########################

	Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	para_dim = Vh[PARAMETER].dim()
	if verbose:
		print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()) )

	########################################################################
	#####Set up the forward problem#########################
	def u_boundary(x, on_boundary):
	    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

	u_bdr = dl.Expression("x[1]", degree=1)
	u_bdr0 = dl.Constant(0.0)
	bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
	bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)

	# f = dl.Constant(0.0)
	c = dl.Constant(1.0)
	# f = dl.interpolate( dl.Expression('max(0.5,exp(-25*(pow(x[0]-0.5,2) +  pow(x[1]-0.5,2))))',degree=5), Vh[STATE])

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

	pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

	########################################################################
	# Construct the linear observable
	# Targets only on the bottom
	#Targets only on the bottom
	x_targets = np.linspace(0.1,0.9,nx_targets)
	y_targets = np.linspace(0.1,0.5,ny_targets)
	targets = []
	for xi in x_targets:
	    for yi in y_targets:
	        targets.append((xi,yi))
	targets = np.array(targets)
	ntargets = targets.shape[0]

	if verbose:
		print( "Number of observation points: {0}".format(ntargets) )

	B = assemblePointwiseObservation(Vh[STATE], targets)

	observable = LinearStateObservable(pde,B)

	return observable
	