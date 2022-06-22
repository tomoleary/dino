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

path_to_hippylib = '../../hippylib/'
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH',path_to_hippylib))
from hippylib import *

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import LinearStateObservable


def poisson_linear_observable(mesh,nx_targets= 10, ny_targets = 5,output_folder ='poisson_setup/',\
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

	f = dl.Constant(0.0)

	def pde_varf(u,m,p):
	    return dl.exp(m)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx

	pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

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
	