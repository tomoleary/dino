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
import sys, os
import time
sys.path.append( os.environ.get('HIPPYLIB_PATH', "../") )
import hippylib as hp

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

np.random.seed(seed=0)

from hyperelasticityModelSettings import hyperelasticity_problem_settings


def HyperelasticityPrior(Vh_PARAMETER, correlation_length, mean=None, anis_diff=None):
    var = correlation_length / 0.16
    # Delta and gamma
    delta = (var * correlation_length) ** (-0.5)
    gamma = delta * correlation_length ** 2
    if anis_diff is None:
        theta0 = 1.
        theta1 = 1.
        alpha = math.pi / 4.
        anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
        anis_diff.set(theta0, theta1, alpha)
    if mean is None:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, robin_bc=True)
    else:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, mean=mean, robin_bc=True)


class hyperelasticity_varf:

	def __init__(self,Vh,my_ds,traction,body_force=None,mean = None):
		self.Vh = Vh
		self.my_ds = my_ds
		self.traction = traction
		if body_force is None:
			self.body_force = dl.Constant((0.0,0.0))
		else:
			self.body_force = body_force

		if mean is None:
			self._mean_function = dl.Constant(0.0)
		else:
			self.mean_function = dl.Constant(mean)

	def __call__(self,u,m,p):
		d = u.geometric_dimension()
		Id = dl.Identity(d)
		F = Id + dl.grad(u)
		C = F.T*F

		# Lame parameters
		E = dl.exp(m) + dl.Constant(1.0)
		nu = 0.4
		mu = E/(2.0*(1.0 + nu))
		lmbda = (E*nu)/((1.0+nu)*(1.0 - 2.0*nu))

		# Invariants of the deformation tensors
		Ic, J = dl.tr(C), dl.det(F)

		# Stored strain energy density
		psi = (mu/2.0)*(Ic - 3.0) - mu*dl.ln(J) + (lmbda/2.0)*(dl.ln(J))**2

		# Total potential energy:
		Pi = psi*dl.dx + dl.dot(self.body_force,u)*dl.dx + dl.dot(self.traction,u)*self.my_ds(1)

		return dl.derivative(Pi,u,p)


sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import LinearStateObservable, StateSpaceIdentityOperator



def hyperelasticity_state_observable(mesh,output_folder ='hyperelasticity_setup/',verbose = False,seed = 0):

	########################################################################
	#####Set up the mesh and finite element spaces#########################

	Vh2 = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
		Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()) )

	def left_boundary(x, on_boundary):
		return on_boundary and ( x[0] < dl.DOLFIN_EPS)

	def right_boundary(x, on_boundary):
		return on_boundary and (x[0] > 1.0 - dl.DOLFIN_EPS)

	u_left = dl.Constant((0.0,0.0))

	# u_bdr = dl.Expression("x[1]", degree=1)
	u_bdr0 = dl.Constant((0.0,0.0))
	bc = dl.DirichletBC(Vh[hp.STATE], u_left, left_boundary)
	bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, left_boundary)

	# Traction boundary conditions

	boundary_subdomains = dl.MeshFunction("size_t",mesh,mesh.topology().dim()-1)
	boundary_subdomains.set_all(0)

	dl.AutoSubDomain(right_boundary).mark(boundary_subdomains,1)

	my_ds = dl.ds(subdomain_data = boundary_subdomains)

	right_traction_expr = dl.Expression(("a*exp(-1.0*pow(x[1] - 0.5,2)/b)", "c*(1.0 + (x[1]/d))"), a=0.06, b=4, c=0.03, d=10,degree=5)
	right_t = dl.interpolate(right_traction_expr,Vh[hp.STATE])

	pde_varf = hyperelasticity_varf(Vh,my_ds,right_t)

	pde = CustomNonlinearPDEProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

	u_trial = dl.TrialFunction(Vh[hp.STATE])
	u_test = dl.TestFunction(Vh[hp.STATE])

	M = dl.PETScMatrix()
	dl.assemble(dl.inner(u_trial,u_test)*dl.dx, tensor=M)

	B = StateSpaceIdentityOperator(M)
	# B = StateSpaceIdentityOperator()

	observable = LinearStateObservable(pde,B)

	return observable


def hyperelasticity_pointwise_observable(mesh,output_folder ='hyperelasticity_setup/',verbose = False,seed = 0,\
																				nx_targets = 10,ny_targets = 10):

	########################################################################
	#####Set up the mesh and finite element spaces#########################

	Vh2 = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
		Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()) )

	def left_boundary(x, on_boundary):
		return on_boundary and ( x[0] < dl.DOLFIN_EPS)

	def right_boundary(x, on_boundary):
		return on_boundary and (x[0] > 1.0 - dl.DOLFIN_EPS)

	u_left = dl.Constant((0.0,0.0))

	# u_bdr = dl.Expression("x[1]", degree=1)
	u_bdr0 = dl.Constant((0.0,0.0))
	bc = dl.DirichletBC(Vh[hp.STATE], u_left, left_boundary)
	bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, left_boundary)

	# Traction boundary conditions

	boundary_subdomains = dl.MeshFunction("size_t",mesh,mesh.topology().dim()-1)
	boundary_subdomains.set_all(0)

	dl.AutoSubDomain(right_boundary).mark(boundary_subdomains,1)

	my_ds = dl.ds(subdomain_data = boundary_subdomains)

	right_traction_expr = dl.Expression(("a*exp(-1.0*pow(x[1] - 0.5,2)/b)", "c*(1.0 + (x[1]/d))"), a=0.06, b=4, c=0.03, d=10,degree=5)
	right_t = dl.interpolate(right_traction_expr,Vh[hp.STATE])

	pde_varf = hyperelasticity_varf(Vh,my_ds,right_t)

	pde = CustomNonlinearPDEProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

	x_targets = np.linspace(0.1,0.9,nx_targets)
	y_targets = np.linspace(0.1,0.9,ny_targets)
	targets = []
	for xi in x_targets:
	    for yi in y_targets:
	        targets.append((xi,yi))
	targets = np.array(targets)
	ntargets = targets.shape[0]

	if verbose:
		print( "Number of observation points: {0}".format(ntargets) )

	B = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)

	# u_trial = dl.TrialFunction(Vh[hp.STATE])
	# u_test = dl.TestFunction(Vh[hp.STATE])

	# M = dl.PETScMatrix()
	# dl.assemble(dl.inner(u_trial,u_test)*dl.dx, tensor=M)

	# B = StateSpaceIdentityOperator(M)
	# B = StateSpaceIdentityOperator()

	observable = LinearStateObservable(pde,B)

	return observable

class CustomNonlinearPDEProblem(hp.PDEVariationalProblem):

	def solveFwd(self,state,x):
		u = hp.vector2Function(x[hp.STATE],self.Vh[hp.STATE])
		m = hp.vector2Function(x[hp.PARAMETER],self.Vh[hp.PARAMETER])
		p = dl.TestFunction(self.Vh[hp.ADJOINT])

		F = self.varf_handler(u,m,p)
		du = dl.TrialFunction(self.Vh[hp.STATE])
		JF = dl.derivative(F, u, du)

		if True:
			# Assert that we are in a serial instance
			assert self.Vh[hp.STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')
			# Parameters for the problem
			tolerance = 1e-4
			max_iterations = 200
			# Line search
			line_search = True
			max_backtrack = 10
			alpha0 = 1.0
			# Printing
			verbose = False
			

			# Initialize while loop variables
			converged = False 
			iteration = 0
			delta_u = self.generate_state()
			delta_u.zero()
			while not converged:
				iteration += 1
				# from scipy.sparse import csc_matrix, csr_matrix
				# from scipy.sparse import linalg as spla
				import scipy.sparse as sp
				import scipy.sparse.linalg as spla
				# Assemble A and enforce BCS
				A_dl,b_dl = dl.assemble_system(JF,F,bcs =self.bc0) # bc0 is the state variable bc
				residual_norm = dl.norm(b_dl)
				if verbose:
					print('At iteration ',iteration,'the residual norm = ',residual_norm)
				converged = (residual_norm < tolerance)

				A_mat = dl.as_backend_type(A_dl).mat()
				row,col,val = A_mat.getValuesCSR() # I think they only give the csr, so we convert
				A_csc = sp.csr_matrix((val,col,row)).tocsc()
				t0 = time.time()
				A_lu = spla.splu(A_csc)
				if verbose:
					print('Sparse LU build took ',time.time() - t0,'s for iteration',iteration)
				t1 = time.time()
				delta_u.set_local(A_lu.solve(-b_dl.get_local()))
				if verbose:
					print('Sparse LU solve took',time.time() - t1,'s')

				
				if line_search:
					alpha = alpha0
					
					backtrack_iteration = 0
					searching = True
					while searching:
						backtrack_iteration += 1
						u.vector().axpy(alpha,delta_u)
						res_new = dl.assemble(F)
						for bc in self.bc0:
							bc.apply(res_new)
						# This is not sufficient descent, just descent, possibly problematic
						if dl.norm(res_new) < residual_norm:
							searching = False
						else:
							if verbose:
								print('Need to take a smaller step')
							u.vector().axpy(-alpha,delta_u)
							alpha *=0.5
						if backtrack_iteration > max_backtrack:
							break

				else:
					u.vector().axpy(1.,delta_u)

				if iteration > max_iterations:
					print('Maximum iterations for nonlinear PDE solve reached, moving on.')
					print('Final residual norm = ',residual_norm)
					break

			# Meets the interface condition for `solveFwd`
			state.zero()
			state.axpy(1.,u.vector())


		else:	
			F = self.varf_handler(u,m,p)
			du = dl.TrialFunction(self.Vh[hp.STATE])
			JF = dl.derivative(F, u, du)
			problem = dl.NonlinearVariationalProblem(F,u,self.bc,JF)
			solver = dl.NonlinearVariationalSolver(problem)

			prm = solver.parameters
			# print('newton solver parameters = ',solver.parameters['newton_solver'].keys())
			if False:
				prm['newton_solver']['absolute_tolerance'] = 1E-4
				prm['newton_solver']['report'] = True
				prm['newton_solver']['relative_tolerance'] = 1E-3
				prm['newton_solver']['maximum_iterations'] = 200
				prm['newton_solver']['relaxation_parameter'] = 1.0
				# print(dl.info(solver.parameters, True))
			if True:
				prm['nonlinear_solver']='snes' 
				prm['snes_solver']['line_search'] = 'basic'
				prm['snes_solver']['linear_solver']= 'lu'
				prm['snes_solver']['report'] = False
				prm['snes_solver']['error_on_nonconvergence'] = True
				prm['snes_solver']['absolute_tolerance'] = 1E-5
				prm['snes_solver']['relative_tolerance'] = 1E-5
				prm['snes_solver']['maximum_iterations'] = 50
				prm['newton_solver']['absolute_tolerance'] = 1E-3
				prm['newton_solver']['relative_tolerance'] = 1E-2
				prm['newton_solver']['maximum_iterations'] = 200
				prm['newton_solver']['relaxation_parameter'] = 1.0

				# print(dl.info(solver.parameters, True))
			iterations, converged = solver.solve()

			state.zero()
			state.axpy(1.,u.vector())

def hyperelasticity_model(settings = hyperelasticity_problem_settings()):
	# Set up the mesh, finite element spaces, and PDE forward problem
	ndim = settings['ndim']
	nx = settings['nx']
	ny = settings['ny']
	mesh = dl.UnitSquareMesh(nx, ny)
	Vh2 = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
		Vh[hp.STATE].dim(), Vh[hp.PARAMETER].dim(), Vh[hp.ADJOINT].dim()) )

	# Now for the PDE formulation

	# Dirichlet boundary conditions

	def left_boundary(x, on_boundary):
		return on_boundary and ( x[0] < dl.DOLFIN_EPS)

	def right_boundary(x, on_boundary):
		return on_boundary and (x[0] > 1.0 - dl.DOLFIN_EPS)

	u_left = dl.Constant((0.0,0.0))

	# u_bdr = dl.Expression("x[1]", degree=1)
	u_bdr0 = dl.Constant((0.0,0.0))
	bc = dl.DirichletBC(Vh[hp.STATE], u_left, left_boundary)
	bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, left_boundary)

	# Traction boundary conditions

	boundary_subdomains = dl.MeshFunction("size_t",mesh,mesh.topology().dim()-1)
	boundary_subdomains.set_all(0)

	dl.AutoSubDomain(right_boundary).mark(boundary_subdomains,1)

	my_ds = dl.ds(subdomain_data = boundary_subdomains)

	right_traction_expr = dl.Expression(("a*exp(-1.0*pow(x[1] - 0.5,2)/b)", "c*(1.0 + (x[1]/d))"),\
												 a=0.06, b=4, c=0.03, d=10, degree=5)
	right_t = dl.interpolate(right_traction_expr,Vh[hp.STATE])

	pde_varf = hyperelasticity_varf(Vh,my_ds,right_t)

	pde = CustomNonlinearPDEProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

	# Set up the prior

	correlation = settings['correlation_length']
	mean_function = dl.project(dl.Constant(0.37), Vh[hp.PARAMETER])
	prior = HyperelasticityPrior(Vh[hp.PARAMETER], correlation, mean = mean_function.vector())
 

	# Set up the observation and misfits
	nx_targets = settings['nx_targets']
	ny_targets = settings['ny_targets']
	ntargets = nx_targets*ny_targets
	assert ntargets == settings['ntargets'], 'Inconsistent target dimensions in settings'

	#Targets only on the bottom
	x_targets = np.linspace(0.1,0.9,nx_targets)
	y_targets = np.linspace(0.1,0.9,ny_targets)
	targets = []
	for xi in x_targets:
		for yi in y_targets:
			targets.append((xi,yi))
	targets = np.array(targets)

	print( "Number of observation points: {0}".format(ntargets) )

	misfit = hp.PointwiseStateObservation(Vh[hp.STATE], targets)

	model = hp.Model(pde, prior, misfit)
	model.targets = targets
	
	return model

def hyperelasticity_model_wrapper(settings = hyperelasticity_problem_settings()):
	model = hyperelasticity_model(settings = settings)

	mw_settings = hf.hippylibModelWrapperSettings()
	mw_settings['rel_noise'] = settings['rel_noise']
	mw_settings['seed'] = settings['seed']

	modelwrapper = hf.hippylibModelWrapper(model,settings = mw_settings)
	if hasattr(model,'targets'):
		modelwrapper.targets = model.targets		
	# Setup the inverse problem
	# modelwrapper.setUpInverseProblem()

	return modelwrapper

