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

from confusionModelSettings import confusion_problem_settings
from confusionLinearObservable import confusion_linear_observable


def confusion_model_wrapper(settings = confusion_problem_settings()):
	# Set up the mesh, finite element spaces, and PDE forward problem

	ndim = settings['ndim']
	nx = settings['nx']
	ny = settings['ny']
	mesh = dl.UnitSquareMesh(nx, ny)

	nx_targets = settings['nx_targets']
	ny_targets = settings['ny_targets']

	observable_kwargs = {'nx_targets':nx_targets,'ny_targets':ny_targets}
	observable = confusion_linear_observable(mesh,**observable_kwargs)

	pde = observable.problem
	Vh = pde.Vh
	# Set up the prior

	gamma = settings['gamma']
	delta = settings['delta']
		
	theta0 = settings['prior_theta0']
	theta1 = settings['prior_theta1']
	alpha  = settings['prior_alpha']
		
	anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree = 1)
	anis_diff.set(theta0, theta1, alpha)

	# Set up the prior
	prior = hp.BiLaplacianPrior(Vh[hp.PARAMETER], gamma, delta, anis_diff, robin_bc=True)
	print("Prior regularization: (delta_x - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(delta, gamma,2))    

	targets = observable._targets_

	ntargets = len(targets)

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

