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

import math

def confusion_problem_settings(settings = {}):
	"""
	"""
	
	settings['ntargets'] = 50
	settings['nx_targets'] = 10
	settings['ny_targets'] = 5
	settings['gamma'] = 0.1
	settings['delta'] = 1.0
	settings['nx'] = 64
	settings['ny'] = 64
	settings['ndim'] = 2
	settings['jacobian_full_rank'] = 50
	settings['formulation'] = 'confusion'
	settings['rel_noise'] = 0.01

	# For prior anisotropic tensor
	settings['prior_theta0'] = 2.0
	settings['prior_theta1'] = 0.5
	settings['prior_alpha'] = math.pi/4

	# For sampling
	settings['seed'] = 1


	return settings