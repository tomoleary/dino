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
import numpy as np

corrs = [0.3]

nxnys = [(64,64)]

for correlation_length in corrs:
	for nx,ny in nxnys:
		print(80*'#')
		print(('Running for corr = '+str(correlation_length)+' nx,ny = '+str((nx,ny))).center(80))
		os.system('mpirun -n 1 python hyperelasticityProblemSetup.py -ninstance 1 -correlation_length '+str(correlation_length)+' -nx '+str(nx)+' -ny '+str(ny))


