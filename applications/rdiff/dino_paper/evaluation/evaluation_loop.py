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

from argparse import ArgumentParser
# Arguments to be parsed from the command line execution
parser = ArgumentParser(add_help=True)
# Weights directory
parser.add_argument("-weights_dir", dest='weights_dir', required=True, help="Weights directory",type=str)
args = parser.parse_args()


ndatas = ['16','64','256','1024','4096','7168']

weights_dir = args.weights_dir

os.system('python link_weights.py')

for ndata in ndatas[:1]:

	warg = ' -weights_dir '+weights_dir
	narg = ' -ndata '+ndata
	# os.system('python evaluate_network_accuracies.py '+warg+narg)
	# os.system('python evaluate_network_gradients.py -logging_dir postproc/gradients/'+warg+narg)
	os.system('python evaluate_network_jacobians.py -logging_dir postproc/jacobians/'+warg+narg)