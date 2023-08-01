				Derivative Informed Neural Operator

			     _____                      ___           ___     
			    /  /::\       ___          /__/\         /  /\    
			   /  /:/\:\     /  /\         \  \:\       /  /::\   
			  /  /:/  \:\   /  /:/          \  \:\     /  /:/\:\  
			 /__/:/ \__\:| /__/::\      _____\__\:\   /  /:/  \:\ 
			 \  \:\ /  /:/ \__\/\:\__  /__/::::::::\ /__/:/ \__\:\
			  \  \:\  /:/     \  \:\/\ \  \:\~~\~~\/ \  \:\ /  /:/
			   \  \:\/:/       \__\::/  \  \:\  ~~~   \  \:\  /:/ 
			    \  \::/        /__/:/    \  \:\        \  \:\/:/  
			     \__\/         \__\/      \  \:\        \  \::/   
			                               \__\/         \__\/    

		An Efficient Framework for High-Dimensional Parametric Derivative Learning


* PDE data generation is handled by `FEniCS` `hIPPYlib`, and `hippyflow`. For this [`hIPPYlib`](https://github.com/hippylib/hippylib) and [`hippyflow`](https://github.com/hippylib/hippyflow) must be installed. 

With conda

* `conda create -n dino -c uvilla -c conda-forge fenics==2019.1.0 tensorflow=2.7.0 matplotlib scipy`

Assumes that the environmental variables `HIPPYLIB_PATH`, `HIPPYFLOW_PATH` and `DINO_PATH` have been set.

* `export HIPPYLIB_PATH=path/to/hippylib`
* `export HIPPYFLOW_PATH=path/to/hippyflow`
* `export DINO_PATH=path/to/dino`


## Machine learning in Tensorflow (Beware of version / eager execution)

Neural network training is handled by `keras` within `Tensorflow`. The way that the Jacobians are extracted at present requires that some tensorflow v2 behaviour is disabled. This creates issues with eager execution in later versions of tensorflow. This library works with tensorflow `2.7.0`. In the future, `dino` may be reworked to handle the eager execution issue in later versions of tensorflow.  
