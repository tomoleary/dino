# Instructions for Hyperelasticity problem

## 1. Generate the training data


First in order to generate the training data run one of the following commands:

`python hyperelasticityProblemSetup.py`

or with many simultaneous MPI processes:

`python generateHyperelasticity.py`

The command line arguments `-save_jacobian_data`, `-save_as` are set to `True` (`1`) by default. In order to generate a basis for PCANet (i.e., KLE of the input parameter), additional set the argument `-save_kle` to `True` (`1`). The data will initially be saved to `./data/` in a subfolder that specifies the specifics of the problem. When the data become large, it is also suitable to save them to a different location (e.g. a dedicated storage location) by modifying the location in `hyperelasticityProblemSetup.py`, or simply move the data after the process is complete.

## 2. Training the neural networks

The neural network scripts are all located in `dino_training/`. To run all neural network trainings used in the DINO paper, run

 `python training_runs.py` 

 Note that these runs may take very long, and were all run on a cluster with 1TB of RAM. The data are assumed to be loaded from a subfolder in `data/`. If this was moved somewhere else I suggest using symbolic links, (e.g., in bash `ln -s /path/to/moved/data/ data/`). 

 When these runs finish they will output trained weights (as pickled dictionaries) to a folder `trained_weights/` within the `dino_training/` directory. The reason the neural networks are not directly [saved and loaded using tensorflow](https://www.tensorflow.org/tutorials/keras/save_and_load) is due to the significant computational graph overhead due to extracting the Jacobians from the neural network. Perhaps a better way to handle breaking up the training of DINOs from their evaluation and deployment would be to instance an identical architecture (without the Jacobian computational graph overhead) at the end of training, and copying over the weights from the trained DINO to the copy, and then saving the copy using tensorflow.

## 3. Evaluate the trained neural networks

Once the neural networks are trained, and their weights have been saved, the networks can be evaluated using the following codes, which are located in `dino_training/evaluation/`.

Once in `dino_training/evaluation/`

`python evaluation_loop.py -weights_dir ../trained_weights/`


These scripts will output dictionaries of evaluated accuracies, gradient errors and Jacobian errors to `dino_training/evaluation/postproc/`




