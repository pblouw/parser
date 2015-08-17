This repository contains code for running the simulations described in the Proceeedings of the Cognitive Science Society paper [Constraint-based Parsing with Distributed Representations](https://mindmodeling.org/cogsci2015/papers/0051/paper0051.pdf). Below are instructions to get the model running and to replicate the results presented in the paper.

Dependencies
------------

The simulations require Numpy, a library for scientific computing in python. Generating the plots requires matplotlib, a plotting library. The easist way to get these packages is to install [Anaconda](https://store.continuum.io/cshop/anaconda/), a free python distribution that includes everything you need. Simply install Anaconda and then ensure that it is your default python installation (or, create an environment with each dependency and activate this environment from the command line before you proceed)

Running Simulations
-------------------

To reproduce the results in the paper, simply execute run.py and
wait for the simulation to complete. The results will be written to file in the 'results/' folder. Executing plot.py in this folder will generate the graphs depicted in the paper. Adjustments to the dimensionality of the vectors used in the simulations and other parameters of potential interest can be made in the body of run.py

Further documentation is included in each of the scripts contained in this repository, but if you have any questions after reading through the material, please contact me at pblouw@uwaterloo.ca 

