# I-ReaxFF: stand for Intelligent-Reactive Force Field

ffield: the parameter file from machine learning
reax.lib  the parameter file converted from ffield for usage with GULP

*.gen file is final MD result from siesta or ReaxFF-(GULP), and can be visilized by ASE(command ase gui *.gen)

HBondCage.gif shows a molecular dynamics simulations results with ReaxFF (paramters from machine learning) at temperature of 50 K, which equivalent to a geomentry optimization but using a simulated annealing algorithm.


Requirement: the following package need to be installed
1. TensorFlow, pip install tensorflow --user or conda install tensorflow
2. Numpy,pip install numpy --user
3. matplotlib, pip install matplotlib --user

Install this package just run commond in shell "python setup install --user"

An example can be found in director I-ReaxFF/Al
