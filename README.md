# I-ReaxFF: stand for Intelligent-Reactive Force Field

- I-ReaxFF is a differentiable ReaxFF framework based on TensorFlow, with which we can get the first and high order derivatives of energies, and also can optimize ReaxFF parameters with integrated optimizers in TensorFlow.
---

* ffield.json: the parameter file from machine learning
* reaxff_nn.lib  the parameter file converted from ffield.json for usage with GULP

## Requirement
 the following package need to be installed
1. TensorFlow, pip install tensorflow --user or conda install tensorflow
2. PyTorch, conda install pyTorch
3. Numpy,pip install numpy --user
4. matplotlib, pip install matplotlib --user

Install this package after download this package and run commond in shell ``` python setup install --user ```. 
Alternatively, this package can be install without download the package through pip
``` pip install --user irff ```.


## Refference
Feng Guo et.al., Intelligent-ReaxFF: Evaluating the reactive force field parameters with machine learning, Computational Materials Science 172, 109393, 2020. 

Feng Guo et.al., ReaxFF-MPNN machine learning potential: a combination of reactive force field and message passing neural networks,Physical Chemistry Chemical Physics, 23, 19457-19464, 2021.

Feng Guo et.al., ReaxFF-nn: A Reactive Machine Learning Potential in GULP and the Applications in Low Dimensional Carbon nanostructures (Submitted)
