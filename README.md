# I-ReaxFF: stand for Intelligent-Reactive Force Field

- I-ReaxFF is a differentiable ReaxFF framework based on TensorFlow, with which we can get the first and high order derivatives of energies, and also can optimize **ReaxFF** and **ReaxFF-nn** (Reactive Force Field with Neural Networks) parameters with integrated optimizers in TensorFlow.

---

* ffield.json: the parameter file from machine learning
* reaxff_nn.lib  the parameter file converted from ffield.json for usage with GULP

## Requirement
 the following package need to be installed
1. TensorFlow, pip install tensorflow --user or conda install tensorflow
2. Numpy,pip install numpy --user
3. matplotlib, pip install matplotlib --user

Install this package after download this package and run commond in shell ``` python setup install --user ```. 
or using new command 
```shell
pip install .
```
Alternatively, this package can be install without download the package through pip
``` pip install --user irff ```.


## Refference
1. Feng Guo et.al., Intelligent-ReaxFF: Evaluating the reactive force field parameters with machine learning, Computational Materials Science 172, 109393, 2020. 

2. Feng Guo et.al., ReaxFF-MPNN machine learning potential: a combination of reactive force field and message passing neural networks,Physical Chemistry Chemical Physics, 23, 19457-19464, 2021.

3. Feng Guo et.al., ReaxFF-nn: A Reactive Machine Learning Potential in GULP/LAMMPS and the Applications in the Thermal Conductivity Calculations of Carbon Nanostructures, Physical Chemistry Chemical Physics, 27, 10571-10579, 2025.

### Use ReaxFF-nn with LAMMPS:
https://gitee.com/fenggo/ReaxFF-nn_for_lammps

https://github.com/fenggo/ReaxFF-nn_for_lammps

