# I-ReaxFF: stand for Intelligent-Reactive Force Field

- I-ReaxFF is a differentiable ReaxFF framework based on TensorFlow, with which we can get the first and high order derivatives of energies, and also can optimize ReaxFF parameters with integrated optimizers in TensorFlow.
---

* ffield,ffield.json: the parameter file for the machine learning potential model.
* train.ipynb: shows a training example of the IRFF-MPNN model with data sets uploaded in the data directory.
* GeomentryOptimization.ipynb: shows how to do geometry optimizations with the IRFF-MPNN model.    
* MolecularDynamics.ipynb: shows how to do molecular dynamics with the IRFF-MPNN model.  
* StaticCompress.ipynb: shows the computation of the static compression of the solid nitromethane.   
(The .py files have the same content as .ipynb files, but supplied as runnable python file).   
To run all test files, just using commond like "python test.py".

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
Feng Guo et.al., Intelligent-ReaxFF: Evaluating the reactive force field parameters with machine learning, Computational Materials Science 172 (2020) 109393 

Feng Guo et.al., ReaxFF-MPNN machine learning potential: a combination of reactive force field and message passing neural networks,Physical Chemistry Chemical Physics, 2021, DOI: 10.1039/D1CP01656C
