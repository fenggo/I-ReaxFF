1. example_ch4h2o: A example for building a machine learning potential from SIESTA calculations for a simple system, contains H2O and CH4 molecules.
2. example_c: A example for building a machine learning potential from Quantum Espresso DFT calculations for a Carbon.

## Introductions for building a mathine learning potential

### I. Use the following command to run the train process

In this example two train script have providied, "train_without_force.py" and "train.py"

* train_without_force.py
```bash
nohup ./train_without_force.py --f=1 --t=1 --s=10000 --z=1 > py.log 2>&1 &
```
* options:

--f: whether optimize the four-body parameter

--t: whether optimize the three-body parameter

--h: whether optimize the hydrogen-bond parameter

--z: whether evaluate the zero point energy

--s: the train step

* train.py
```bash
./train.py   
```
Variables need to be set in the Python source file "train.py".

Important variable introductions:

* weight_force

e.g. weight_force  = {'h2o16-0':0,'ch4w2-0':1}, where 'h2o16' represents the structure name, 'h2o16-0' represents the 
number of orders of the batch of data. 0 represents forces that are not used for train, and 1 represents forces that are to be trained.

Force training will use a large amount of GPU memory, in practice, we only train the force of one structure.

The output of this script:

```bash
  step: 830 loss: 0.0407071 accs: 0.800650 h22-v: 0.7193 h2o2-0: 0.8611 ch4w2-0: 0.7977 h2o16-0: 0.8246  force: 0.473940 pen: 13.7729 me: 0.0873 time: 1.9033
```

the value after "loss" are losses of energy per atom, and the value behind "force" are losses of forces per atom, the loss of force smaller than 0.1 
is enough for reaction simulations. However, the smaller this value is, the higher precision the potential will has, you can train the loss force as small as you can.

### II. Add new data (structure) to the current training data-set

1. A structure file should be constructed, in the ".gen" format, it can early be converted from POSCAR file, 
   if you are familiar with this format, for example, using ASE code:
```python
from ase.io import read, write

atoms=read('POSCAR')
atoms.write('structure_name.gen')
```

2. prepare the md.traj file

   The md.traj file contains the structures from the potential energy surface, it canbe the trajectories of a short time molecular dynamics simulations, or it canbe strech the valence bond, or rotate the a functional group, or swing a valence angle. 
  
   This step is iteratively repeated, till performence of the potential satisfactory. 

3. run the lm.py script to call DFT calculation and generate the new data containing energies and force of some structure.

```bash
nohup ./lm.py --f=1 --t=1 --s=10000 --z=1 > py.log 2>&1 &
```
or 
```bash
./lm.py 
```
run it directly.

4. convert the parameter file "ffield.json" to the GULP format or lammps format

to GULP format
```bash
./json_to_lib.py
```
to LAMMPS format
```bash
./json_to_ffield.py
```
These two script can be found in "I-ReaxFF/tools/" directory.

