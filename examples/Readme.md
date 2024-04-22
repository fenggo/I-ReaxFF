1. example_ch4h2o: A example for building a machine learning potential from SIESTA calculations for a simple system, contains H2O and CH4 molecules.
2. example_c: A example for building a machine learning potential from Quantum Espresso DFT calculations for Carbon.
3. example_reax: A example for using a machine learning algorithoms training the ReaxFF parameters.
4. example_phonon: Using Phononpy and GULP compute the 2rd order force constant

## Introductions for building a mathine learning potential

### I. Use the following command to run the train process
```bash
nohup ./train.py --f=1 --t=1 --s=10000 --z=1 > py.log 2>&1 &
```
* options:

--f: whether optimize the four-body parameter

--t: whether optimize the three-body parameter

--h: whether optimize the hydrogen-bond parameter

--z: whether evaluate the zero point energy

--s: the train step

### II. Add new data (structure) to the current training data-set

1. A structure file should be constructed, in the ".gen" format, it can early be converted from POSCAR file, 
   if you familiar with this format, for example, using ASE code:
```python
from ase.io import read,write

atoms=read('POSCAR')
atoms.write('structure_name.gen')
```

2. prepare the md.traj file

   The md.traj file contains the structures from the potential energy surface, it canbe the trajectories of a short time molecular dynamics simulations, or it canbe strech the valence bond, or rotate the a functional group, or swing a valence angle. 
  
   This step is iteratively repeated, till performence of the potential satisfactory. 

3. run the lm.py script to call DFT calculation and trian the new data 
```bash
nohup ./lm.py --f=1 --t=1 --s=10000 --z=1 > py.log 2>&1 &
```
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
### III. Introduction of "train.py" script and "lm.py" script
