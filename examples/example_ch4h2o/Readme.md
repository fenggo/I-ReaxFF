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

