#!/usr/bin/env python
from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

atoms = read('md.traj',index=-1)
structure = AseAtomsAdaptor.get_structure(atoms)

structure.to(filename="POSCAR")
