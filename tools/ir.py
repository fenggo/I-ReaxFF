#!/usr/bin/env python
import numpy as np
from irff.irff_force import IRFF_FORCE
from irff.MPNN import MPNN
from ase.io import read,write


atoms = read('siesta.traj',index=29)

ir = IRFF_FORCE(atoms=atoms,
          libfile='ffield.json')

ir.get_pot_energy(atoms)
print(ir.bop_si)
print(ir.bop_pi)
print(ir.bop_pp)

