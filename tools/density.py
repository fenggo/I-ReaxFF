#!/usr/bin/env python
import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.data import chemical_symbols


atoms = read('siesta.traj')

masses = np.sum(atoms.get_masses())
volume = atoms.get_volume()
density = masses/volume/0.602214129

print('Volume: ',volume, 'Masses: ',masses,'Density: ',density)
