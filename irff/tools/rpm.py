#!/usr/bin/env python
from irff.molecule import packmol, press_mol
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter
from ase import Atoms




# packmol(strucs=['cl20mol','hmxmol','co2','CO','no2','NO','h2o'],
#         supercell=[2,2,2],
#         w=True)

A = read('siesta.traj',index=-1)
A = press_mol(A,inbox=False)
A.write('pos.gen')


