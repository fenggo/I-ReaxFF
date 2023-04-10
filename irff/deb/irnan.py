#!/usr/bin/env python
from irff.irff import IRFF
from irff.irff_np import IRFF_NP
from ase.io import read,write
import numpy as np
from ase.io.trajectory import Trajectory,TrajectoryWriter


def irff_deb(gen='poscar.gen'):
    atoms  = read(gen)
    ir = IRFF(atoms=atoms,
              libfile='ffield.json',
              nn=True,
              autograd=True)
    ir.calculate(atoms)
    print(ir.grad)

    ir_ = IRFF_TF(atoms=atoms,
              libfile='ffield.json',
              nn=True)
    ir_.calculate(atoms)
    print(ir_.grad.numpy())


if __name__ == '__main__':
   ''' use commond like ./mpnn.py <opt> to run it
       use --h to see options
   '''
   irff_deb(gen='md.traj')



