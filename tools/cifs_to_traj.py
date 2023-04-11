#!/usr/bin/env python
from os import getcwd,listdir
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator


def cifs_to_traj():
    traj =  TrajectoryWriter('structures.traj',mode='w')
    current_dir = getcwd()
    cifs = listdir(current_dir)
    
    for cif_ in cifs:
        if cif_.endswith('.cif'):
           atoms = read(cif_)
           calc = SinglePointCalculator(atoms,energy=0.0)
           atoms.calc = calc
           traj.write(atoms=atoms)

    traj.close()


if __name__=='__main__':
   cifs_to_traj()


