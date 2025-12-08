#!/usr/bin/env python
import sys
import argparse
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator


help_ = './poscars_to_traj.py --f=init_POSCARS'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--f',default='init_POSCARS',type=str, help='poscars file name')
args = parser.parse_args(sys.argv[1:])


def poscars_to_traj(fposcar):
    fbp = open(fposcar,'r')
    lines = fbp.readlines()
    fbp.close()

    traj =  TrajectoryWriter('Individuals.traj',mode='w')
    k        = 0
    s        = 0 
    # energies = []
    

    for line in lines:
        if line.find('ID_')>=0:
            if k>0:
                fpos.close()
                atoms = read('POSCAR')

                atoms.calc = SinglePointCalculator(atoms,energy=0.0)
                traj.write(atoms=atoms)
                s += 1

            fpos = open('POSCAR','w')
            print(line[:-1], file=fpos)
            k += 1
        else:
            print(line[:-1], file=fpos)
    
    fpos.close()
    
    atoms = read('POSCAR')
    atoms.calc = SinglePointCalculator(atoms,energy=0.0)
    traj.write(atoms=atoms)

    traj.close()


if __name__=='__main__':
   poscars_to_traj(args.f)


