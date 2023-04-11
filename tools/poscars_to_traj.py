#!/usr/bin/env python
import sys
import argparse
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator


help_ = './poscars_to_traj.py --f=BESTgatheredPOSCARS'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--f',default='BESTgatheredPOSCARS',type=str, help='poscars file name')
args = parser.parse_args(sys.argv[1:])


def poscars_to_traj(fposcar):
    fbp = open(fposcar,'r')
    lines = fbp.readlines()
    fbp.close()

    traj =  TrajectoryWriter('structures.traj',mode='w')
    k = 0
    
    for line in lines:
        if line.find('EA')>=0:
            if k>0:
                fpos.close()
                atoms = read('POSCAR')
                calc = SinglePointCalculator(atoms,energy=0.0)

                atoms.set_calculator(calc)
                traj.write(atoms=atoms)

            fpos = open('POSCAR','w')
            print(line[:-1], file=fpos)
            k += 1
        else:
            print(line[:-1], file=fpos)
    
    fpos.close()
    
    atoms = read('POSCAR')
    calc = SinglePointCalculator(atoms,energy=0.0)
    atoms.set_calculator(calc)
    traj.write(atoms=atoms)

    traj.close()


if __name__=='__main__':
   poscars_to_traj(args.f)


