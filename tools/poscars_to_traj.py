#!/usr/bin/env python
import sys
import argparse
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator


help_ = './poscars_to_traj.py --f=gatheredPOSCARS'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--f',default='gatheredPOSCARS',type=str, help='poscars file name')
args = parser.parse_args(sys.argv[1:])


class Stack():
    def __init__(self,entry=[]):
        self.entry = entry
        
    def push(self,x):
        self.entry.append(x) 

    def pop(self):
        return self.entry.pop()
    
    def close(self):
        self.entry = None
    


def poscars_to_traj(fposcar):
    fbp = open(fposcar,'r')
    lines = fbp.readlines()
    fbp.close()

    traj =  TrajectoryWriter('structures.traj',mode='w')
    k        = 0
    s        = 0 
    energies = []
    

    with open('Individuals') as f:
         for line in f.readlines():
             st = Stack([])
             for x in line:
                if x!=']':
                    st.push(x)
                else:
                    x_ = ' '
                    while x_ !='[':
                        x_ = st.pop()
             line = ''.join(st.entry)
             l = line.split()
             
             if len(l)>=10:
                if l[0] != 'Gen':
                   energies.append(float(l[3]))
         st.close()
                   
    print(energies)

    for line in lines:
        if line.find('EA')>=0:
            if k>0:
                fpos.close()
                atoms = read('POSCAR')

                atoms.calc = SinglePointCalculator(atoms,energy=energies[s])
                traj.write(atoms=atoms)
                s += 1

            fpos = open('POSCAR','w')
            print(line[:-1], file=fpos)
            k += 1
        else:
            print(line[:-1], file=fpos)
    
    fpos.close()
    
    atoms = read('POSCAR')
    atoms.calc = SinglePointCalculator(atoms,energy=energies[s])
    traj.write(atoms=atoms)

    traj.close()


if __name__=='__main__':
   poscars_to_traj(args.f)


