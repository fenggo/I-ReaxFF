#!/usr/bin/env python
import numpy as np
from os import getcwd,chdir,mkdir,system
from os.path import exists
import sys
import argparse
from ase.io import read
from ase.io.trajectory import Trajectory
from irff.dft.siesta import siesta_opt #, write_siesta_in


class Stack():
    def __init__(self,entry=[]):
        self.entry = entry
        
    def push(self,x):
        self.entry.append(x) 

    def pop(self):
        return self.entry.pop()
    
    def close(self):
        self.entry = None

def read_individuals():
    enthalpy  = []
    gene      = {}
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
                   g = int(l[0])
                   i = int(l[1])
                   e = float(l[3])
                   d = float(l[5])
                   if l[6]=='N/A':
                     f = 99999
                   else:
                     f = float(l[6])
                   if g in gene:  
                      gene[g].append((i,e,d,f))
                   else:
                      gene[g] = [(i,e,d,f)]
                   # enthalpy.append(float(l[3]))
         st.close()

    k = gene.keys()
    k_ = max(k)
    return gene[k_]


def calc_individuals(traj,density=1.88,ids=None,step=50,ncpu=8):
    images = Trajectory(traj)
    if not ids:
       ids = []

       res = read_individuals()

       for i,e,d,f in res:
           if d>density and f<0.0:
              ids.append(i)

    root_dir   = getcwd()
    if not exists('density.log'):
       with open('density.log','w') as fd:
            print('# Crystal_id Density',file=fd)
         
    for s in ids:
        work_dir = root_dir+'/'+str(s)

        if exists(str(s)):
           continue
        else:
           mkdir(str(s))

        chdir(work_dir)
        system('cp ../../Specific/*.psf ./')
        img = siesta_opt(images[s-1],ncpu=ncpu,us='F',VariableCell='true',tstep=step,
                         xcf='GGA',xca='PBE',basistype='split')
        system('mv siesta.out siesta-{:d}.out'.format(s))
        system('mv siesta.traj md_{:d}.traj'.format(s))
        system('rm siesta.* ')
        atoms = img[-1]
        masses = np.sum(atoms.get_masses())
        volume = atoms.get_volume()
        density = masses/volume/0.602214129
        
        chdir(root_dir)

        with open('density.log','a') as fd:
             print('{:5d} {:10.6f}'.format(s,density),file=fd)


if __name__=='__main__': 
   parser = argparse.ArgumentParser(description='nohup ./calc_individuals.py --d=1.9 --n=8> py.log 2>&1 &')
   parser.add_argument('--d',default=1.95,type=float, help='the density that big than this value')
   parser.add_argument('--n',default=8,type=int, help='the number of CPU used to calculate')
   parser.add_argument('--s',default=300,type=int, help='the max number of steps used for geometry optimiztion')
   parser.add_argument('--i',default=[],type=list, help='the list of crystal id to be calculated')
   args = parser.parse_args(sys.argv[1:])

   # ids = range(190,239)
   calc_individuals('Individuals.traj',ids=args.i,density=args.d,step=300,ncpu=args.n)
   
