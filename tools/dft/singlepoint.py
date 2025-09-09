#!/usr/bin/env python
import subprocess
import numpy as np
from os import getcwd,chdir,mkdir,system
from os.path import exists
import sys
import argparse
from ase.io import read
from ase.io.trajectory import Trajectory,TrajectoryWriter
from irff.dft.siesta import siesta_opt,single_point #, write_siesta_in


class Stack():
    def __init__(self,entry=[]):
        self.entry = entry
        
    def push(self,x):
        self.entry.append(x) 

    def pop(self):
        return self.entry.pop()
    
    def close(self):
        self.entry = None

def calc_individuals(traj,density=1.88,ids=None,step=50,ncpu=8):
    images = Trajectory(traj)
    # if not ids:
    #    ids = []
    #    res = read_individuals()
    #    for i,e,d,f in res:
    #        if d>density and f<0.0:
    #           ids.append(i)
    # else:
    res = TrajectoryWriter('results.traj',mode='w')
    ids = [int(i) for i in range(len(images))]

    root_dir   = getcwd()
    if not exists('density.log'):
       with open('density.log','w') as fd:
            print('# Crystal_id Density Energy',file=fd)

    for s in ids:
        work_dir = root_dir+'/'+str(s)

        if exists(str(s)):
           continue
        else:
           mkdir(str(s))

        chdir(work_dir)
        system('cp ../*.psf ./')
        # img = siesta_opt(images[s-1],ncpu=ncpu,us='F',VariableCell='true',tstep=step,
        #                  xcf='GGA',xca='PBE',basistype='split')
                         # xcf='VDW',xca='DRSLL',basistype='split')
                         # xcf='VDW',xca='DRSLL')
        img = single_point(images[s-1],id=0,
                           xcf='GGA',xca='PBE',basistype='split',
                           val={'C':4.0,'H':1.0,'O':6.0,'N':5.0,'F':7.0,'Al':3.0},
                           cpu=ncpu)               
        system('rm *.xml ')
        system('rm INPUT_TMP.* ')
        system('rm fdf-* ')
        atoms = img # [-1]
        atoms.write('POSCAR.{:d}'.format(s))
        res.write(atoms=atoms)
        masses = np.sum(atoms.get_masses())
        volume = atoms.get_volume()
        density = masses/volume/0.602214129
        energy  = atoms.get_potential_energy()
        
        chdir(root_dir)

        with open('density.log','a') as fd:
             print('{:5d} {:10.6f} {:10.8f}'.format(s,density,energy),file=fd)
    res.close()


if __name__=='__main__': 
   parser = argparse.ArgumentParser(description='nohup ./calc_individuals.py --d=1.9 --n=8> py.log 2>&1 &')
   parser.add_argument('--d',default=1.95,type=float, help='the density that big than this value')
   parser.add_argument('--n',default=16,type=int, help='the number of CPU used to calculate')
   parser.add_argument('--s',default=300,type=int, help='the max number of steps used for geometry optimiztion')
   parser.add_argument('--t',default='md.traj',type=str, help='the trajectory file name')
   parser.add_argument('--i',default='',type=str, help='the list of crystal id to be calculated')
   args = parser.parse_args(sys.argv[1:])

   # ids = range(190,239)
   calc_individuals(args.t,ids=args.i,density=args.d,step=300,ncpu=args.n)

