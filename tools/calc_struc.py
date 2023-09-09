#!/usr/bin/env python
import numpy as np
from os import getcwd,chdir,mkdir,system
from os.path import exists
from ase.io import read
from ase.io.trajectory import Trajectory
from irff.dft.siesta import siesta_opt #, write_siesta_in


def opt_structure(ncpu=8,T=2500,us='F',gen='poscar.gen',l=0,i=-1,step=200):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=i)
    # A = press_mol(A)
    print('\n-  running siesta opt ...')
    vc = 'true' if l else 'false'
    siesta_opt(A,ncpu=ncpu,us=us,VariableCell=vc,tstep=step,
               xcf='GGA',xca='PBE',basistype='split')

def calc_strutures(traj,step=50,ncpu=8):
    images = Trajectory(traj)
    structures = range(190,239)
   
    root_dir   = getcwd()
    with open('density.log','w') as fd:
         print('# Crystal_id Density',file=fd)
    for s in structures:
        work_dir = root_dir+'/'+str(s)

        if exists(s):
           continue
        else:
           mkdir(s)

        chdir(work_dir)
        system('cp ../*.psf ./')
        img = siesta_opt(images[s-1],ncpu=ncpu,us='F',VariableCell='true',tstep=step,
                         xcf='GGA',xca='PBE',basistype='split')
        system('mv siesta.out siesta-{:d}.out'.format(s))
        system('rm siesta.* ')
        atoms = img[-1]
        masses = np.sum(atoms.get_masses())
        volume = atoms.get_volume()
        density = masses/volume/0.602214129
        
        chdir(root_dir)

        with open('density.log','a') as fd:
             print('{:5d} {:10.6f}'.format(s,density),file=fd)


if __name__=='__main__': 
   calc_strutures('../structures.traj',step=50,ncpu=8)
   