#!/usr/bin/env python
import argh
import argparse
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.AtomDance import AtomDance
from irff.zmatrix import zmat_to_atoms

# get_ipython().run_line_magic('matplotlib', 'inline')

# zmat = [
# [ 'O',   0,   -1,   -1,   -1,   0.0000,  0.0000,   0.0000 ],
# [ 'C',   1,    0,   -1,   -1,   1.5336,  0.0000,   0.0000 ],
# [ 'O',   2,    1,    0,   -1,   1.5138,  180.0,   0.0000 ] ]
# atoms = zmat_to_atoms(zmat)

# view(atoms)
     
# with open('zmat.pkl','w') as f:
#      pickle.dump(zmat,f)
#      data = pickle.load(f)  

def wz(gen='cbd.gen',i=-1):
    atoms  = read(gen,index=i)
    ad     = AtomDance(atoms=atoms,rmax=1.25)
    zmat   = ad.InitZmat
    ad.write_zmat(zmat)
    ad.close()


def stretch(gen='poscar.gen',i=0,j=1,f=-1,s=1.2,e=1.9,dr=0.1):
    atoms  = read(gen,index=f)
    ad     = AtomDance(atoms=atoms,rmax=1.25)
    zmat   = ad.InitZmat
    traj   = TrajectoryWriter('md.traj',mode='w')
    
    # find index
    ii = ad.zmat_id.index(i)
    jj = ad.zmat_id.index(j)
    
    find = False
    if j == ad.zmat_index[ii][0]:
       find = True
       i_ = ii
    elif i == ad.zmat_index[jj][0]:
       find = True
       i_ = jj
    else:
       print('this bond is not found in the zmatrix!')
       return 
    
           
    # view(atoms)
    r = s
    while r<e:
        r += dr
        zmat[i_][0] = r
        atoms  = ad.zmat_to_cartation(atoms,zmat)
        ad.ir.calculate(atoms)

        atoms.calc = SinglePointCalculator(atoms,energy=ad.ir.E)
        traj.write(atoms=atoms)
 

if __name__ == '__main__':
   ''' use commond like ./zmat_pes.py w --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [wz,stretch])
   argh.dispatch(parser)
