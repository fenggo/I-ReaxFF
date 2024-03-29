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


def pes(gen='cbd.gen',i=-1):
    atoms  = read(gen,index=i)
    ad     = AtomDance(atoms=atoms,rmax=1.25)
    zmat   = ad.InitZmat
    traj   = TrajectoryWriter('md.traj',mode='w')
    # view(atoms)
    r = 1.47 
    for i_ in range(23):
        zmat[46][0] = r+0.01*i_
        atoms  = ad.zmat_to_cartation(atoms,zmat)
        ad.ir.calculate(atoms)

        atoms.calc = SinglePointCalculator(atoms,energy=ad.ir.E)
        traj.write(atoms=atoms)
 

if __name__ == '__main__':
   ''' use commond like ./zmat_pes.py w --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [wz,pes])
   argh.dispatch(parser)
