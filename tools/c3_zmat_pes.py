#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.AtomDance import AtomDance
from irff.zmatrix import zmat_to_atoms
# get_ipython().run_line_magic('matplotlib', 'inline')

zmat = [[ 'C',  0,  -1,  -1,  -1,   0.0000,  0.0000,   0.0000 ],
        [ 'C',  1,   0,  -1,  -1,   1.5401,  0.0000,   0.0000 ],
        [ 'C',  2,   1,   0,  -1,   1.3748,135.4097,   0.0000 ]]
atoms = zmat_to_atoms(zmat)


# atoms  = read('c3.gen',index=-1)
ad     = AtomDance(atoms=atoms,rmax=1.33)
images  = ad.stretch([0,1],nbin=30,rst=1.1,red=1.7,scale=1.26,traj='md-c3.traj')
ad.close()


zmat = [[ 'C',  0,  -1,  -1,  -1,   0.0000,  0.0000,   0.0000 ],
        [ 'C',  1,   0,  -1,  -1,   1.5401,  0.0000,   0.0000 ]]
atoms = zmat_to_atoms(zmat)
# atoms  = read('c3.gen',index=-1)
ad     = AtomDance(atoms=atoms,rmax=1.33)
images  = ad.stretch([0,1],nbin=30,rst=1.1,red=1.7,scale=1.26,traj='md-c2.traj')
ad.close()

# view(images)

