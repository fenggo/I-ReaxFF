#!/usr/bin/env python
import numpy as np
from ase.io import read
from irff.molecule import press_mol

''' scale the crystal box, while keep the molecule structure unchanged
'''


A = read('md.traj',index=-1)
A = press_mol(A)
x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m

natom = len(A)

'''
rotae matrix
     cos_x   -sin_x
     sin_x    cos_x

'''

rotater = np.array([[[-1.0,0.0,0.0],
                    [ 0.0,1.0,0.0],
                    [ 0.0,0.0,-1.0] ]])

x_ = np.matmul(x_.reshape([natom,1,3]),rotater)
print(x_.shape)

A.set_positions(x_.reshape([natom,3]))
A.write('POSCAR')
