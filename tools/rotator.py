#!/usr/bin/env python
import numpy as np
from ase.io import read
from irff.molecule import press_mol

'''  
  rotae matrix
     cos_x   -sin_x
     sin_x    cos_x
'''


A = read('md.traj',index=-1)
A = press_mol(A)
x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m

natom = len(A)

rotater = np.array([[[1.0,0.0,0.0],
                     [ 0.0,0.0,-1.0],
                     [ 0.0,1.0,0.0] ]])

x_ = np.matmul(x_.reshape([natom,1,3]),rotater)
print(x_.shape)

cell  = A.get_cell()
cell_ = np.matmul(cell.reshape([3,1,3]),rotater)

A.set_positions(x_.reshape([natom,3]))
A.set_cell(cell_.reshape([3,3]))
A.write('POSCAR')
