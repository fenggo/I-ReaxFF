#!/usr/bin/env python
# coding: utf-8
from irff.checkLoneAtom import checkLoneAtoms,checkLoneAtom
from irff.irff_np import IRFF_NP
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase.visualize import view
from irff.prep_data import prep_data
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
import matplotlib.colors as col
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


images = Trajectory('md.traj')
view(images)
atoms = images[-1]



ir = IRFF_NP(atoms=atoms,
            libfile='ffield.json',
            rcut=None,
            nn=True)
# ir.get_pot_energy(atoms)
# ir.logout()
ir.calculate_Delta(atoms)
# print(ir.ebond)



fig = plt.figure()
ax = fig.gca(projection='3d')
 
# set figure information
ax.set_title("Bond Energy")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
mine = np.min(ir.ebond)

print(mine)
cmap = cm.ScalarMappable(col.Normalize(mine,0.0), cm.rainbow)

for i in range(ir.natom-1):
    for j in range(i+1,ir.natom):
        if ir.ebond[i][j]<-1.0:
           # print(ir.ebond[i][j])
           x = [atoms.positions[i][0],atoms.positions[j][0]]
           y = [atoms.positions[i][1],atoms.positions[j][1]]
           z = [atoms.positions[i][2],atoms.positions[j][2]]
           ax.plot(x,y,z,c=cmap.to_rgba(ir.ebond[i][j]),linewidth=3)
        
ca = np.linspace(mine,0,100)
cmap.set_array(ca)
plt.colorbar(cmap,label='Color Map(Unit: eV)')
plt.show()
# plt.savefig('bondEnergy.eps')

