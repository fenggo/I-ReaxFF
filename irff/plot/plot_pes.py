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
# get_ipython().run_line_magic('matplotlib', 'inline')


atoms  = read('h2o.gen',index=-1)
ad     = AtomDance(atoms=atoms,rmax=1.33)
zmat   = ad.InitZmat
# zmat[2][1] = 109.0


atoms_dft  = Trajectory('h2o.traj')
r,ang      = [],[]
Edft,Ereax = [],[]
r1,E1,E2   = [],[],[]

for atoms in atoms_dft:
    Edft.append(atoms.get_potential_energy())
    ad.ir.calculate(atoms)
    Ereax.append(ad.ir.E)
    zmat = ad.get_zmatrix(atoms)
    r.append(zmat[2][0])
    ang.append(zmat[2][1])
    if ang[-1] >= 179.0:
       r1.append(zmat[2][0])
       E1.append(ad.ir.E)
       E2.append(atoms.get_potential_energy())

edft_min = min(Edft)
ereax_min= min(Ereax) 
Edft     = np.array(Edft) - edft_min 
Ereax    = np.array(Ereax)- ereax_min

E1       = np.array(E1) - ereax_min
E2       = np.array(E2) - edft_min 

fig = plt.figure()
plt.xlabel(r"$R_{O-H}$")
plt.xlabel(r"$Angle_{H-O-H}$")
plt.plot(r1,E1,color='b',label='ReaxFF-MPNN') 
plt.plot(r1,E2,color='r',label='DFT') 
plt.legend()
plt.savefig('h2oevsr.svg') 
plt.close()


a       = 90.0
r_,ang_ = [],[]
for i in range(31):
    a  += 3.0
    ang_.append(a)
x = 0.7
for j in range(21):
    x += 0.03
    r_.append(x)

r_,ang_ = np.meshgrid(r_,ang_)
# print(r_.shape,ang_.shape)
i_,j_ = r_.shape 
Ereax_ = np.zeros((i_,j_))

for i in range(i_):
    for j in range(j_):
        zmat[2][0] = r_[i][j]
        zmat[2][1] = ang_[i][j]
        atoms = ad.zmat_to_cartation(atoms,zmat)
        ad.ir.calculate(atoms)
        Ereax_[i][j] = ad.ir.E

Ereax_ = Ereax_ - ereax_min
fig = plt.figure()
ax  = Axes3D(fig)
# ax  = plt.subplot(111, projection='3d')
ax.set_zlabel(r'$Energy(eV)$', fontdict={'size': 15, 'color': 'b'})
ax.set_ylabel(r'$Angle_{H-O-H}$', fontdict={'size': 15, 'color': 'b'})
ax.set_xlabel(r'$Radius_{OH}$', fontdict={'size': 15, 'color': 'b'})
ax.plot_surface(r_,ang_,Ereax_,cmap=plt.get_cmap('rainbow'))
ax.scatter(xs=r,ys=ang,zs=Edft,c='r', s=5, alpha=0.7, label='DFT', marker='^')
# ax.scatter(xs=r,ys=ang,zs=Ereax,c='b', s=5, alpha=0.7, label='Ereax', marker='^')
# ax.contourf(r,ang,Edft,zdir='z', offset=0.0, cmap=plt.get_cmap('rainbow'))

plt.legend()
plt.savefig('h2opes.svg') 
plt.close()
ad.close()

