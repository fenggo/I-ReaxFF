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

zmat = [
[ 'H',  1,  -1,  -1,  -1,   0.0000,  0.0000,   0.0000 ],
[ 'O',  0,   1,  -1,  -1,   0.9686,  0.0000,   0.0000 ],
[ 'H',  2,   0,   1,  -1,   1.0552,115.3987,   0.0000 ],
[ 'O',  4,  -1,  -1,  -1,   0.0000,  0.0000,   0.0000 ],
[ 'C',  3,   4,  -1,  -1,   1.3073,  0.0000,   0.0000 ],
[ 'O',  5,   3,   4,  -1,   1.2773,179.9999,   0.0000 ] ]
atoms = zmat_to_atoms(zmat)


atoms  = read('hco.gen',index=-1)
ad     = AtomDance(atoms=atoms,rmax=1.33)
# zmat   = ad.InitZmat
# zmat[2][1] = 109.0


atoms_dft  = Trajectory('pes-hco.traj')
r,r_       = [],[]

r1,r1_     = [],[]
E1,E1_     = [],[]
r2,r2_     = [],[]
E2,E2_     = [],[]
r3,r3_     = [],[]
E3,E3_     = [],[]
r4,r4_     = [],[]
E4,E4_     = [],[]

R,R_       = [],[]
Edft,Ereax = [],[]


for atoms in atoms_dft:
    Edft.append(atoms.get_potential_energy())
    ad.ir.calculate(atoms)
    Ereax.append(ad.ir.E)
    # zmat = ad.get_zmatrix(atoms)
    r.append(ad.ir.r[4][2])
    r_.append(ad.ir.r[4][3])



# print(len(R),len(R_))
edft_min = min(Edft)
ereax_min= min(Ereax) 
Edft     = np.array(Edft) - edft_min 
Ereax    = np.array(Ereax)- ereax_min


R,R_ = np.meshgrid(R,R_)

i_,j_  = R.shape 
Ereax_ = np.zeros((i_,j_))

for i in range(i_):
    for j in range(j_):
        r1 = R[i][j]
        r2 = R_[i][j]
        images = ad.stretch([2,4],ToBeMoved=[4,3,5],atoms=atoms,nbin=2,rst=r1-0.0001,red=r1,
        	                neighbors=ad.neighbors)
        atoms  = images[-1]
        images = ad.stretch([4,3],ToBeMoved=[3,5],atoms=atoms,nbin=2,rst=r2-0.0001,red=r2,
        	                neighbors=ad.neighbors)
        atoms  = images[-1]
        # ad.ir.calculate(atoms)
        Ereax_[i][j] = atoms.get_potential_energy() # ad.ir.E

Ereax_ = Ereax_ - ereax_min
fig = plt.figure()
ax  = Axes3D(fig)   

ax.set_zlabel(r'$Energy(eV)$', fontdict={'size': 15, 'color': 'b'})
ax.set_ylabel(r'$Radius_{O-C}$', fontdict={'size': 15, 'color': 'b'})
ax.set_xlabel(r'$Radius_{H-O}$', fontdict={'size': 15, 'color': 'b'})

ax.plot_surface(R,R_,Ereax_,alpha=0.85,cmap=plt.get_cmap('rainbow'))
ax.scatter(xs=r,ys=r_,zs=Edft,c='r', s=5, alpha=0.7, label='DFT', marker='*')
# # ax.contourf(r,ang,Edft,zdir='z', offset=0.0, cmap=plt.get_cmap('rainbow'))

plt.legend()
plt.savefig('hcopes.svg') 
plt.close()
ad.close()

