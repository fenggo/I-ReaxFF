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

def wz(gen='cbd2.gen'):
    atoms  = read(gen,index=-1)
    ad     = AtomDance(atoms=atoms,rmax=1.25)
    zmat   = ad.InitZmat
    ad.write_zmat(zmat)
    ad.close()


def pes(gen='cbd2.gen'):
    atoms  = read(gen,index=-1)
    ad     = AtomDance(atoms=atoms,rmax=1.25)
    zmat   = ad.InitZmat
    traj   = TrajectoryWriter('md.traj',mode='w')
    
    #zmat[4][1] = 0.0

    atoms  = ad.zmat_to_cartation(atoms,zmat)
    # view(atoms)
 
    r   = np.linspace(1.4,2.0,50)
    ang = np.linspace(100.0,180.0,50)

    R,A   = np.meshgrid(r,ang)
    i_,j_   = R.shape
    E       = np.zeros((i_,j_))


    for i in range(i_):
        for j in range(j_):
            zmat[3][0] = R[i][j]
            zmat[4][1] = zmat[3][1] = A[i][j]
 
            atoms  = ad.zmat_to_cartation(atoms,zmat)
            ad.ir.calculate(atoms)

            E[i][j] = ad.ir.E
            # print(R1[i][j],R2[i][j],E[i][j])
            atoms.calc = SinglePointCalculator(atoms,energy=ad.ir.E)
            traj.write(atoms=atoms)

    e_min = np.min(E)
    E = E- e_min
    traj.close()
    ad.close()

    fig = plt.figure()
    ax  = Axes3D(fig)   

    ax.set_zlabel(r'$Energy(eV)$', fontdict={'size': 15, 'color': 'b'})
    ax.set_ylabel(r'$R_{O-O}$', fontdict={'size': 15, 'color': 'b'})
    ax.set_xlabel(r'$Angle_{C-O \dots O}$', fontdict={'size': 15, 'color': 'b'})

    ax.plot_surface(R,A,E,alpha=0.85,cmap=plt.get_cmap('rainbow'))
    # ax.scatter(xs=r,ys=r_,zs=Edft,c='r', s=5, alpha=0.7, label='DFT', marker='*')
    # ax.contourf(R1,R2,E,zdir='z', offset=0.0, cmap=plt.get_cmap('rainbow'))

    # plt.legend()
    plt.savefig('co22pes.svg') 
    plt.close()
 

if __name__ == '__main__':
   ''' use commond like ./zmat_pes.py wz --g=POSCAR to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [wz,pes])
   argh.dispatch(parser)



