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

def wz(gen='cbd.gen'):
    atoms  = read(gen,index=-1)
    ad     = AtomDance(atoms=atoms,rmax=1.25)
    zmat   = ad.InitZmat
    ad.write_zmat(zmat)
    ad.close()


def pes(gen='cbd.gen'):
    atoms  = read(gen,index=-1)
    ad     = AtomDance(atoms=atoms,rmax=1.25)
    zmat   = ad.InitZmat
    traj   = TrajectoryWriter('md.traj',mode='w')
    # view(atoms)
    
    r2 = r1 = np.linspace(1.0,1.5,50)
    R1,R2   = np.meshgrid(r1,r2)
    i_,j_   = R1.shape
    E       = np.zeros((i_,j_))
    zmat[2][1] = 180.0 

    for i in range(i_):
        for j in range(j_):
            r1 = R1[i][j]
            r2 = R2[i][j]
 
            zmat[1][0] = r1
            zmat[2][0] = r2
            atoms  = ad.zmat_to_cartation(atoms,zmat)
            ad.ir.calculate(atoms)

            E[i][j] = ad.ir.E
            atoms.calc = SinglePointCalculator(atoms,energy=ad.ir.E)
            traj.write(atoms=atoms)
    e_min = np.min(E)
    E = E- e_min
    traj.close()
    ad.close()

    fig = plt.figure()
    ax  = Axes3D(fig)   

    ax.set_zlabel(r'$Energy(eV)$', fontdict={'size': 15, 'color': 'b'})
    ax.set_ylabel(r'$R_{C-O}$', fontdict={'size': 15, 'color': 'b'})
    ax.set_xlabel(r'$R_{C-O}$', fontdict={'size': 15, 'color': 'b'})

    ax.plot_surface(R1,R2,E,alpha=0.85,cmap=plt.get_cmap('rainbow'))
    # ax.scatter(xs=r,ys=r_,zs=Edft,c='r', s=5, alpha=0.7, label='DFT', marker='*')
    # ax.contourf(r,ang,Edft,zdir='z', offset=0.0, cmap=plt.get_cmap('rainbow'))

    plt.legend()
    plt.savefig('co2pes.svg') 
    plt.close()
 

if __name__ == '__main__':
   ''' use commond like ./zmat_pes.py w --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [wz,pes])
   argh.dispatch(parser)