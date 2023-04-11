#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import cm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.colors as col
import matplotlib.lines as mlines
from ase import Atoms
from ase.io import read,write
# from irff.plot.rffbd import RFFBD
from irff.irff_np import IRFF_NP
from ase.io.trajectory import Trajectory


def pam(traj,color={'C':'grey','H':'steelblue','O':'crimson','N':'dodgerblue'}, 
        size={'C':8000,'H':1200,'O':7000,'N':7000},
        bondColor='olive',boxColor='steelblue',bondWidth=20,
        elev=10,azim=120,Axis=True,Box=True,t=1,text='edge',labelnode=False):
    images     = Trajectory(traj)

    atoms      = images[0]
    sym        = atoms.get_chemical_symbols()

    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)
    ir.calculate_Delta(atoms)
    # plot scatter points
    fig, ax = plt.subplots(figsize=(8,8)) 
    plt.axis('off')

    mid = int(0.5*len(images))
    for i_ in [0,-1]:
        atoms = images[i_]
        ir.calculate_Delta(atoms)
        if i_==0 :
           alp= 0.9
        else:
           alp= 0.3
        for i,atom in enumerate(atoms):
            x_,y_,z_ = [],[],[]
            x_.append(atom.x)
            y_.append(atom.y)
            z_.append(atom.z)

            plt.scatter(atom.y, atom.z, c=color[sym[i]],
                       marker='o',s=size[sym[i]],label=sym[i],
                       alpha=alp)

        for i in range(ir.natom-1):
            for j in range(i+1,ir.natom):
                if ir.r[i][j]<ir.re[i][j]*1.25 and ir.ebond[i][j]<-0.01:
                   x = [atoms.positions[i][0],atoms.positions[j][0]]
                   y = [atoms.positions[i][1],atoms.positions[j][1]]
                   z = [atoms.positions[i][2],atoms.positions[j][2]]
                   # plt.plot(y,z,c=bondColor,linewidth=5,alpha=0.8)
                   line = mlines.Line2D(y,z,lw=bondWidth*ir.bo0[i][j],ls='-',alpha=alp,color=bondColor)
                   line.set_zorder(0)
                   ax.add_line(line)

#     ax.ylabel('Y', fontdict={'size': 15, 'color': 'b'})
#     ax.xlabel('X', fontdict={'size': 15, 'color': 'b'})
    svg = traj.replace('.traj','.svg')
    plt.savefig(svg,transparent=True)
    # plt.show()


if __name__ == '__main__':
   pam('md.traj')

