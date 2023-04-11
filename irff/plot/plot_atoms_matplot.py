#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import cm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.colors as col
from ase import Atoms
from ase.io import read,write
# from irff.plot.rffbd import RFFBD
from irff.irff_np import IRFF_NP


def p3d(traj='ps0.1.lammpstrj',frame=0,
    atomType =['C','H','O','N'],
          color={'C':'g','H':'khaki','O':'r','N':'b'},
          size = {'C':80,'H':40,'O':76,'N':76}):
    # atoms = LammpsHistory(traj=traj,frame=frame,atomType=atomType)
    atoms = read('poscar.gen')
    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)
    ir.calculate(atoms)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
     
    # set figure information
    ax.set_title("Atomic Configuration")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    e = []
    for e_ in ir.ebond:
        for em in e_:
            e.append(em)
    # print(e)
    mine = min(e)
    cmap = cm.ScalarMappable(col.Normalize(mine,0.0), cm.rainbow)

    for i in range(ir.natom):
        print('-  ploting bonds for atom {0} ... '.format(i),end='\r')
        for j in range(ir.natom):
            bd = ir.atom_name[i] + '-' + ir.atom_name[j]
            r = np.sqrt(np.sum(np.square(atoms.positions[j]-atoms.positions[i])))
            if j>i:
               if ir.r[i][j]<ir.rcut[bd]: 
                  x = [atoms.positions[i][0],atoms.positions[j][0]]
                  y = [atoms.positions[i][1],atoms.positions[j][1]]
                  z = [atoms.positions[i][2],atoms.positions[j][2]]
                  ax.plot(x,y,z,c=cmap.to_rgba(ir.ebond[i][j]),linewidth=1)

    print(' ',end='\n')
    # for a in atoms:   
    #     ax.scatter(a.x, a.y, a.z, c=color[a.symbol],s=size[a.symbol],label=a.symbol)
        
    ca = np.linspace(mine,0,100)
    cmap.set_array(ca)
    plt.colorbar(cmap,label='Color Map(Unit: eV)')
    # plt.show()
    plt.savefig('bondEnergy3d.eps')
    plt.close()


if __name__ == '__main__':
   p3d()

