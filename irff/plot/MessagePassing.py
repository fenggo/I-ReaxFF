#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from ase import Atoms
from ase.io import read,write
from irff.irff_np import IRFF_NP
from ase.io.trajectory import Trajectory


def messagePassing(atoms,color={'C':'dimgray','H':'silver','O':'crimson','N':'dodgerblue'}, 
                   size={'C':320,'H':90,'O':180,'N':320},
                   bondColor='darkgoldenrod',boxColor='steelblue',
                   bondWidth=1,latticeWidth=2,bocut=0.0001,
                   elev=45,azim=45,Axis=True,Box=True,t=0,text='edge',labelnode=False,
                   ball_scale=20,ray_scale=100,n_ray_steps=10,
                   show=False,show_element=False,
                   figname='messagepassing.svg'):
    ''' avilable colors: ghostwhite whitesmoke olive '''
    positions  = atoms.get_positions()
    sym        = atoms.get_chemical_symbols()

    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)
    ir.calculate(atoms)
    normalized_charges = ir.q.copy()
    normalized_charges[np.isnan(normalized_charges)] = 0
    max_charge = np.max(np.abs(normalized_charges))
    normalized_charges /= max_charge
    normalized_charges = (normalized_charges + 1) / 2
    color_map = plt.get_cmap("bwr_r")
    colors = color_map(normalized_charges)

    ball_sizes = np.array([ir.p['rvdw_'+e] for e in ir.atom_name])*ball_scale
    ray_full_sizes = ball_sizes + np.abs(ir.q)*ray_scale  
    ray_sizes = np.array([
            np.linspace(ray_full_sizes[i], ball_sizes[i], n_ray_steps, endpoint=False)
                        for i in range(ir.natom)]).T
    # plot scatter points
    # fig, ax = plt.subplots() 
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    bonds      = []
    bondColors = [] #color_map(0.5)
    bWs        = [] 
    for i in range(ir.natom-1):
        for j in range(i+1,ir.natom):
            if ir.H[t][i][j]>bocut:
               bonds.append((atoms.positions[i],atoms.positions[j]))
               ax.text(0.5*(atoms.positions[i][0]+atoms.positions[j][0]),
                       0.5*(atoms.positions[i][1]+atoms.positions[j][1]),
                       0.5*(atoms.positions[i][2]+atoms.positions[j][2]),
                       r'$%3.3f$' %ir.H[t][i][j],
                       ha="center", va="center", zorder=100,fontsize=16,color='k')
            bondColors.append(color_map(0.5*(normalized_charges[i]+normalized_charges[j])))
            bWs.append(ir.H[t][i][j])
    sticks = Line3DCollection(bonds,color=bondColors,
                              linewidths=bondWidth*np.array(bWs),
                              alpha=0.8)
    ax.add_collection(sticks)

    if show_element:
       for i,atom in enumerate(atoms):
           ax.text(atom.x,atom.y,atom.z,ir.atom_name[i]+str(i),color="black",
                   ha="center", va="center", zorder=100,fontsize=16)
    ax.scatter(*atoms.positions.T, c=colors,s=ball_sizes**2,alpha=1.0)
    ax.set_facecolor((0.05, 0.05, 0.05))
    ax.get_figure().set_facecolor((0.05, 0.05, 0.05))

    # Plots the rays
    for i in range(n_ray_steps):
        ax.scatter(*atoms.positions.T, s=ray_sizes[i]**2, c=colors,
            linewidth=0, alpha=0.05)
    # ymin_ =min( ymin,yl+0.1*yr)
    # ax.text(xmin,ymin_,zmin_,r'$BO^{t=%d}$' %t,fontsize=16)

    if Box:
       # plot lattice
       cell = atoms.get_cell()
       crystalVetexes = [np.zeros(3),cell[0],cell[1],cell[2]]  
       crystalVetexes.append(cell[0]+cell[1])
       crystalVetexes.append(cell[0]+cell[2])
       crystalVetexes.append(cell[1]+cell[2])
       crystalVetexes.append(crystalVetexes[4]+cell[2])
       edges = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,6],[2,4],[3,5],[3,6],[4,7],[5,7],[6,7]]
       for e in edges:
           i,j = e
           x = [crystalVetexes[i][0],crystalVetexes[j][0]]
           y = [crystalVetexes[i][1],crystalVetexes[j][1]]
           z = [crystalVetexes[i][2],crystalVetexes[j][2]]
           ax.plot(x,y,z,c=boxColor,linewidth=latticeWidth,alpha=0.9)

    fig.tight_layout()
    ax.azim = azim
    ax.elev = elev
    ax.axis("off") # Remove frame
    plt.savefig(figname,transparent=True)
    if show: 
       plt.show()
    plt.close()
    # _set_box(axes, atoms.coord, center, size, zoom)


# def _set_box(axes, coord, center, size, zoom):
#     """
#     This ensures an approximately equal aspect ratio in a 3D plot under
#     the condition, that the :class:`Axes` is quadratic on the display.
#     """
#     if center is None:
#         center = (
#             (coord[:, 0].max() + coord[:, 0].min()) / 2,
#             (coord[:, 1].max() + coord[:, 1].min()) / 2,
#             (coord[:, 2].max() + coord[:, 2].min()) / 2,
#         )

#     if size is None:
#         size = np.array([
#             coord[:, 0].max() - coord[:, 0].min(),
#             coord[:, 1].max() - coord[:, 1].min(),
#             coord[:, 2].max() - coord[:, 2].min()
#         ]).max()
    
#     axes.set_xlim(center[0] - size/(2*zoom), center[0] + size/(2*zoom))
#     axes.set_ylim(center[1] - size/(2*zoom), center[1] + size/(2*zoom))
#     axes.set_zlim(center[2] - size/(2*zoom), center[2] + size/(2*zoom))
    
#     # Make the axis lengths of the 'plot box' equal
#     # The 'plot box' is not visible due to 'axes.axis("off")'
#     axes.set_box_aspect([1,1,1])

if __name__ == '__main__':
   atoms = read('c2h6.gen')
   messagePassing(atoms,color={'C':'grey','H':'steelblue','O':'crimson','N':'dodgerblue'}, 
                  size={'C':5000,'H':800,'O':5000,'N':5000},
                  bondColor='olive',boxColor='steelblue',bondWidth=10,
                  elev=0,azim=0,Axis=False,Box=False,text='edge',t=0)



