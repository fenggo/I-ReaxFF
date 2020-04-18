import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from ase import Atoms
from ase.io import read,write
from ..irff_np import IRFF_NP


def view(atoms,color={'C':'dimgray','H':'silver','O':'crimson','N':'dodgerblue'}, 
	            size={'C':320,'H':90,'O':180,'N':320},
                bondColor='darkgoldenrod',boxColor='steelblue',bondWidth=2,
                elev=45,azim=45,Axis=True,Box=True):
    ''' avilable colors: ghostwhite whitesmoke olive '''
    positions  = atoms.get_positions()
    sym        = atoms.get_chemical_symbols()

    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)
    ir.calculate_Delta(atoms)

    # plot scatter points
    fig = plt.figure()
    ax = Axes3D(fig)
    if not Axis:
       ax.axis('off')
    x_,y_,z_ = [],[],[]
    for i,atom in enumerate(atoms):
        x_.append(atom.x)
        y_.append(atom.y)
        z_.append(atom.z)
        ax.scatter(atom.x, atom.y, atom.z, c=color[sym[i]],
                   marker='o',s=size[sym[i]],label=sym[i],
                   alpha=0.9)

    for i in range(ir.natom-1):
        for j in range(i+1,ir.natom):
            if ir.r[i][j]<ir.re[i][j]*1.25 and ir.ebond[i][j]<-0.01:
               x = [atoms.positions[i][0],atoms.positions[j][0]]
               y = [atoms.positions[i][1],atoms.positions[j][1]]
               z = [atoms.positions[i][2],atoms.positions[j][2]]
               ax.plot(x,y,z,c=bondColor,linewidth=5,alpha=0.6)

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
           ax.plot(x,y,z,c=boxColor,linewidth=bondWidth,alpha=0.8)

    ax.view_init(elev=elev,azim=azim) # azim: rotate around z，elev: around y
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'b'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'b'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'b'})
    plt.show()



