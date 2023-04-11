#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory


def gofr(atoms,pair=('C','C'),bins=0.01,rcut=8.0):
    volume = atoms.get_volume()
    nbin   = int(rcut/bins)
    gr_    = np.zeros([nbin])
    natom1 = 0
    natom2 = 0
    natom  = len(atoms)
    symbols= atoms.get_chemical_symbols()
    for atom in atoms:
        # print(dir(atom))
        if atom.symbol == pair[0]:
           natom1 += 1
        if atom.symbol == pair[1]:
           natom2 += 1

    rou   = natom2/volume
    pi    = 3.1415926
    s     = 1.0
    positions = atoms.get_positions()
    cell  = atoms.get_cell()

    xi    = np.expand_dims(positions,axis=0)
    xj    = np.expand_dims(positions,axis=1)
    vr    = xj-xi
    rcell = np.linalg.inv(cell).astype(dtype=np.float32)
  
    vrf   = np.dot(vr,rcell)                   # PBC
    vrf   = np.where(vrf-0.5>0,vrf-1.0,vrf)
    vrf   = np.where(vrf+0.5<0,vrf+1.0,vrf)  
    vr    = np.dot(vrf,cell)
    r     = np.sqrt(np.sum(vr*vr,axis=2))

    for i in range(natom-1):
        for j in range(i,natom):
            # print(type(symbols[i]),symbols[i],symbols[j],symbols[i]==pair[0],pair[0])
            if (symbols[i]==pair[0] and symbols[j]==pair[1]) or \
               (symbols[i]==pair[1] and symbols[j]==pair[0]):
               r_ = r[i][j]
               if r_<=rcut and r_>0.00001:

                  nb = int(r_/bins)
                  # print(nb)
                  rsq= ((nb+1)*bins)**2
                  gr_[nb] += s/(rou*4*pi*rsq*bins*natom1)

    bin_= np.array([bins*i+0.5*bins for i in range(nbin)])
    # bin_= np.linspace(0.0,rcut,nbin)
    return bin_,gr_



if __name__ == '__main__':
   g = []
   pair=('O','H')
   images = Trajectory('gulp.traj')

   for atoms in images:
       r,gr = gofr(atoms,pair=pair)
       g.append(gr)
   gr = np.average(g,axis=0)

   plt.figure()
   plt.ylabel('pair radical distribution function')
   plt.xlabel('Radius (Angstrom)')
   # plt.xlim(0,i)
   # plt.ylim(0,np.max(hist)+0.01)

   plt.plot(r,gr,alpha=0.8,
            linestyle='-',# marker=markers[i_],markersize=5,
            color='red',label='{:s}-{:s}'.format(pair[0],pair[1]))

   plt.legend(loc='best',edgecolor='yellowgreen')
   plt.savefig('PairDistributionFunction.pdf',transparent=True) 
   plt.close() 


