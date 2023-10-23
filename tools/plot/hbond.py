#!/usr/bin/env python
import numpy as np
from os import getcwd, listdir
import matplotlib.pyplot as plt
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read 
from irff.irff_np import IRFF_NP



gens  = listdir(getcwd())
gens  = [gen for gen in gens if gen.endswith('.gen')]
atoms = read(gens[0])

ir = IRFF_NP(atoms=atoms,nn=True,libfile='ffield.json')
ir.calculate(atoms)

ir1 = IRFF_NP(atoms=atoms,nn=True,libfile='ffield_CHO.json')
ir1.calculate(atoms)

E,Ehb,D = [],[],[]
Ehb1  = []

for gen in gens:
    atoms = read(gen)
    ir.calculate(atoms)
    ir1.calculate(atoms)
    Ehb.append(-ir.Ehb)
    # ecoul = ir.Ecoul if abs(ir.Ecoul)>0.00000001 else 0.0
    # Ec.append(ecoul)
    E.append(ir.E)
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    D.append(density)
    Ehb1.append(-ir1.Ehb)

    print('Ehbond: {:8.4f}, Density: {:9.6}'.format(Ehb[-1],density) )

plt.figure()   
plt.ylabel(r'$Density$ ($g/cm^3$)')
plt.xlabel(r'$-1 \times HB$ $Energy$ ($eV$)')

# plt.subplot(2,1,1)
plt.scatter(Ehb,D,alpha=0.8,
            edgecolor='r', s=35,color='none',marker='o',
            label=r'$Total$ $HB$ $Energy$')

plt.scatter(Ehb1,D,alpha=0.8,
            edgecolor='b', s=35,color='none',marker='s',
            label=r'$HB(C-H\dots O)$ $Energy$')

x = np.linspace(0.59,0.77)
y = x*0.67 + 1.41

plt.plot(x,y,color='k',linestyle='-.')
# plt.subplot(2,1,2)
# plt.scatter(E,D,alpha=0.8,
#             edgecolor='r', s=20,color='none',
#             label=r'$Total$ $Energy$')

plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
plt.savefig('hbond.pdf',transparent=True) 
plt.close()

