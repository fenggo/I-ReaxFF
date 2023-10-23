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

Ehb,D = [],[]

for gen in gens:
    atoms = read(gen)
    ir.calculate(atoms)
    Ehb.append(-ir.Ehb)
    # ecoul = ir.Ecoul if abs(ir.Ecoul)>0.00000001 else 0.0
    # Ec.append(ecoul)
    #e.append(ir.E-ir.zpe)
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    D.append(density)

    print('Ehbond: {:8.4f}, Density: {:9.6}'.format(Ehb[-1],density) )

plt.figure()   
plt.ylabel(r'$Hydrogen Bond Energy$ ($\times -1 eV$)')
plt.xlabel(r'$Density$ ($g/cm^3$)')


plt.scatter(Ehb,D,alpha=0.8,
            edgecolor='r', s=20,color='none',
            label=r'$Hydrogen Bond Energy v.s. Density$')

plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
plt.savefig('hbond.pdf',transparent=True) 
plt.close()

