#!/usr/bin/env python
from irff.nwchem import decompostion
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# decompostion(gen='rdx.gen',ncpu=12)
images1 = Trajectory('stretch0.traj')
images2 = Trajectory('stretch.traj')

traj = TrajectoryWriter('decompostion.traj',mode='w')

rs,es = [],[]

for image in images1:
    vr  = image.positions[3] - image.positions[18]
    vr2 = np.square(vr)
    r = np.sqrt(np.sum(vr2))
    rs.append(r)

    e = image.get_potential_energy()
    es.append(e)
    #calc = SinglePointCalculator(A,energy=e)
    #image.set_calculator(calc)
    traj.write(atoms=image)

for image in images2:
    vr  = image.positions[3] - image.positions[18]
    vr2 = np.square(vr)
    r = np.sqrt(np.sum(vr2))
    rs.append(r)

    e = image.get_potential_energy()
    es.append(e)
    #calc = SinglePointCalculator(A,energy=e)
    #image.set_calculator(calc)
    traj.write(atoms=image)

traj.close()


plt.figure()             # test
plt.ylabel('Energies (eV)')
plt.xlabel('React coordinate')

plt.plot(rs,es,linestyle='-',marker='o',markerfacecolor='snow',
         markeredgewidth=1,markeredgecolor='k',
         ms=4,c='r',alpha=0.01,label='N-NO2')

plt.legend(loc='best')
plt.savefig('energies.png') 
plt.close() 

# 1 hatree = 27.2114eV=4.3597*10^(-18)J=2565.5kJ/mol/NA=627.5094kcal/mol

