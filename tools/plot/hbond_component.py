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

ir2 = IRFF_NP(atoms=atoms,nn=True,libfile='ffield_CHN.json')
ir2.calculate(atoms)

E,Ehb,D    = [],[],[]
Ehb1,Ehb2  = [],[]
ids        = []

for gen in gens:
    atoms = read(gen)
    id_ = gen.split('.')[0]
    ids.append(id_)
    ir.calculate(atoms)
    ir1.calculate(atoms)
    ir2.calculate(atoms)
    Ehb.append(-ir.Ehb)
    # ecoul = ir.Ecoul if abs(ir.Ecoul)>0.00000001 else 0.0
    # Ec.append(ecoul)
    E.append(ir.E)
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    D.append(density)
    Ehb1.append(-ir1.Ehb)
    Ehb2.append(-ir2.Ehb)

    print('Ehbond: {:8.4f}, Ehbond1: {:8.4f}, Ehbond2: {:8.4f}'.format(Ehb[-1],Ehb1[-1],Ehb2[-1]) )


fig = plt.figure()
plt.xlabel('$ID$')
plt.ylabel('$Density(g \cdot cm^3)$')
plt.xticks(range(0,len(ids)),labels=ids,rotation=65)

#plt.ylim(0.775,1.0)
# plt.yticks(np.arange(1.72, 1.91, 0.02))
x_width = range(0,len(ids))
x1_width = [i-0.25 for i in x_width]
x2_width = [i for i in x_width]
x3_width = [i+0.25 for i in x_width]

plt.bar(x1_width, Ehb, color='tomato', width=0.25, label='$All HB$')
plt.bar(x2_width, Ehb1,color='turquoise', width=0.25, label='$HB(C-H\dots O)$')
plt.bar(x3_width, Ehb2,color='blue', width=0.25, label='$HB(C-H\dots N)$')

plt.legend(loc='upper right',edgecolor='#fb8402')
# plt.title('$b$', x=-0.05, y=-0.16, fontsize=30)

plt.savefig('hb_component.pdf',format='pdf',bbox_inches='tight',transparent=True)
plt.close()

