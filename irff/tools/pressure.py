#!/usr/bin/env python
from ase.io import read
from ase.units import kJ # ,GPa
from ase.eos import EquationOfState
from ase.optimize import BFGS,FIRE
from irff.md import opt
from irff.irff import IRFF
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt


images = Trajectory('pre_opt.traj',mode='r')

ir = IRFF(atoms=images[0],
        libfile='ffield.json',
        rcut=None,
        nn=True)
GPa = 1.60217662*1.0e2

v_,p   = [],[]
v0  = images[0].get_volume()

for atoms_ in images:
    ir.calculate(atoms=atoms_,CalStress=True)
    stress  = ir.results['stress']

    nonzero = 0
    stre_   = 0.0
    for _ in range(3):
        if abs(stress[_])>0.0000001:
           nonzero += 1
           stre_   += -stress[_]
    pressure = stre_*GPa/nonzero

    p.append(pressure)
    v = atoms_.get_volume()
    v_.append(v/v0)
    print(' * V/V0',v_[-1],v,pressure)
 
 
fig, ax = plt.subplots() 
plt.ylabel(r'$Pressure$ ($GPa$)')
plt.xlabel(r'$V/V_0$')
plt.plot(v_,p,label=r'$IRFF-MPNN$', color='blue', 
         marker='o',markerfacecolor='none',
         markeredgewidth=1, 
         ms=5,alpha=0.8,
         linewidth=1, linestyle='-')
# plt.fill_between(X_plot[:, 0], y_gpr - y_std, y_gpr + y_std, color='darkorange',
#                  alpha=0.2)
plt.legend(loc='best',edgecolor='yellowgreen')
plt.savefig('pv.pdf') 
plt.close()
