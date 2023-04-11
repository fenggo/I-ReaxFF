#!/usr/bin/env python
from ase.io import read
# from ase.units import kJ,GPa
from ase.eos import EquationOfState
from ase.optimize import BFGS,FIRE
from irff.md import opt
from irff.irff import IRFF
from ase.io.trajectory import TrajectoryWriter
import matplotlib.pyplot as plt


traj = TrajectoryWriter('pre_opt.traj',mode='w')

atoms  = read('sc.gen')
cell   = atoms.get_cell()
atoms_ = atoms.copy()
configs= []
GPa    = 1.60217662*1.0e2
s      = 1.0
i      = 0
ir     = IRFF(atoms=atoms_,libfile='ffield.json',nn=True)

v_,p   = [],[]
v0     = atoms.get_volume()

while s>0.59:
      print(' * lattice step:',i)
      f= s**(1.0/3.0)
      cell_ = cell.copy()
      cell_ = cell*f
      atoms_.set_cell(cell_)
      atoms_ = opt(atoms=atoms_,fmax=0.02,step=120,optimizer=FIRE)
      configs.append(atoms_)
      traj.write(atoms=atoms_)
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

      s= s*0.98
      i += 1

fig, ax = plt.subplots() 
plt.ylabel(r'$Pressure$ ($GPa$)')
plt.xlabel(r'$V/V_0$')
plt.plot(v_,p,label=r'$IRFF-MPNN$', color='blue', 
         marker='o',markerfacecolor='none',
         markeredgewidth=1, 
         ms=5,alpha=0.8,
         linewidth=1, linestyle='-')

plt.legend(loc='best',edgecolor='yellowgreen')
plt.savefig('pv.pdf') 
plt.close()
