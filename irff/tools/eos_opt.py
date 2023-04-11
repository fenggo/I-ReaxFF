#!/usr/bin/env python
from ase.io import read
from ase.units import kJ
from ase.eos import EquationOfState
from ase.optimize import BFGS,QuasiNewton,FIRE
from irff.md import opt,bhopt
from ase.io.trajectory import TrajectoryWriter


# configs = read('Ag.traj@0:5')  # read 5 configurations
# # Extract volumes and energies:
# volumes = [ag.get_volume() for ag in configs]
# energies = [ag.get_potential_energy() for ag in configs]
# eos = EquationOfState(volumes, energies)
# v0, e0, B = eos.fit()
# print(B / kJ * 1.0e24, 'GPa')365222
# eos.plot('nm-eos.png')
traj = TrajectoryWriter('eos_opt.traj',mode='w')

atoms  = read('NM.gen')
cell   = atoms.get_cell()
atoms_ = atoms.copy()
# atoms_ = opt(atoms=atoms_,fmax=0.01,step=200)
configs = []
s = 1.1
i = 0
while s>0.70:
      print(' * lattice step:',i)
      f= s**(1.0/3.0)
      cell_ = cell.copy()
      cell_ = cell*f
      atoms_.set_cell(cell_)
      # if i>9 and i<15:
      atoms_ = opt(atoms=atoms_,fmax=0.02,step=120,optimizer=FIRE)
      # else:
      #   atoms_ = bhopt(atoms=atoms_,fmax=0.02,step=120,temperature=30,optimizer=BFGS)

      configs.append(atoms_)
      traj.write(atoms=atoms_)
      s= s*0.98
      i += 1

