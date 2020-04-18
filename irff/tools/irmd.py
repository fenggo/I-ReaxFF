#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argh
import argparse
from irff.irff import IRFF
from ase.io import read,write
import numpy as np
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase import units


def md(gen='poscar.gen',index=0,totstep=100):
    atoms      = read(gen,index=index)
    atoms.calc = IRFF(atoms=atoms,
                      libfile='ffield.json',
                      rcut=None,
                      nn=True)
    # dyn = BFGS(atoms)
    dyn = VelocityVerlet(atoms, 0.1 * units.fs)  # 5 fs time step.

    def printenergy(a=atoms):
        """Function to print the potential, kinetic and total energy"""
        natom = len(a)
        epot  = a.get_potential_energy()/natom
        ekin  = a.get_kinetic_energy()/natom
        T     = ekin / (1.5 * units.kB)
        try:
           assert T<=8000.0,'Temperature goes too high!'
        except:
           print('Temperature goes too high, stop at step %d.' %dyn.nsteps)
           dyn.max_steps = dyn.nsteps-1
        # print(a.get_forces())
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV' % (epot, ekin,T, epot + ekin))
        
    traj = Trajectory('md.traj', 'w', atoms)
    dyn = VelocityVerlet(atoms, 0.1 * units.fs)  # 5 fs time step.
    
    dyn.attach(printenergy,interval=1)
    dyn.attach(traj.write,interval=1)
    dyn.run(totstep)


if __name__ == '__main__':
   ''' use commond like ./irmd.py md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [md])
   argh.dispatch(parser)

   
