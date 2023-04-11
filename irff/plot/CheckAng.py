#!/usr/bin/env python
import numpy as np
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units
from ase.visualize import view
from irff.irff_np import IRFF_NP
from irff.irff import IRFF
import matplotlib.pyplot as plt
from irff.AtomOP import AtomOP


def check_angel(ffield='ffield.json',nn='T',gen='poscar.gen',traj='C1N1O2H30.traj'):
    images = Trajectory(traj)
    traj_  = TrajectoryWriter(traj[:-5]+'_.traj',mode='w')
    atoms = images[0]
    nn_=True if nn=='T'  else False
    
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 rcut=None,
                 nn=nn_)
    natom = ir.natom
    
    Empnn,Esiesta = [],[]
    eb,eb_ = [],[]
    eang_ = []
    theta0= []
    l = len(images)
    for _ in range(l-10,l):
        atoms = images[_]
        ir.calculate(atoms)
        Empnn.append(ir.E)
        Esiesta.append(atoms.get_potential_energy())
        atoms_ = atoms.copy()
        atoms_.set_initial_charges(charges=ir.q)
        calc = SinglePointCalculator(atoms_,energy=ir.E)
        atoms_.set_calculator(calc)
        traj_.write(atoms=atoms_)
        eang_.append(ir.Eang)
        # print(ir.Eang)
        # print(ir.SBO3)
    for a,ang in enumerate(ir.angi):
        th0 = ir.thet0[a] #*180.0/3.14159
        th  = ir.theta[a] # *180.0/3.14159
        i,j,k = ir.angi[a][0],ir.angj[a][0],ir.angk[a][0]
        print('%s-%s-%s'%(ir.atom_name[i],ir.atom_name[j],ir.atom_name[k]),
               'thet0: %8.6f' %th0,
               'thet: %f8.6' %th,
               'sbo3 %8.6f:' %ir.SBO3[a],
               'Dpi: %8.6f' %ir.Dpi[ir.angj[a][0]],
               'pbo: %8.6f' %ir.PBO[ir.angj[a][0]],
               'nlp: %8.6f' %ir.nlp[ir.angj[a][0]],
               'Dang: %8.6f' %ir.Dang[ir.angj[a][0]],
               # 'eang:',ir.eang[a],'expang:',ir.expang[a],
             )


if __name__ == '__main__':
   check_angel(traj='C4H6N4O4-3.traj')
