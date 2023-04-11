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


def LearningResult_angel(ffield='ffield.json',nn='T',gen='poscar.gen',traj='C1N1O2H30.traj'):
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

# LearningResult_angel(traj='swing.traj')


def LearningResultAngel(ffield='ffield.json',nn='T',gen='poscar.gen',traj='C2H4-1.traj'):
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
    # images_= []
    # for _ in range(0,50):
    eang_ = []
    
    for _,atoms in enumerate(images):
        # atoms = images[_]
        ir.calculate(atoms)
        Empnn.append(ir.E)
        Esiesta.append(atoms.get_potential_energy())
        atoms_ = atoms.copy()
        atoms_.set_initial_charges(charges=ir.q)
        calc = SinglePointCalculator(atoms_,energy=ir.E)
        atoms_.set_calculator(calc)
        traj_.write(atoms=atoms_)
        
        # print(ir.Eang)
        eang_.append(ir.Eang)
        
    traj_.close()
    fig, ax = plt.subplots() 

    plt.plot(eang_,label=r'$E_{Angle}$', color='blue', linewidth=2, linestyle='-.')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('Eang-%s.eps' %traj[:-4]) 
    # plt.show()
    plt.close()
    

if __name__ == '__main__':
   # LearningResultAngel(traj='C1N1O2H3-0.traj')
   LearningResult_angel(traj='C4H6N4O4-1.traj')
