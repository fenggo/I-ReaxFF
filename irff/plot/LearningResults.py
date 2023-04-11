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


def learning_result(ffield='ffield.json',nn='T',traj='C2H4.traj'):
    images = Trajectory(traj)
    traj_  = TrajectoryWriter(traj[:-5]+'_.traj',mode='w')
    atoms = images[0]
    nn_=True if nn=='T'  else False
    
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 nn=nn_)
    natom = ir.natom
    
    Empnn,Esiesta = [],[]
    eb,eb_ = [],[]
    # images_= []
    # for _ in range(10,11):
    for atoms in images:
        ir.calculate(atoms)
        Empnn.append(ir.E)
        Esiesta.append(atoms.get_potential_energy())
        atoms_ = atoms.copy()
        atoms_.set_initial_charges(charges=ir.q)
        calc = SinglePointCalculator(atoms_,energy=ir.E)
        atoms_.set_calculator(calc)
        traj_.write(atoms=atoms_)

    traj_.close()
    fig, ax = plt.subplots() 

    plt.plot(Empnn,label=r'$E_{MPNN}$', color='blue', linewidth=2, linestyle='-.')
    plt.plot(Esiesta,label=r'$E_{SIESTA}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    # plt.savefig('result-%s.pdf' %traj[:-4]) 
    plt.show()
    plt.close()
    

# learning_result(traj='nmr.traj')

