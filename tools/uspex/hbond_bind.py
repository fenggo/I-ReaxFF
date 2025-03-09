#!/usr/bin/env python
import sys
import argparse
import copy
import numpy as np
from os import system
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol,Molecules,enlarge,moltoatoms
from irff.md.gulp import opt

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='Individuals.traj',type=str, help='trajectory file')
parser.add_argument('--s', default=0,type=int, help='strat index of crystal structure')
parser.add_argument('--e', default=-1,type=int, help='end index of crystal structure')
#parser.add_argument('--b', default=1,type=int, help='compute binding energy')
parser.add_argument('--p', default=100,type=int, help='step of optimization')
parser.add_argument('--n', default=8,type=int, help='ncpu')
args = parser.parse_args(sys.argv[1:])

def bind_energy(A,emol=None,step=20,moleclue_energy={}):
    m_  = Molecules(A,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,
                            "O-O":1.4,"others": 1.68},
                    check=True)
    nmol = len(m_)
    ir.calculate(A)
  
    A = opt(atoms=A,step=step,l=0,t=0.0000001,n=args.n, x=1,y=1,z=1)
    system('mv md.traj md_{:d}.traj'.format(i))
    eg = A.get_potential_energy()

    if emol is None:
        emol = 0.0
        for m in m_:
            if m.label not in moleclue_energy:
                m.cell = np.array([[15.0,0.0,0.0],[0.0,15.0,0.0],[0.0,0.0,15.0]])
                atoms = moltoatoms([m])
                atoms = opt(atoms=atoms,step=1000,l=0,t=0.0000001,n=args.n, x=1,y=1,z=1)
                moleclue_energy[m.label] = atoms.get_potential_energy()
                print('-  Molecular Energy {:10s}: {:f}'.format(m.label,
                                                moleclue_energy[m.label]))
            emol += moleclue_energy[m.label]
    eb = (emol-eg)
    # print('emol: ',emol,eg[0],eg[-1],'ebind: ',eb)
    return eb, eb/nmol,emol,moleclue_energy # eg[-1]

images  = Trajectory(args.g)
atoms   = images[0]
if args.e<0:
   args.e = len(images)-1
imags   = [i for i in range(args.s,args.e+1)]

ir = IRFF_NP(atoms=atoms,nn=True,libfile='ffield.json')
ir.calculate(atoms)

E,Ehb,D = [],[],[]
Eb      = []
eb,eb_per_mol,emol = 0.0, 0.0, 0.0
mole    = {}

with open('hbond.dat','w') as fd:
     print('# Crystal_id hbond_energy binding_energy eb_per_mol density',file=fd)

for i,s in enumerate(imags):
    atoms = images[s-1]
    # if args.b:
    #    atoms = opt(atoms=atoms,step=50,l=1,t=0.0000001,n=args.n, x=1,y=1,z=1)
    atoms = press_mol(atoms)
    x     = atoms.get_positions()
    m     = np.min(x,axis=0)
    x_    = x - m
    atoms.set_positions(x_)

    ir.calculate(atoms)
    Ehb.append(-ir.Ehb)
    
    eb,eb_per_mol,emol,mole = bind_energy(atoms,step=args.p,
                                          moleclue_energy=mole)
    Eb.append(eb)
    E.append(ir.E)
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    D.append(density)
    print('CY: {:6d} ehbond: {:8.4f}, ebind: {:8.4f},'
          ' emol: {:8.4f}, Density: {:9.6}'.format(s,
                      Ehb[-1],eb,emol,density) )
    with open('hbond.dat','a') as fd:
         print(s,Ehb[-1],eb,eb_per_mol,density,file=fd) 

plt.figure()   
plt.ylabel(r'$Density$ ($g/cm^3$)')
plt.xlabel(r'$-$ $HB$ $Energy$ ($eV$)')

# plt.subplot(2,1,1)
plt.scatter(Ehb,D,alpha=0.8,
            edgecolor='r', s=35,color='none',marker='o',
            label=r'$HB$ $Energy(4CL-20)$')

plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
plt.savefig('hbond.pdf',transparent=True)
plt.close()

