#!/usr/bin/env python
import sys
import argparse
import copy
import numpy as np
from os import system
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol,Molecules,enlarge
from irff.md.gulp import opt

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='md.traj',type=str, help='trajectory file')
parser.add_argument('--s', default=0,type=int, help='strat index of crystal structure')
parser.add_argument('--e', default=-1,type=int, help='end index of crystal structure')
parser.add_argument('--b', default=0,type=int, help='compute binding energy')
parser.add_argument('--n', default=8,type=int, help='ncpu')
args = parser.parse_args(sys.argv[1:])

def bind_energy(A,emol=None):
    ff = [1.0,5.0] #,1.9 ,2.0,2.5,3.0,3.5,4.0
    cell = A.get_cell()
    e  = []
    eg = []
    m_  = Molecules(A,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,
                            "O-O":1.4,"others": 1.68},
                    check=True)
    nmol = len(m_)
    for i,f in enumerate(ff):
        m = copy.deepcopy(m_)
        _,A = enlarge(m,cell=cell,fac=f,supercell=[1,1,1])
        ir.calculate(A)
        e.append(ir.E)        
        if i==0 or emol is None:
            A = opt(atoms=A,step=500,l=0,t=0.0000001,n=args.n, x=1,y=1,z=1)
            system('mv md.traj md_{:d}.traj'.format(i))
            e_ = A.get_potential_energy()
        else:
            e_ = emol
        eg.append(e_)
    eb = (eg[-1]-eg[0])
    return eb, eb/nmol, eg[-1]

images  = Trajectory('Individuals.traj')
atoms   = images[0]
if args.e<0:
   args.e = len(images)
imags   = [i for i in range(args.s,args.e+1)]

ir = IRFF_NP(atoms=atoms,nn=True,libfile='ffield.json')
ir.calculate(atoms)

E,Ehb,D = [],[],[]
Eb      = []
eb,eb_per_mol,emol = 0.0, 0.0, 0.0

with open('hbond.dat','w') as fd:
     print('# Crystal_id hbond_energy binding_energy eb_per_mol density',file=fd)

for i,s in enumerate(imags):
    atoms = images[s-1]
    if args.b:
       atoms = opt(atoms=atoms,step=1000,l=1,t=0.0000001,n=args.n, x=1,y=1,z=1)
    atoms = press_mol(atoms)
    x     = atoms.get_positions()
    m     = np.min(x,axis=0)
    x_    = x - m
    atoms.set_positions(x_)

    ir.calculate(atoms)
    Ehb.append(-ir.Ehb)
    if args.b:
       if i==0:
          eb,eb_per_mol,emol = bind_energy(atoms)
       else:
          eb,eb_per_mol,emol = bind_energy(atoms,emol=emol)
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
plt.xlabel(r'$-1$ $HB$ $Energy$ ($eV$)')

# plt.subplot(2,1,1)
plt.scatter(Ehb,D,alpha=0.8,
            edgecolor='r', s=35,color='none',marker='o',
            label=r'$Total$ $HB$ $Energy$')

plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
plt.savefig('hbond.pdf',transparent=True)
plt.close()

