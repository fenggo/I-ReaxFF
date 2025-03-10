#!/usr/bin/env python
import sys
import argparse
import copy
import numpy as np
from ase.io.trajectory import Trajectory
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol,Molecules,enlarge # , moltoatoms
# from irff.md.gulp import opt
from irff.deb.compare_energies import deb_gulp_energy

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='Individuals.traj',type=str, help='trajectory file')
parser.add_argument('--s', default=0,type=int, help='strat index of crystal structure')
parser.add_argument('--e', default=-1,type=int, help='end index of crystal structure')
parser.add_argument('--p', default=200,type=int, help='step of optimization')
parser.add_argument('--n', default=8,type=int, help='ncpu')
args = parser.parse_args(sys.argv[1:])

def bind_energy(A,species=None):
    ff = [1.0,5.0] #,1.9 ,2.0,2.5,3.0,3.5,4.0
    cell = A.get_cell()
    e   = []
    ehb = []
    ev  = []
    ec  = []
    images = []
    m_  = Molecules(A,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,
                            "O-O":1.4,"others": 1.68},
                    species=species,check=True)
    nmol = len(m_)
    for f in ff:
        m = copy.deepcopy(m_)
        m,A = enlarge(m,cell=cell,fac=f,supercell=[1,1,1])
        images.append(A)
    (e,ebond_,eunder_,eover_,elone_,eang_,etcon_,epen_,
    etor_,efcon_,ev,ehb,ec) = deb_gulp_energy(images, ffield='reaxff_nn')
    ehb_ = ehb[-1]-ehb[0]
    ev_  = ev[-1]-ev[0]
    ec_  = ec[-1]-ec[0]
    e_   = e[-1]-e[0]
    # print('NM: ',nmol,'\nDhb',ehb,'\nDv: ',ev,'\nDc: ',ec)
    return nmol,ehb_, ev_, ec_ ,e_ 

images  = Trajectory(args.g)
atoms   = images[0]
if args.e<0:
   args.e = len(images)-1
imags   = [i for i in range(args.s,args.e+1)]


E,Ehb,D = [],[],[]
Eb      = []
eb,eb_per_mol,emol = 0.0, 0.0, 0.0

with open('hbond.dat','w') as fd:
     print('# Crystal_id hbond_energy binding_energy eb_per_mol density',file=fd)

for i,s in enumerate(imags):
    atoms = images[s-1]
    atoms = press_mol(atoms)
    x     = atoms.get_positions()
    m     = np.min(x,axis=0)
    x_    = x - m
    atoms.set_positions(x_)
    nmol,ehb,ev,ec,e = bind_energy(atoms)

    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    D.append(density)
    eb = e-ehb
    print('CY: {:6d}, NM: {:2d} ehbond: {:8.4f}, evdw: {:8.4f}, ebind: {:8.4f},'
          ' Density: {:9.6}'.format(s,nmol,ehb,ev,eb,density))
    with open('hbond.dat','a') as fd:
         print(s,ehb,eb,density,file=fd) 

