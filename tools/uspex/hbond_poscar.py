#!/usr/bin/env python
from os import getcwd, listdir
from os.path import isdir
import sys
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from ase.io import read 
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol,Molecules,enlarge # , moltoatoms
from irff.md.gulp import opt
from irff.deb.compare_energies import deb_gulp_energy

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='Individuals.traj',type=str, help='trajectory file')
parser.add_argument('--o', default=0,type=int, help='whether optimize the crystal structure')
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
    nmol = {}
    for m in m_:
        if m.label in nmol:
           nmol[m.label] += 1
        else:
           nmol[m.label]  = 1
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
    # print('NM: ',nmol,'\nDhb',ehb[0],ehb[-1])
    return nmol,-ehb[0], ev_, ec_ ,e_ 


gens  = listdir(getcwd())
gens  = ['{:s}/POSCAR.{:s}_opt'.format(gen,gen) 
         for gen in gens if isdir(gen) and not gen.startswith('.')]
atoms = read(gens[0])


E,Ehb,D = {},{},{}
I,Eb    = {},{}
eb,eb_per_mol,emol = 0.0, 0.0, 0.0

with open('hbond.dat','w') as fd:
     print('# Crystal_id hbond_energy binding_energy density',file=fd)

for i,gen in enumerate(gens):
    # print(gen)
    s_ = gen.split('.')[1]
    s  = s_.split('_')[0]

    if s in ['b1792']:
       continue
    
    atoms = read(gen)
    if args.o:
       atoms = opt(atoms=atoms,step=args.p,l=1,t=0.0000001,n=args.n, x=1,y=1,z=1)

    atoms = press_mol(atoms)
    x     = atoms.get_positions()
    m     = np.min(x,axis=0)
    x_    = x - m
    atoms.set_positions(x_)
    mols,ehb,ev,ec,e = bind_energy(atoms)
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129

    nmol = 0
    for mol in mols:
        nmol += mols[mol]
    
    if 'C2N4H4O4' in mols and 'C6N12O12H6' in mols:
       label = '{:d}:{:d}'.format(mols['C6N12O12H6'],mols['C2N4H4O4'])
       if mols['C6N12O12H6'] == mols['C2N4H4O4']:
          label = '1:1'
    elif 'O4N4C2H4' in mols:
       label = 'FOX-7'
    elif 'C4H8N8O8' in mols and 'C6N12O12H6' in mols:
       label = '2CL-20/1HMX(exp.)'
    elif 'C6H6N12O12' in mols:
       label = 'CL-20'
    elif 'C6N12O12H6' in mols:
       label = 'CL-20\\HMX'
    elif 'C4H8N8O8' in mols:
       label = 'HMX'
    else:
       label = 'Others'
    # print(mols,label)

    if label in I:
       I[label].append(s)
       Ehb[label].append(ehb/nmol)
       Eb[label].append(e/nmol)
       D[label].append(density)
    else:
       I[label]   = [s]
       Ehb[label] = [ehb/nmol]
       Eb[label]  = [e/nmol]
       D[label]   = [density]
    # eb = e-ehb
    print('CY: {:6s}, NM: {:2d} ehbond: {:8.4f}, evdw: {:8.4f}, ebind: {:8.4f},'
          ' Density: {:9.6}'.format(s,nmol,ehb,ev,e,density))
    with open('hbond.dat','a') as fd:
         print(s,ehb,e,e/nmol,density,file=fd) 

plt.figure()   
plt.ylabel(r'$Density$ ($g/cm^3$)')
plt.xlabel(r'$- Intramolecular$ $HB$ $Energy$ ($eV$)')

markers = {'1:3':'^','1:4':'v','1:2':'>','1:1':'s','1:5':'<','1:6':'p','2:2':'D',
           '2:1':'d','3:1':'P','4:1':'h', 'CL-20\\HMX':'*',
           '2CL-20/1HMX(exp.)':'8','FOX-7':'P','Others':'X','CL-20':'X',
            'HMX':'P'}

colors = {'1:3':'b','1:4':'#7e9bb7','1:2':'g','1:1':'#61dcb8','1:5':'#27b2af','1:6':'#b83945','2:2':'#ffde18',
           '2:1':'#035f37','3:1':'#edb31e','4:1':'#377483','FOX-7':'#2ec0c2', 'CL-20\\HMX':'r',
           '2CL-20/1HMX(exp.)':'#4f845c','Others':'#e3e457','CL-20':'y',
           'HMX':'#9fba95'} # #008040 翡翠绿 

# plt.subplot(2,1,1)
left = ['h1218','g1942','c641','g909']
right = ['f240','g1446','g1255']
hide = []
labels = ['4:1','3:1','2:1','1:1','1:2','1:3','1:4','1:5','1:6']
for l in I:
    if l not in labels:
       labels.append(l)
for label in labels:
    ehb = Ehb[label]
    d   = D[label]
    if not d:
       continue
    mark = markers[label]
    # if label==r'$\beta-HMX(exp.)$':
    #    continue
    if label in labels:
       if label in ['HMX','FOX-7','CL-20']:
          plt.scatter(ehb,d,alpha=0.8,
                      color=colors[label], s=35,marker=mark,
                      label=label)
       else:
          plt.scatter(ehb,d,alpha=0.8,
                edgecolor=colors[label], s=35,marker=mark,
                color='none',
                label=label)
    else:
       plt.scatter(ehb,d,alpha=0.8,
                edgecolor=colors[label], s=35,marker=mark,
                color='none')
       
    for i,lb in enumerate(I[label]):
        print(label,lb)
        if lb == 'fox7':
           lb_ = r'$\alpha-FOX-7$'
        elif lb == 'cl20':
           lb_ = r'$\varepsilon-CL-20$'
        elif lb == 'exp24':
           lb_ = '2CL-20\\1HMX(Bolton, Ref.9)'
        elif lb == 'hmx':
           lb_ = r'$\beta-HMX$'
        else:
           lb_ = lb.upper()
        if lb in hide:
           continue
        if lb in left:
           plt.text(ehb[i]-0.01,d[i]-0.001,lb_,ha='center',color='k',
                 fontsize=5)
        elif lb in right:
           plt.text(ehb[i]+0.005,d[i]-0.001,lb_,ha='center',color='k',
                 fontsize=5)
        else:
           plt.text(ehb[i],d[i]+0.002,lb_,ha='center',color='k',
                 fontsize=5)

# x = np.linspace(0.59,0.77)
# y = x*0.67 + 1.41
# plt.plot(x,y,color='k',linestyle='-.')

plt.legend(loc='best',edgecolor='yellowgreen',ncol=3,fontsize=9) # loc = lower left upper right best
plt.savefig('hbond.svg',transparent=True) 
plt.close()
