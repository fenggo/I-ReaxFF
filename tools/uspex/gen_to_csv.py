#!/usr/bin/env python
from os import getcwd,chdir,listdir,system
import numpy as np
from ase.io.trajectory import Trajectory
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
from irff.deb.compare_energies import deb_gulp_energy
# from irff.molecule import press_mol

cdir    = getcwd()
files   = listdir(cdir)
poscars = [gen for gen in files if gen.endswith('.gen') ]
images  = []

for gen in poscars:
    gen_ = gen.split('.')[0]
    atoms = read(gen)
    images.append(atoms.copy())

(e,ebond_,eunder_,eover_,elone_,eang_,etcon_,epen_,
    etor_,efcon_,ev,ehb,ec) = deb_gulp_energy(images, ffield='reaxff_nn')

f = open('train.csv','w')
print(',material_id,energy,density,hydrogen_bond_energy,cif,volume')

for i,atoms in enumerate(images):
    gen_ = poscars[i].split('.')[0]
    structure = AseAtomsAdaptor.get_structure(atoms)
    structure.to(filename="POSCAR")

    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    # if density_>density:
    structure = AseAtomsAdaptor.get_structure(atoms)
    structure.to(filename="{:s}.cif".format(gen_))

    print('{:d}, {:s}, {:f}, {:f}, {:f}\"#'.format(i,gen_,
                       e[i],density,ehb[i]),end=' ',file=f)
    with open("{:s}.cif".format(gen_),'w') as fcif:
         for line in fcif:
             print(line.strip(),file=f)
    print('\",{:f}'.format(volume))
    