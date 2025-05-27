#!/usr/bin/env python
from os import getcwd,chdir,listdir,system
import numpy as np
from tqdm import tqdm
from ase.io.trajectory import Trajectory
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
# from irff.deb.compare_energies import deb_gulp_energy
from irff.md.gulp import write_gulp_in,get_reax_energy
# from irff.molecule import press_mol

cdir    = getcwd()
files   = listdir(cdir)
poscars = [gen for gen in files if gen.endswith('.gen') ]
images  = []
ncpu    = 1

f = open('train.csv','w')
print(',material_id,energy,density,hydrogen_bond_energy,cif,volume',file=f)

for i in tqdm(range(len(poscars))):
    gen   = poscars[i]
    gen_  = gen.split('.')[0]
    atoms = read(gen)
    images.append(atoms.copy())

    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',
                      lib='reaxff_nn.lib')
    if ncpu==1:
       system('gulp<inp-gulp>out')
    else:
       system('mpirun -n {:d} gulp<inp-gulp>out'.format(ncpu))

    (e,eb_,el_,eo_,eu_,ea_,ep_,
     etc_,et_,ef_,ev_,ehb,ecl_,esl_)= get_reax_energy(fo='out')

    structure = AseAtomsAdaptor.get_structure(atoms)
    structure.to(filename="POSCAR")

    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    # print('loading structures {:d}/{:d} ...\r'.format(i,n_images),end='\r')
    structure = AseAtomsAdaptor.get_structure(atoms)
    structure.to(filename="{:s}.cif".format(gen_))
    
    print('{:d},{:s},{:f},{:f},{:f},\"'.format(i,gen_,
                       e,density,ehb),end='',file=f)
    with open("{:s}.cif".format(gen_),'r') as fc:
         for line in fc.readlines():
             print(line.strip(),file=f)
    print('\",{:f}'.format(volume),file=f)

f.close()
