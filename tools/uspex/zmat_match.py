#!/usr/bin/env python
from ase.io import read
from ase import Atoms
from irff.AtomDance import AtomDance

def match_zat(gen='b_0.gen',match='a_0.gen',first_atom=0,first_atom_=0):
    atoms  = read(gen,index=-1)
    ad     = AtomDance(atoms=atoms,rmax=1.1,
                     rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,"O-O":1.4,"others": 1.68},
                     FirstAtom=first_atom)
    zmat   = ad.InitZmat
    ind    = ad.zmat_id
    symbol = atoms.get_chemical_symbols()
    
    atoms_ = read(match,index=-1)
    ad_    = AtomDance(atoms=atoms_,rmax=1.1,
                       rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,"O-O":1.4,"others": 1.68},
                       FirstAtom=first_atom_)
    zmat_  = ad_.InitZmat
    ind_   = ad_.zmat_id
    symbol_= atoms_.get_chemical_symbols()
    ad.write_zmat(zmat,uspex=True)
    ad.close()
    ad_.close()

    #first_atom = 25
    neighbors_  = ad_.neighbors
    neighbors   = ad.neighbors
    zid     = []
    zid     = []

    for i,i_ in enumerate(ind_):
        if symbol_[i_]==symbol[ind[i]]:
           zid.append(i_)
           nei_ = neighbors_[i_]
        else:
           print('an error occured!')
           break
    # print('atomic order: ',zid)
    return ind,ind_,atoms,atoms_

# gens = ['b_0.gen','b_1.gen','b_2.gen','b_3.gen']
# mats = ['a_0.gen','a_1.gen','a_2.gen','a_3.gen']

gen = 'md_eps.traj' 
mat = 'md.traj'

first_atom  = [0,1,2,3]
first_atom_ = [123,135,129,141]
inds,inds_,image,image_  = [],[],[],[]


ind,ind_,atoms,atoms_ = match_zat(gen=gen,match=mat,
                                  first_atom=first_atom,first_atom_=first_atom_)
inds.append(ind)
inds_.append(ind_)
image.append(atoms)
image_.append(atoms_)
print(ind,'\n',ind_)


natom  = len(atoms)
cell   = atoms.get_cell()

positions_ = []
elems_     = []

for n,ind in enumerate(inds):
    ind_      = inds_[n]
    atoms     = image[n]
    atoms_    = image_[n]
    positions = [None for i in range(natom)]
  
    for i,i_ in zip(ind,ind_):
        positions[i_] = atoms.positions[i]
    elem = atoms_.get_chemical_symbols()

    positions_.extend(positions)
    elems_.extend(elem)

A = Atoms(elems_, positions_)
A.set_cell(cell)
A.set_pbc([True,True,True])

A.write('POSCAR.mat')
