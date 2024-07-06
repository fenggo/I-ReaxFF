#!/usr/bin/env python
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
from ase.io import read,write
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import numpy as np
from irff.reax_nn import ReaxFF_nn
from irff.irff_np import IRFF_NP
from irff.irff_autograd import IRFF
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
import matplotlib.pyplot as plt
from irff.md.gulp import get_gulp_forces


ir = ReaxFF_nn(dataset={'md':'md.traj'},
               MessageFunction=3,
               mf_layer=[9,2],
               be_layer=[9,1],
               libfile='ffield.json')
ir.initialize()
ir.session()  

f = {}

print('\n---- reax_nn ----\n')
for s in ir.bop:
    E,ebond,elone,eover,eunder,eang,etor,etcon,epen,evdw,ecoul = ir.sess.run([ir.E[s],ir.ebond[s],
                                          ir.elone[s],ir.eover[s],ir.eunder[s],
                                          ir.eang[s],ir.etor[s],ir.etcon[s],ir.epen[s],
                                          ir.evdw[s],ir.ecoul[s]],
                                                                   feed_dict=ir.feed_dict)
    bo = ir.sess.run(ir.bo0[s],feed_dict=ir.feed_dict)
    # print('\n bo \n',np.squeeze(bo,axis=2))
    # bop = ir.sess.run(ir.bop[s],feed_dict=ir.feed_dict)
    # print('\n bop \n',np.squeeze(bop,axis=2))
    # d= ir.sess.run(ir.Deltap[s],feed_dict=ir.feed_dict)
    # print('\n Deltap \n',np.squeeze(d,axis=1))
    print(E,ebond,eang,etor,evdw,ecoul,etor,etcon,epen)

    ir.get_forces(s)
    f = ir.sess.run(ir.forces[s],feed_dict=ir.feed_dict)
    # E = ir.sess.run(ir.EBD[s],feed_dict=ir.feed_dict)
    # print(E)

print('\n---- irff ----\n')
images = Trajectory('md.traj')
ir_ = IRFF(atoms=images[0],libfile='ffield.json',nn=True)
ir2 = IRFF_NP(atoms=images[0],libfile='ffield.json',nn=True)
ir2.calculate(atoms=images[0])
print(ir2.E)
print(ir2.Ebond)
print(ir2.Eang)
print(ir2.Etor)
print(ir2.Evdw)
print(ir2.Ecoul)
print(ir2.Etor)
print(ir2.Etcon)
print(ir2.Epen)
# print('\n bo \n',ir2.bo0)
# print('\n bop \n',ir2.bop)
# print('\n Deltap \n',ir2.Deltap)

forces = images[0].get_forces()


for i,img in enumerate(images):
    ir_.calculate(atoms=img)
    ir2.calculate(atoms=img)
    print('--     IR     --      RForce     --     IRNP     --' )
    print(ir_.E,E[i],ir2.E)
    print(ir_.Elone.item(),elone[i],ir2.Elone)
    print(ir_.Eover.item(),eover[i],ir2.Eover)
    print(ir_.Eunder.item(),eunder[i],ir2.Eunder)
    print(ir_.Eang.item(),eang[i],ir2.Eang)
    print(ir_.Etor.item(),etor[i],ir2.Etor)
    # print('\n IR-dpi \n',ir2.Dpil)
 
 
print('\n----  forces  ----\n')
ir_.calculate(atoms=images[0])
for i in range(ir_.natom):
    print(ir_.results['forces'][i],'----' ,f[0][i],'----',forces[i])

# get_gulp_forces(images)
# print('\n lammps: \n')
# images = Trajectory('md.traj')
# atoms  = images[0]
# forces = atoms.get_forces()
# for f in forces:
#     print(f)
    
