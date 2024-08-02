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
from irff.deb.compare_energies import deb_gulp_energy

traj= 'md.traj'

ir = ReaxFF_nn(dataset={'md':traj},
               MessageFunction=3,
               mf_layer=[9,2],
               be_layer=[9,1],
               screen=False,
               libfile='ffield.json')
ir.initialize()
ir.session()  

f = {}

print('\n---- reax_nn ----\n')
for s in ir.bop:
    E,ebond,elone,eover,eunder,eang,etor,etcon,epen,efcon,evdw,ehbi,ecoul = ir.sess.run([ir.E[s],
                                          ir.ebond[s],
                                          ir.elone[s],ir.eover[s],ir.eunder[s],
                                          ir.eang[s],ir.etor[s],ir.etcon[s],ir.epen[s],
                                          ir.efcon[s],
                                          ir.evdw[s],ir.ehb[s],
                                          ir.ecoul[s]],feed_dict=ir.feed_dict)

    bo = ir.sess.run(ir.bo0[s],feed_dict=ir.feed_dict)
    # print('\n bo \n',np.squeeze(bo,axis=2))
    # bop = ir.sess.run(ir.bop[s],feed_dict=ir.feed_dict)
    # print('\n bop \n',np.squeeze(bop,axis=2))
    # d= ir.sess.run(ir.Deltap[s],feed_dict=ir.feed_dict)
    # print('\n Deltap \n',np.squeeze(d,axis=1))
    
    # (w,cosw,cos2w,sijk,sjkl,
    #  f10,f11) = ir.sess.run([ir.w[s]['C-C-C-C'],ir.cos_w[s]['C-C-C-C'],ir.cos2w[s]['C-C-C-C'],
    #                                 ir.s_ijk[s]['C-C-C-C'],ir.s_jkl[s]['C-C-C-C'],
    #                                 ir.f_10[s]['C-C-C-C'],ir.f_11[s]['C-C-C-C']],
    #                         feed_dict=ir.feed_dict)
    # cijk = ir.sess.run(ir.cijk, feed_dict=ir.feed_dict)
    # print('\n w \n',w)
    # print('\n cosw \n',cosw)
    # print('\n cos2w \n',cos2w)
    # print('\n sijk \n',sijk)
    # print('\n sjkl \n',sjkl)
    # print('\n f10 \n',f10)
    # print('\n f11 \n',f11)
    # for v in cosw:
    #     print(v)

    ir.get_forces(s)
    f = ir.sess.run(ir.forces[s],feed_dict=ir.feed_dict)
    # E = ir.sess.run(ir.EBD[s],feed_dict=ir.feed_dict)
    # print(E)

print('\n---- irff ----\n')
images = Trajectory(traj)
ir_ = IRFF(atoms=images[-1],libfile='ffield.json',nn=True)
ir2 = IRFF_NP(atoms=images[-1],libfile='ffield.json',nn=True)
ir2.calculate(atoms=images[-1])

# print('\n bo \n',ir2.bo0)
# print('\n bop \n',ir2.bop)
# print('\n Deltap \n',ir2.Deltap)
forces = images[-1].get_forces()
(e_,ebond_,eunder_,eover_,elone_,eang_,etcon_,epen_,
 etor_,efcon_,evdw_,ehb_,ecoul_) = deb_gulp_energy(images, ffield='reaxff_nn')

for i,img in enumerate(images):
    ir_.calculate(atoms=img)
    ir2.calculate(atoms=img)
    print('--     IR     --      RForce     --     GULP     --' )
    print('etol   : ',ir_.E,E[i],e_[i])
    print('ebond  : ',ir_.Ebond.item(),ebond[i],ebond_[i])
    print('elone  : ',ir_.Elone.item(),elone[i],elone_[i])
    print('eover  : ',ir_.Eover.item(),eover[i],eover_[i])
    print('eunder : ',ir_.Eunder.item(),eunder[i],eunder_[i])
    print('eang   : ',ir_.Eang.item(),eang[i],eang_[i])
    print('etcon  : ',ir_.Etcon.item(),etcon[i],etcon_[i])  
    print('epen   : ',ir_.Epen.item(),epen[i],epen_[i])  
    print('etor   : ',ir_.Etor.item(),etor[i],etor_[i])
    print('efcon  : ',ir_.Efcon.item(),efcon[i],efcon_[i])  
    print('evdw   : ',ir_.Evdw.item(),evdw[i],evdw_[i])
    print('ecoul  : ',ir_.Ecoul.item(),ecoul[i],ecoul_[i])
    # print('ehb    : ',ir_.Ehb.item(),ehb,ehb_[i])  
 
 
 
print('\n----  forces  ----\n')
ir_.calculate(atoms=images[-1])
for i in range(ir_.natom):
    print(ir_.results['forces'][i],'----' ,f[-1][i],'----',forces[i])

# get_gulp_forces(images)
# print('\n gulp: \n')
# images = Trajectory('gulp_force.traj')
# atoms  = images[0]
# forces = atoms.get_forces()
# for f in forces:
#     print(f)
    
