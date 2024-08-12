#!/usr/bin/env python
from ase.io.trajectory import Trajectory,TrajectoryWriter
from irff.reax_force import ReaxFF_nn_force
from irff.irff_np import IRFF_NP
from irff.irff_autograd import IRFF
from irff.data.ColData import ColData
import matplotlib.pyplot as plt
from irff.md.gulp import get_gulp_forces


ir = ReaxFF_nn_force(dataset={'md':'md.traj'},
                     screen=True,
                     libfile='ffield.json')
ir.forward()

print('\n---- reax_nn_force ----\n')
for s in ir.bop:
    print('\n evdw \n',ir.evdw[s])
    print('\n ehb \n',ir.ehb[s])  
    print(ir.Evdw[s].shape)

print('\n---- irff ----\n')
images = Trajectory('md.traj')
ir_ = IRFF(atoms=images[0],libfile='ffield.json',nn=True)
ir2 = IRFF_NP(atoms=images[0],libfile='ffield.json',nn=True)

forces = images[0].get_forces()


for i,img in enumerate(images):
    ir_.calculate(atoms=img)
    ir2.calculate(atoms=img)
    print('--     IR     --      RForce     --     IRNP     --' )
    print('E      : ',ir_.E,ir.E[s][i].item(),ir2.E)
    print('Eover  : ',ir_.Eover.item(),ir.eover[s][i].item(),ir2.Eover)
    print('Eunder : ',ir_.Eunder.item(),ir.eunder[s][i].item(),ir2.Eunder)
    print('Elone  : ',ir_.Elone.item(),ir.elone[s][i].item(),ir2.Elone)
    print('Eang   : ',ir_.Eang.item(),ir.eang[s][i].item(),ir2.Eang)
    print('Epen   : ',ir_.Epen.item(),ir.epen[s][i].item(),ir2.Epen)
    print('Etor   : ',ir_.Etor.item(),ir.etor[s][i].item(),ir2.Etor)
    print('Efcon  : ',ir_.Efcon.item(),ir.efcon[s][i].item(),ir2.Efcon)
    print('Evdw   : ',ir_.Evdw.item(),ir.evdw[s][i].item(),ir2.Evdw)
    print('Ehb    : ',ir_.Ehb.item(),ir.ehb[s][i].item(),ir2.Ehb)
    # print('\n IR-dpi \n',ir2.Dpil)
 
 
# print('\n----  forces  ----\n')
# ir_.calculate(atoms=images[0])
# for i in range(ir_.natom):
#     print(ir_.results['forces'][i],'----' ,ir.force[s][0][i].detach().numpy(),
#              '----',forces[i])

# get_gulp_forces(images)
# print('\n lammps: \n')
# images = Trajectory('md.traj')
# atoms  = images[0]
# forces = atoms.get_forces()
# for f in forces:
#     print(f)

