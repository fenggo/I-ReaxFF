#!/usr/bin/env python
from ase.io.trajectory import Trajectory,TrajectoryWriter
from irff.reaxff_torch import ReaxFF_nn as ReaxFF_nn_torce
from irff.reax_nn import ReaxFF_nn
from irff.irff_np import IRFF_NP
from irff.irff_autograd import IRFF
from irff.data.ColData import ColData
import matplotlib.pyplot as plt
from irff.md.gulp import get_gulp_forces


ir = ReaxFF_nn_torce(dataset={'md':'md.traj'},
               weight_energy={'others':1.0},
               weight_force={'md':1},
               cons=['acut'],
               libfile='ffield.json',
               screen=True,
               lambda_bd=100.0,
               lambda_pi=0.0,
               lambda_reg=0.001,
               lambda_ang=0.0 )
ir.forward('md')

ir1 = ReaxFF_nn(dataset={'md':'md.traj'},
               weight_force={'md':1},
               MessageFunction=3,
               mf_layer=[9,2],
               be_layer=[9,1],
               screen=True,
               libfile='ffield.json')
ir1.initialize()
ir1.session()  

print('\n---- reaxff_nn ----\n')
for s in ir1.bop:
    E,ebond,elone,eover,eunder,eang,etor,etcon,epen,efcon,evdw,ehb,ecoul = ir1.sess.run([ir1.E[s],
                                          ir1.ebond[s],
                                          ir1.elone[s],ir1.eover[s],ir1.eunder[s],
                                          ir1.eang[s],ir1.etor[s],ir1.etcon[s],ir1.epen[s],
                                          ir1.efcon[s],
                                          ir1.evdw[s],ir1.ehb[s],
                                          ir1.ecoul[s]],feed_dict=ir1.feed_dict)

    bo = ir1.sess.run(ir1.bo0[s],feed_dict=ir1.feed_dict)

    ir1.get_forces(s)
    f = ir1.sess.run(ir1.forces[s],feed_dict=ir1.feed_dict)
    Ecoul = ir1.sess.run(ir1.Ecoul[s],feed_dict=ir1.feed_dict)


print('\n---- reaxff_nn_torce ----\n')

print('\n evdw \n',ir1.evdw[s])
print('\n ehb \n',ir1.ehb[s])  
# print(ir1.Evdw[s].shape)
bo0=ir.bo0[s].detach().numpy()
print(bo0.shape,bo.shape)
# for i in range(ir1.natom[s]-1):
#     for j in range(i+1,ir1.natom[s]):
#         if bo[i][j][0] >= 0.00001 or bo0[0][i][j] >= 0.00001:
#            print(i,j,bo0[0][i][j],bo[i][j][0])

print('\n---- irff ----\n')
images = Trajectory('md.traj')
ir_ = IRFF(atoms=images[0],libfile='ffield.json',nn=True)
ir2 = IRFF_NP(atoms=images[0],libfile='ffield.json',nn=True)

# forces = images[0].get_forces()

for i,img in enumerate(images):
    ir_.calculate(atoms=img)
    ir2.calculate(atoms=img)
    print('--     RTFlow    --      RTorce     --     IRNP     --' )
    print('E      : ',E[i],ir.E[s][i].item(),ir2.E)
    print('Ebond  : ',ebond[i],ir.ebond[s][i].item(),ir2.Ebond)
    print('Eover  : ',eover[i],ir.eover[s][i].item(),ir2.Eover)
    print('Eunder : ',eunder[i],ir.eunder[s][i].item(),ir2.Eunder)
    print('Elone  : ',elone[i],ir.elone[s][i].item(),ir2.Elone)
    print('Eang   : ',eang[i],ir.eang[s][i].item(),ir2.Eang)
    print('Epen   : ',epen[i],ir.epen[s][i].item(),ir2.Epen)
    print('Etor   : ',etor[i],ir.etor[s][i].item(),ir2.Etor)
    print('Efcon  : ',efcon[i],ir.efcon[s][i].item(),ir2.Efcon)
    print('Evdw   : ',evdw[i],ir.evdw[s][i].item(),ir2.Evdw)
    print('Ehb    : ',ehb[i],ir.ehb[s][i].item(),ir2.Ehb)
    print('Ec     : ',ecoul[i],ir.ecoul[s][i].item(),ir2.Ecoul)
    # print('\n IR-dpi \n',ir2.Dpil)
 
# print(ir.force)
print('\n----  forces  ----\n')
ir_.calculate(atoms=images[0])
for i in range(ir_.natom):
    # print(i,f[0][i],'----' ,ir.force[s][0][i].detach().numpy(),'----',forces[i])
    print(i,f[0][i],'----' ,ir.force[s][0][i].detach().numpy(),'----',ir_.results['forces'][i])

print('\n---- ecoul ----\n')
for i in range(ir.natom[s]):
    for j in range(ir.natom[s]):
        ec_ = ir.Ecoul[s][0][i][j].detach().numpy()
        ec  = Ecoul[i][j][0]
        if ec_>0.00 or ec >0.000:
           if abs(ec_-ec)>0.001: 
              print(i,j,ec,ec_)


        
