#!/usr/bin/env python
from ase.io.trajectory import Trajectory,TrajectoryWriter
from irff.reax_force import ReaxFF_nn_force
from irff.reax_nn import ReaxFF_nn
from irff.irff_np import IRFF_NP
from irff.irff_autograd import IRFF
from irff.data.ColData import ColData
import matplotlib.pyplot as plt
from irff.md.gulp import get_gulp_forces

traj = 'md.traj'

ir = ReaxFF_nn_force(dataset={'md':traj},
                     MessageFunction=3,
                     mf_layer=[9,2],
                     be_layer=[9,1],
                     screen=True,
                     libfile='ffield.json')
ir.forward()

ir2 = ReaxFF_nn(dataset={'md':traj},
               MessageFunction=3,
               mf_layer=[9,2],
               be_layer=[9,1],
               screen=True,
               libfile='ffield.json')
ir2.initialize()
ir2.session()  

for s in ir.bop:
    (E,ebond,elone,eover,eunder,eang,etor,etcon,epen,efcon,evdw,
                ehb,ecoul) = ir2.sess.run([ir2.E[s],
                                           ir2.ebond[s],
                  ir2.elone[s],ir2.eover[s],ir2.eunder[s],
                  ir2.eang[s],ir2.etor[s],ir2.etcon[s],ir2.epen[s],
                                          ir2.efcon[s],
                                          ir2.evdw[s],ir2.ehb[s],
                            ir2.ecoul[s]],feed_dict=ir2.feed_dict)
ir2.get_forces(s)
f = ir2.sess.run(ir2.forces[s],feed_dict=ir2.feed_dict)

print('\n----    reax_nn    ----\n')
Ehb = ir2.sess.run(ir2.Ehb[s],feed_dict=ir2.feed_dict)
print('\n Ehb \n',Ehb.shape)

 
print('\n---- reax_nn_force ----\n')
print('\n Ehb \n',ir.Ehb[s].shape)
print(ir.hbs)

Ehb_ = ir.Ehb[s].detach().numpy()[0]
for i,e in enumerate(Ehb):
    if abs(e[0])>1e-5 or abs(Ehb_[i])>1e-5:
       print(i,e[0],Ehb_[i])
# print(ir.nhb)

print('\n---- irff ----\n')
images = Trajectory(traj)
ir_ = IRFF_NP(atoms=images[0],libfile='ffield.json',nn=True)

forces = images[0].get_forces()

for i,img in enumerate(images):
    ir_.calculate(atoms=img)
    print('-------     IR   --    RTC   --   RTF   --' )
    print('E      : ',ir_.E,ir.E[s][i].item(),E[i])
    print('Eover  : ',ir_.Eover,ir.eover[s][i].item(),eover[i])
    print('Eunder : ',ir_.Eunder,ir.eunder[s][i].item(),eunder[i])
    print('Elone  : ',ir_.Elone,ir.elone[s][i].item(),elone[i])
    print('Eang   : ',ir_.Eang,ir.eang[s][i].item(),eang[i])
    print('Epen   : ',ir_.Epen,ir.epen[s][i].item(),epen[i])
    print('Etcon  : ',ir_.Etcon,ir.etcon[s][i].item(),etcon[i])
    print('Etor   : ',ir_.Etor,ir.etor[s][i].item(),etor[i])
    print('Efcon  : ',ir_.Efcon,ir.efcon[s][i].item(),efcon[i])
    print('Evdw   : ',ir_.Evdw,ir.evdw[s][i].item(),evdw[i])
    print('Ecoul  : ',ir_.Ecoul,ir.ecoul[s][i].item(),ecoul[i])
    print('Ehb    : ',ir_.Ehb,ir.ehb[s][i].item(),ehb[i])
    # print('\n IR-dpi \n',ir2.Dpil)
 

 
# print('\n----  forces  ----\n')
# print('--  IR  --   RTC  --  GULP  --' )
# ir_.calculate(atoms=images[0])
# # ir_.results['forces'][i]
# for i in range(ir_.natom):
#     print(f[0][i],'----',ir.force[s][0][i].detach().numpy(), '----',forces[i])


