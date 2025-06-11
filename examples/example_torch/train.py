#!/usr/bin/env python
from os import  system
import time
import json as js
import numpy as np
import torch
from ase import Atoms
from irff.reaxff_torch import ReaxFF_nn
from irff.data.ColData import ColData
from irff.reax_force_data import reax_force_data
from irff.intCheck import init_bonds,check_tors

getdata = ColData()

dataset = {}
strucs = ['ch4w2']

batch        = 50
batchs       = {'others':batch}
for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)
 
constants  = ['lp1',# 'lp2',
              'theta0','val8',
              'valang','val9','val10','vale',
              'valboc','cot1','cot2','coa1','coa2','coa3','coa4',
              #'pen1','pen2','pen3','pen4',
              #'vdw1','gammaw','rvdw','alfa','Devdw',
              'ovun5', 'ovun6','ovun7','ovun8',
              #'ovun2','ovun3','ovun4','ovun1',
              'val','lp3',# 'cutoff',
              'hb1','hb2',
              'acut' ]    
#  'val7':(0.0,10.0),
#  'lp2':(0.0,999.0),'pen1':(-60.0,196.0),'ovun1':(0.0,2.0)
 
clip = {'Devdw':[0,0.856],
        #'pen1':[-45,90.0], 
        'ovun1':(0.0,1.0),
        'Dehb':(-3.998,0.0),'Dehb_C-H-O':(-3.9,-0.35),
        'rohb':(1.877,2.392),'hb1':(2.72,3.64),'hb2':(18.7,19.64),
        'rvdw_C':(1.84,2.399),'rvdw_O':(1.84,2.50),'rvdw_H':(1.62,2.39),
        'rvdw_N':(1.9,2.79), 'rvdw_H-N':(1.65,2.4),'rvdw_H-O':(1.64,2.79),
        'rvdw_C-H':(1.64,2.38),
       }
data = {}
# while True:          # clip values
#     for key in clip:
#         if isinstance(clip[key],list):
#            clip[key][0] = clip[key][0]*0.99
#            clip[key][1] = clip[key][1]*0.99

rn = ReaxFF_nn(dataset=dataset,data=data,
            weight_energy={'others':0.20},
            weight_force={'tkx2':1.0,'c3h8':1},
            cons=constants,
            libfile='ffield.json',
            clip=clip,
            screen=True,
            lambda_bd=100.0,
            lambda_pi=0.0,
            lambda_reg=0.001,
            lambda_ang=0.0,
            device={'cf21-1':'cpu:0','others':'cuda'})
data = rn.data
# print(rn.cons)
optimizer = torch.optim.Adam(rn.parameters(), lr=0.0001 )
# rn.cuda()
# rn.compile()
n_epoch = 100001   

for epoch in range(n_epoch):
    los   = []
    los_e = []
    los_f = []
    los_p = []
    start = time.time()
    optimizer.zero_grad()
    for st in rn.strcs:
        E,F  = rn(st)             # forward
        loss = rn.get_loss(st)
        loss.backward(retain_graph=False)
        los.append(loss.item())
        los_e.append(rn.loss_e.item()/rn.natom[st])
        los_f.append(rn.loss_f.item()/rn.natom[st])
        los_p.append(rn.loss_penalty.item())
    optimizer.step()        # update parameters 
    rn.clamp()
    use_time = time.time() - start
    print( "eproch: {:5d} loss : {:10.5f} energy: {:7.5f} force: {:7.5f} pen: {:10.5f} time: {:6.3f}".format(epoch,
            np.mean(los),np.mean(los_e),np.mean(los_f),np.mean(los_p),use_time))

    if epoch%1000==0:
       rn.save_ffield('ffield_{:d}.json'.format(epoch))
       # rn.save_ffield('ffield.json')
print('\n------------------- Loss for Batch ------------------')
print('- Batch     TolLoss    EnergyLoss   ForceLoss         -')
for st,l,l_e,l_f,l_p in zip(rn.strcs,los,los_e,los_f,los_p):
    print('{:10s} {:11.5f} {:10.5f} {:10.5f}'.format(st,l,l_e,l_f))
rn.close()
