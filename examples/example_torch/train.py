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
 

rn = ReaxFF_nn(dataset=dataset,
               weight_energy={'others':1.0},
               weight_force={'ch4w2':1.0},
               cons=['acut'],
               libfile='ffield.json',
               screen=True,
               lambda_bd=100.0,
               lambda_pi=0.0,
               lambda_reg=0.001,
               lambda_ang=0.0,
               device={'ch4w2-0':'cpu:0','others':'cpu:0'})
optimizer = torch.optim.Adam(rn.parameters(), lr=0.0001 )
# rn.cuda()
# rn.compile()

natom = 0
for st in rn.strcs:
    natom += rn.batch[st]*rn.natom[st]

n_epoch = 101

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
    use_time = time.time() - start
    print( "eproch: {:5d} loss : {:10.5f} energy: {:7.5f} force: {:7.5f} pen: {:10.5f} time: {:6.3f}".format(epoch,
            np.mean(los),np.mean(los_e),np.mean(los_f),np.mean(los_p),use_time))

    if epoch%1000==0:
       rn.save_ffield('ffield_{:d}.json'.format(epoch))
    
print('\n------------- Loss for Batch -------------')
print('-  Batch  TolLoss EnergyLoss ForceLoss   -')
for st,l,l_e,l_f,l_p in zip(rn.strcs,los,los_e,los_f,los_p):
    print('{:10s} {:.5f} {:.5f} {:.5f}'.format(st,l,l_e,l_f))


