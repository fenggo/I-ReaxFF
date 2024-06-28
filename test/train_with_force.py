#!/usr/bin/env python
import torch
from ase import Atoms
from irff.reax_force import ReaxFF_nn_force
from irff.data.ColData import ColData

getdata = ColData()

dataset = {}
strucs = ['gp4',#'gp5',#'gpp'
          ]

weight={'c2h4':100.0,'gphit3':4.0,'cnt5-1':5.0,'nomb':0.0,'others':2.0}
batchs  = {'others':100}

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

rn = ReaxFF_nn_force(dataset={'gp4':'data/gp4-0.traj'},
                     weight_energy={'others':1.0},
                     weight_force={'others':1.0},
                     libfile='ffield.json')
# rn.forward()

# param = rn.get_parameter(rn)
# print(param)
# print(list(rn.named_parameters()))
# for p in rn.parameters():
#     print(p)
# print(rn.opt)
optimizer = torch.optim.Adam(rn.parameters(), lr=0.00001 )
 
iter_num = 10000


for step in range(iter_num):
    E,F = rn()
    loss = rn.get_loss()
    optimizer.zero_grad()

    if step%1000==0:
       rn.save_ffield('ffield_{:d}.json'.format(step))
       
    loss.backward(retain_graph=True)
    optimizer.step()

    print( "{:8d} loss: {:10.5f}   energy: {:10.5f}   force: {:10.5f}".format(step,
                loss.item(),rn.loss_e.item(),rn.loss_f.item()))
    if step%1000==0:
       rn.save_ffield('ffield_{:d}.json'.format(step))

