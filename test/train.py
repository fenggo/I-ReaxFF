#!/usr/bin/env python
import torch
from ase import Atoms
from irff.reax_force import ReaxFF_nn_force
 


rn = ReaxFF_nn_force(dataset={'gp4':'gp4.traj'},
                     libfile='ffield.json')
# rn.forward()

# param = rn.get_parameter(rn)
# print(param)
# print(list(rn.named_parameters()))
# for p in rn.parameters():
#     print(p)
# print(rn.opt)
optimizer = torch.optim.Adam(rn.parameters(), lr=0.0001 )
 
iter_num = 100


for step in range(iter_num):
    E,F = rn()
    loss = rn.get_loss()
    # optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    print( "{:8d} loss: {:20.5f} energy: {:20.5f} force: {:20.5f}".format(step,
                loss.item(),rn.loss_e.item(),rn.loss_f.item()))
 
