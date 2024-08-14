#!/usr/bin/env python
from os import  system
import json as js
import torch
from ase import Atoms
from irff.reax_force import ReaxFF_nn_force
from irff.data.ColData import ColData
from irff.reax_force_data import reax_force_data
from irff.intCheck import init_bonds,check_tors

getdata = ColData()

dataset = {}
strucs = ['cf11']

batchs  = {'others':50}

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

data = {}

with open('ffield.json','r') as lf:
     j = js.load(lf)

spec,bonds,offd,angs,torp,hbs = init_bonds(j['p'])
tors = check_tors(spec,torp)

for st in dataset:
    data_ = reax_force_data(structure=st,
                            traj=dataset[st],
                            vdwcut=10.0,
                            rcut=j['rcut'],
                            rcuta=j['rcutBond'],
                            batch=1000,
                            variable_batch=True,
                            m=j['m'],
                            mf_layer=j['mf_layer'],
                            p=j['p'],spec=spec,bonds=bonds,
                            angs=angs,tors=tors,
                            hbs=hbs,
                            screen=True)
    data[st] = data_

# rn.cuda()
# rn.compile()

iter_num = 101

for eproch in range(iter_num):
    for st in data:
        data_ = {st:data[st]}
        rn = ReaxFF_nn_force(data=data_,
                            weight_energy={'others':1.0},
                            weight_force={'others':1.0},
                            cons=['acut'],
                            tors=tors,
                            libfile='ffield.json',
                            screen=True,
                            lambda_bd=1000.0,
                            lambda_pi=0.0,
                            lambda_reg=0.01,
                            lambda_ang=0.0,
                            device='cuda')
        optimizer = torch.optim.Adam(rn.parameters(), lr=0.0001 )
        natom = 0
        for st in rn.strcs:
            natom += rn.batch[st]*rn.natom[st]
        for step in range(10):
            E,F  = rn()
            loss = rn.get_loss()
            optimizer.zero_grad()

            loss.backward(retain_graph=True)
            optimizer.step()
        print( "eproch: {:5d} loss: {:10.5f} energy: {:10.5f} force: {:10.5f} pen: {:10.5f}".format(eproch,
                          loss.item(),rn.loss_e.item()/natom,rn.loss_f.item()/natom,rn.loss_penalty.item()))
        rn.save_ffield('ffield.json')
        rn.close()

    if eproch%100==0:
       # rn.save_ffield('ffield_{:d}.json'.format(step))
       rn.save_ffield('ffield.json')
       # system('cp ffield.json ffield_{:d}.json'.format(step))

# rn.save_ffield('ffield.json')
