#!/usr/bin/env python
import torch
from ase import Atoms
from irff.reax_nn import ReaxFF_nn
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

rn = ReaxFF_nn(dataset={'gp4':'data/gp4-0.traj'},
               libfile='ffield.json',
               screen=True)
rn.run(learning_rate=0.0001,step=1000)

