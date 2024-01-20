#!/usr/bin/env python
import numpy as np
import dpdata
from irff.data.ColData import ColData

''' convert ase structure data to deepmd data format '''


trajdata = ColData()

dataset = {}

strucs = ['gpp',
          'gp',
          'gphit3',
          'cnt8',
          'cnt10',
          ]
batchs = {'others':50}

for mol in strucs:
    trajs = trajdata(label=mol,batch=10000)
    dataset.update(trajs)

# data=dpdata.LabeledSystem('00_fp/000/output.log',fmt='ase/structure')
# print(data)



