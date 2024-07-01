#!/usr/bin/env python
import torch
from ase import Atoms
from irff.reax_nn import ReaxFF_nn
from irff.data.ColData import ColData

getdata = ColData()


rn = ReaxFF_nn(dataset={'gp4-0':'gp4-0.traj'，
                        'gp4-1':'gp4-1.traj'},
               libfile='ffield.json',
               screen=True)
rn.run(learning_rate=0.0001,step=1000)

