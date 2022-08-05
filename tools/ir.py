#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from irff.plot.reax_plot import plbo
# from irff.plot import reax_pldd,reax_plbd
from irff.irff import IRFF
from irff.reax_eager import ReaxFF
from ase.io import read,write
import numpy as np


# direcs={'tatb2':'/home/feng/siesta/tatb2_4'}

atoms = read('siesta.traj',index=29)

ir = IRFF(atoms=atoms,
          libfile='ffield.json',
          rcut=None,
          nn=True)

ir.get_pot_energy(atoms)


# print('\n-  ebond:\n',ir.Ebond.numpy())
# # print('\n-  ebond:\n',ir.Delta.numpy())
# print('\n-  elone:\n',ir.Elone.numpy())
# print('\n-  eover:\n',ir.Eover.numpy())
# print('\n-  eunder:\n',ir.Eunder.numpy())
# print('\n-  etor:\n',ir.Etor.numpy())
# print('\n-  efcon:\n',ir.Efcon.numpy())
# print('\n-  ehb:\n',ir.Ehb.numpy())
grad = ir.tape.gradient(ir.E,ir.positions)
print('\n-  gradient:\n',grad.numpy())

# rn = ReaxFF(libfile='ffield.json',
#             direcs={'tmp':'siesta.traj'},
#             dft='siesta',
#             opt=[],optword='nocoul',
#             batch_size=50,
#             atomic=True,
#             clip_op=False,
#             InitCheck=False,
#             nn=True,
#             pkl=False,
#             to_train=False) 
# molecules = rn.initialize()
# rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

# e_  = rn.get_value(rn.E['tmp'])
# print(e_)

