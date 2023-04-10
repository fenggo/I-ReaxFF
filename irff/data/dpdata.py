#!/usr/bin/env python
import numpy as np


positions  = np.load('trainData/set.000/coord.npy')
box        = np.load('trainData/set.000/box.npy')
energies   = np.load('trainData/set.000/energy.npy')
forces     = np.load('trainData/set.000/force.npy')


print('\n- positions -\n',positions.shape)
print('\n- boxes -\n',box.shape)
print('\n- energies -\n',energies.shape)
print('\n- forces -\n',forces.shape)

# for b in box:
#     print(b)

