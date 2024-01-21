#!/usr/bin/env python
import dpdata
import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory
import os



ms = dpdata.MultiSystems()
path      = './' 
path_list = os.listdir(path)
path_list.sort()
for filename in path_list:
    file  = os.path.join(path, filename)
    fil   = file.split()[0]
    print(file)
    if file.endswith('.traj'):
       sys = dpdata.MultiSystems.from_file(file_name=file, fmt='ase/structure')
       print(sys.systems)
       ms.append(sys)
       # sys.to_deepmd_npy('training_data/') 
       sys.to_deepmd_npy('dp_data/{:s}'.format(fil)) 
       
 

print(ms.systems)
print('\n# the multisystems contains %d systems' % len(ms))
print("# the information of the MultiSystems is:\n", ms)


