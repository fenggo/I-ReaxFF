#!/usr/bin/env python
import json as js
import numpy as np
from scipy.optimize import leastsq # minimize # minimize_scalar
from irff.data.ColData import ColData
from irff.ml.fit import train
from irff.ml.data import get_data,get_md_data_invariance # ,get_bond_data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF
from train import dataset


bonds = ['C-H','C-C','H-H'] 
D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')


trajs = ['md.traj']
for traj in trajs:
    D_md,Bp_md,B_md,R_md,E_md = get_md_data_invariance(traj=traj, 
                                                       bonds=bonds,
                                                       ffield='ffieldData.json')
    # D1,Bp1,B1,R1,E1 = get_bond_data(5,19, traj='md.traj', 
    #                                 bonds=bonds,
    #                                 ffield='ffieldData.json')
    for bd in bonds:
        if bd not in D:
            continue
        if bd not in D_md:
            continue

        D[bd].extend(D_md[bd])
        B[bd].extend(B_md[bd])
        Bp[bd].extend(Bp_md[bd])

train(Bp,D,B,E,bonds=bonds,step=10000,fitobj='BO',learning_rate=0.0001)
    
 