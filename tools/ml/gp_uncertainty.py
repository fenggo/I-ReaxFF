#!/usr/bin/env python
import json as js
import numpy as np
from scipy.optimize import leastsq # minimize # minimize_scalar
from irff.data.ColData import ColData
from irff.ml.fit import train
from irff.ml.data import get_data,get_md_data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF


def fit(step=20000):
    dataset = {}
    strucs = ['tkx','tkx2']
    # strucs = ['tkxmd']

    trajdata = ColData()
    for mol in strucs:
        trajs = trajdata(label=mol,batch=50)
        dataset.update(trajs)

    bonds = ['C-C','C-H','C-N','H-O','C-O','H-H','H-N','N-N','O-N','O-O'] 
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = {}
    for bd in bonds:
        # for i,bp in enumerate(Bp[bd]):
        #     print(i,R[bd][i],D[bd][i][1],np.sum(B[bd][i]),E[bd][i])
        if bd not in D:
           continue
        D_  = np.array(D[bd])
        B_  = np.array(B[bd])

        print('Gaussian Process for {:s} bond ...'.format(bd))
        gp[bd] = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9,
                                        optimizer=None) # fmin_l_bfgs_b
        gp[bd].fit(D_, B_)
        # print(gp[bd].kernel_)
        print('the score of exsiting data: ',gp[bd].score(D_, B_))
        D_md,Bp_md,B_md,R_md,E_md = get_md_data(images=None, traj='md.traj', bonds=['O-N'],ffield='ffieldData.json')
        
        if bd not in D_md:
           continue
        D_  = np.array(D_md[bd])
        B_  = np.array(B_md[bd])
        print('the score of new data: ',gp[bd].score(D_, B_))
        
        B_pred, std_pred = gp[bd].predict(D_, return_std=True)
        # print(len(D[bd]))
        # print(len(D_md[bd]))
        D[bd].extend(D_md[bd])
        B[bd].extend(B_pred.tolist())
        Bp[bd].extend(Bp_md[bd])
 
    train(Bp,D,B,E,bonds=bonds,step=step,fitobj='BO',learning_rate=0.0001)
    
if __name__ == '__main__':
   help_= 'Run with commond: ../gp_uncertainty.py  '
   fit()

