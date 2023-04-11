#!/usr/bin/env python
import argh
import argparse
import json as js
import numpy as np
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from irff.data.ColData import ColData
from irff.ml.data import get_data,get_bond_data,get_md_data
from irff.ml.fit import train

''' Official example:
X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
kernel = DotProduct() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X, y)
gp.score(X, y)
gp.predict(X[:2,:], return_std=True)
'''

def bo(i=4,j=1,traj='md.traj',bonds=None):
    D, Bp, B, R, E = get_bond_data(i,j,images=None, traj=traj,bonds=bonds)
    for i,bp in enumerate(Bp):
        print('step: {:4d} R: {:6.4f} Di: {:6.4f} Bij: {:6.4f} Dj: {:6.4f} '
              'B: {:6.4f} E: {:6.4f}'.format(i,
              R[i],D[i][0],D[i][1],D[i][2],np.sum(B[i]),E[i]))
    print('\n r & E:',R[i],E[i])
    print('\n B\': \n',Bp[i])
    print('\n D: \n',D[i])
    print('\n B: \n',B[i])


def fit(step=1000,obj='BO'):
    dataset = { 'hmx2-13':'aimd_hmx2/hmx2-13/hmx2.traj'}

    trajdata = ColData()
    strucs = []#'hmx2']

    batchs = {'others':50}

    for mol in strucs:
        b = batchs[mol] if mol in batchs else batchs['others']
        trajs = trajdata(label=mol,batch=b)
        dataset.update(trajs)

    bonds = ['C-N']
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    D_,Bp_,B_,R_,E_ = get_md_data(traj='md.traj',bonds=bonds,ffield='ffieldData.json')
    
    kernel = DotProduct() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel,random_state=0)

    for bd in bonds:        ## 高斯模型  
        # gp.fit(D[bd],E[bd])
        X = D[bd]
        Y = B[bd]
        X_= D_[bd]
        gp.fit(X, Y)        ## for the bond order 
        print('the score of current model: ',gp.score(X, Y))
        Y_ = gp.predict(X_) # ,return_std=True

        for bp,b_,b in zip(X_,Y_,B_[bd]):
            print(bp,b_,b)

    ## 高斯模型  
    # for bd in bonds:         
    #     # gp.fit(D[bd],E[bd])
    #     X = B[bd]
    #     Y = E[bd]
    #     X_= B_[bd]
    #     gp.fit(X, Y)        ## for the bond energy
    #     print('the score of current model: ',gp.score(X, Y))
    #     Y_ = gp.predict(X_) # ,return_std=True

    #     for bp,b_,b in zip(X_,Y_,E_[bd]):
    #         print(bp,b_,b)

    # train(Bp,D,B,E,step=step,fitobj=obj,bonds=bonds)

   
if __name__ == '__main__':
   ''' Run with commond: 
      ./gp.py fit --o=BE --s=3000 
      ./gp.py bo  --t=data/n2h4-0.traj --i=2 --j=1 
      '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [fit,bo])
   argh.dispatch(parser)  



