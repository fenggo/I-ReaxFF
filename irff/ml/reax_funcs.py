#!/usr/bin/env python
import json as js
import numpy as np
from scipy.optimize import leastsq # minimize # minimize_scalar
from irff.data.ColData import ColData
from irff.ml.data import get_data,get_bond_data # ,get_md_data
# from irff.ml.fit import train
# from train import dataset


def bond_order(bop,Di,Dj,vali,valj,valboci,valbocj,
               boc1=50.0,boc2=9.0,
               boc3i=8.3542,boc3j=8.3542,
               boc4i=2.8297,boc4j=2.8297,
               boc5i=0.0000,boc5j=0.0000):    
    ''' compute bond-order '''
    f1_     = f1(boc1,boc2,vali,valj,Di,Dj)
    f4_,f5_ = f45(boc3i,boc3j,boc4i,boc4j,boc5i,boc5j,valboci,valbocj,Di,Dj,bop)
    f_     = f1_*f1_*f4_*f5_
    return f_

def f1(boc1,boc2,vali,valj,Di,Dj):
    Div = Di - vali # replace val in f1 with valp, 
    Djv = Dj - valj # different from published ReaxFF model
    f_2 = f2(boc1,Div,Djv)
    f_3 = f3(boc2,Div,Djv)
    f_1 = 0.5*(np.divide(vali+f_2,
                            vali+f_2+f_3) + 
                        np.divide(valj+f_2,
                            valj+f_2+f_3))
    return f_1

def f2(boc1,Di,Dj):
    dexpf2  = np.exp(-boc1*Di)
    dexpf2t = np.exp(-boc1*Dj)
    f_2     =  dexpf2 + dexpf2t
    return f_2

def f3(boc2,Di,Dj):
    dexpf3 = np.exp(-boc2*Di)
    dexpf3t= np.exp(-boc2*Dj)

    delta_exp       = dexpf3+dexpf3t
    dexp            = 0.5*delta_exp 

    f3log = np.log(dexp)
    f_3   = np.divide(-1.0,boc2)*f3log
    return f_3

def f45(boc3i,boc3j,boc4i,boc4j,boc5i,boc5j,valboci,valbocj,Di,Dj,bop):
    Di_boc = Di - valboci # + p['val_'+atomi]
    Dj_boc = Dj - valbocj # + p['val_'+atomj]
    
    # boc3 boc4 boc5 must positive
    boc3 = np.sqrt(np.abs(boc3i*boc3j))
    boc4 = np.sqrt(np.abs(boc4i*boc4j))
    boc5 = np.sqrt(np.abs(boc5i*boc5j))
    
    df4 = boc4*np.square(bop)-Di_boc
    f4r = np.exp(-boc3*(df4)+boc5)

    df5 = boc4*np.square(bop)-Dj_boc
    f5r = np.exp(-boc3*(df5)+boc5)

    f_4 = np.divide(1.0,1.0+f4r)
    f_5 = np.divide(1.0,1.0+f5r)
    return f_4,f_5

def fit(step=1000,obj='BO'):
    unit = 4.3364432032e-2
    Desi = 424.95
    with open('ffield.reax.tkx.json','r') as lf:
        j     = js.load(lf)
        param = j['p']
 
    dataset = {}
    strucs = ['tkx','tkx2']

    trajdata = ColData()
    for mol in strucs:
        trajs = trajdata(label=mol,batch=50)
        dataset.update(trajs)

    bonds = ['C-C','C-H','C-N','H-O','C-O','H-H','H-N','N-N','O-N','O-O'] 
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    bd = 'O-N'
    atomi,atomj = bd.split('-')
    vali = param['val_'+atomi]*1.267
    valj = param['val_'+atomj]*1.267
    valboci = param['valboc_'+atomi]
    valbocj = param['valboc_'+atomj]
    boc1    = param['boc1']
    boc2    = param['boc2']

    # for i,bp in enumerate(Bp[bd]):
    #     print(i,R[bd][i],D[bd][i][1],np.sum(B[bd][i]),E[bd][i])
    D_  = np.array(D[bd])
    B_  = np.array(B[bd])
    bop = np.array(D_[:,1])
    bo  = np.sum(B_,axis=1)
    # print(D_.shape)
    Di = D_[:,0] + D_[:,1]
    Dj = D_[:,2] + D_[:,1]

    def fbo(bop,p):    
        vali,valj,boc3i,boc3j,boc4i,boc4j,boc5i,boc5j= p
        f1_     = f1(boc1,boc2,vali,valj,Di,Dj)
        f4_,f5_ = f45(boc3i,boc3j,boc4i,boc4j,boc5i,boc5j,valboci,valbocj,Di,Dj,bop)
        bo_     = bop*f1_*f1_*f4_*f5_
        return bo_
    
    def residuals(p, bo, bop):
        return bo - fbo(bop, p)
    
    # vali = 5.0
    # valj = 3.8
    boc2,boc3i,boc3j,boc4i,boc4j,boc5i,boc5j= [9.0,8.3542,8.3542,2.8297,2.8297,0.0000,0.0000]
    f1_     = f1(boc1,boc2,vali,valj,Di,Dj)
    f4_,f5_ = f45(boc3i,boc3j,boc4i,boc4j,boc5i,boc5j,valboci,valbocj,Di,Dj,bop)
    f_      = f1_*f1_*f4_*f5_
    bo_     = bop*f_
    for i,d in enumerate(Di):
        print('{:6d} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(i,
                        Di[i],Dj[i],bop[i],bo[i],bo_[i],f_[i],f1_[i],f4_[i]))

    # train(Bp,D,B,E,bonds=bonds,step=step,fitobj=obj)

    # result = minimize(sque, p0, method ='BFGS',tol = 0.0000001,
    #               options={'disp':True,'maxiter': 1000000000})

    #*********** TRAIN ***********
    # p0 = [param['val_'+atomi],param['val_'+atomj],
    #       param['boc3_'+atomi],param['boc3_'+atomj],
    #       param['boc4_'+atomi],param['boc4_'+atomj],
    #       param['boc5_'+atomi],param['boc5_'+atomj] ]     
    
    # p  = leastsq(residuals,p0,args=(bop, bo))
    
    # # print(p)
    # (param['val_'+atomi],param['val_'+atomj],
    #  param['boc3_'+atomi],param['boc3_'+atomj],
    #  param['boc4_'+atomi],param['boc4_'+atomj],
    #  param['boc5_'+atomi],param['boc5_'+atomj] ) = p[0]
    
    # print('vali ={:f}, valj ={:f}: '.format(p[0][0],p[0][1]))
    # print('boc1 ={:f}, boc2 ={:f}: '.format(50.0,boc2))
    # print('boc3i={:f}, boc3j={:f}: '.format(p[0][2],p[0][3]))
    # print('boc4i={:f}, boc4j={:f}: '.format(p[0][4],p[0][5]))
    # print('boc5i={:f}, boc5j={:f}: '.format(p[0][6],p[0][7]))


if __name__ == '__main__':
   help_= 'Run with commond: ../fit_reax.py  '
   fit()

