#!/usr/bin/env python
from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import system, getcwd, chdir,listdir,environ
from irff.rdnn import RDNN
import numpy as np
import tensorflow as tf
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    

def get_v(direcs={'ethane':'/home/feng/siesta/train2/ethanee'},
          batch=1):
    for m in direcs:
        mol = m

    rn = RDNN(libfile='ffield.json',
             direcs=direcs, 
             dft='siesta',
             atomic=True,
             opt=None, 
             cons=None,
             nnopt=True,
             bo_layer=[72,10],
             ea_layer=[36,10],
             eh_layer=[12,10],
             ev_layer=[10,10],
             pkl=True,
             batch_size=batch,
             losFunc='n2',
             bo_penalty=1000.0)

    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    p      = rn.p_
    Dp_ = {}
    for sp in rn.spec:
        if rn.nsp[sp]>0:
           Dp_[sp] = tf.gather_nd(rn.Deltap,rn.atomlist[sp])
    Dp = rn.get_value(Dp_)

    Dpi    = rn.get_value(rn.Dpi)
    fbo    = rn.get_value(rn.fbo)

    bop_si = rn.get_value(rn.bop_si)
    bosi   = rn.get_value(rn.bosi)

    bop_pi = rn.get_value(rn.bop_pi)

    bop_pp = rn.get_value(rn.bop_pp)
    bop    = rn.get_value(rn.bop)
    rbd    = rn.get_value(rn.rbd)
    fo     = rn.get_value(rn.fo)

    Delta_lpcorr = rn.get_value(rn.Delta_lpcorr)
    Delta_lp     = rn.get_value(rn.Delta_lp)
    D            = rn.get_value(rn.D)

    DPIL         = rn.get_value(rn.DPIL)
    DLP          = rn.get_value(rn.DLP)

    nbd    = rn.nbd
    nsp    = rn.nsp
    spec   = rn.spec
    bonds  = rn.bonds
    bd     = 'H-H'
    # bonds  = [bd]

    bdlab  = rn.lk.bdlab
    atlab  = rn.lk.atlab
    atlall = rn.lk.atlall
    alist  = rn.atomlist


    ffbo = open('bo.txt','w')
    for bd in bonds:
        if nbd[bd]>0:
           print('-  bd: %s' %bd,fo[bd].shape)
           for i,bo in enumerate(bosi[bd]):
               print('-bond %s:' %bd,
                     '-rbd- %10.8f' %rbd[bd][i][0],
                     '-fbo- %10.8f' %fbo[bd][i][0],
                     '-fo- %10.8f' %fo[bd][i][0],
                     '-bosi- %10.8f' %bosi[bd][i][0],
                     '-bop- %10.8f' %bop[bd][i][0],
                     '-bopsi- %10.8f' %bop_si[bd][i][0],
                     '-boppi- %10.8f' %bop_pi[bd][i][0],
                     '-boppp- %10.8f' %bop_pp[bd][i][0],
                     file=ffbo)
    ffbo.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   get_v()

