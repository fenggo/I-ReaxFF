#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import argh
import argparse
from .reax import ReaxFF



# system('rm *.pkl')
system('./r2l<ffield>reax.lib')

dirs={'ethw':'/home/gfeng/siesta/train/case1',
     'ethw1':'/home/gfeng/siesta/train/case1/run_1',}
batchs={'ethw':5}

direcs = {}
for mol in batchs:
    nb = batchs[mol] if mol in batchs else 1
    for i in range(nb):
        direcs[mol+'-'+str(i)] = dirs[mol]
batch = 50


def z():
    rn = ReaxFF(libfile='ffield',
             direcs=direcs, 
             dft='siesta',
             rc_scale='none',
             optword='all',opt=['atomic'],
             atomic=False,
             pkl=True,
             batch_size=batch)

    # tc.session(learning_rate=1.0e-4,method='AdamOptimizer')
    # GradientDescentOptimizer AdamOptimizer AdagradOptimizer RMSPropOptimizer
    rn.run_op(learning_rate=60,
              step=500,
              print_step=10,writelib=100) 
    del rn


def r(direcs=None,step=5000,batch=None,total_step=None):
    pw = {'atomic':10.0,'val':0.5,'valboc':0.5,'valang':0.5,'vale':0.5,
          'rosi':0.3,'ropi':0.3,'ropp':0.3,'rvdw':0.3,'rohb':0.3,
          'boc1':0.3,'boc2':0.3,
          'lp2':0.3}
    rn = ReaxFF(libfile='ffield',
             direcs=direcs, 
             dft='siesta',
             rc_scale='none',
             rs=0.03,
             rvs=0.05,
             zw=100.0,
             pw=pw,
             optword='all',
             atomic=True,
             cons=None,
             pkl=True,
             batch_size=batch)

    rn.run_op(learning_rate=1.0e-4,
              step=step,
              print_step=10,writelib=1000) 

    p   = rn.p_
    zpe = rn.zpe_
    rn.close()
    return p,zpe


def t(direcs=None,step=5000,batch=None,total_step=2):
    cons =   ['val','valboc','vale','valang']
    pw = {'atomic':10.0,'val':0.5,'valboc':0.5,'valang':0.5,'vale':0.5,
          'rosi':0.3,'ropi':0.3,'ropp':0.3,'rvdw':0.3,'rohb':0.3,
          'boc1':0.3,'boc2':0.3,
          'lp2':0.3}
    rn = ReaxFF(libfile='ffield',
             direcs=direcs, 
             dft='siesta',
             rc_scale='none',
             rs=0.03,
             rvs=0.05,
             zw=100.0,
             pw=pw,
             optword='all',
             atomic=True,
             cons=None,
             pkl=False,
             batch_size=batch)

    rn.run(total_step=total_step,
           step=step,astep=1000,zstep=500,
           print_step=10,writelib=1000) 

    p   = rn.p_
    zpe = rn.zpe_
    rn.close()
    return p,zpe


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [z,t,m,r])
   argh.dispatch(parser)

