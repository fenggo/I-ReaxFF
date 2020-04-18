#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import argh
import argparse
from irff.reax import ReaxFF
from irff.mpnn import MPNN
from irff.reaxfflib import read_lib,write_lib


direcs = {'alc2f6':'/home/feng/mlff/Al/alc2f6'
          }
batch = 50

p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')

vto = []                     ## define variables to optimize
for key in p:
    k = key.split('_')
    if len(k)==1:
       continue
    kpre = k[0]     
    kc   = k[1]
    kc_  = kc.split('-')

    if kpre=='n.u.':
       continue
    # if kpre=='Devdw':
    #    continue

    if kpre=='atomic':
       vto.append(key)

    if len(kc_) == 1:
       if 'F' in kc_:
          vto.append(key)
    elif len(kc_) == 2:
       if kc=='C-F' or kc=='F-C' or kc=='Al-F' or kc=='F-Al':
          vto.append(key)
    elif len(kc_) == 3:
       app = True
       for sp in kc_:
           if sp not in ['F','C']:
              app = False
       if app:
          vto.append(key)
       # if kc == 'F-C-H' or 'H-C-F' or 'H-F-F' or 'H-F-C':
       #    vto.append(key)
       if kc == 'F-Al-F' or 'F-Al-Al' or 'Al-Al-F' or 'Al-F-F' or 'F-F-Al' :
          vto.append(key)
# print(vto)


def r():
    rn = ReaxFF(libfile='ffield',
                direcs=direcs, 
                dft='siesta',
                atomic=True,
                InitCheck=False,
                optword='nocoul',
                VariablesToOpt=vto,
                pkl=True,
                batch_size=batch,
                losFunc='n2',
                bo_penalty=10.0)

    # tc.session(learning_rate=1.0e-4,method='AdamOptimizer')
    # GradientDescentOptimizer AdamOptimizer AdagradOptimizer RMSPropOptimizer
    rn.run(learning_rate=1.0e-4,
           step=10000,
           method='AdamOptimizer', # SGD
           print_step=10,writelib=1000) 
    rn.close()


def t():
    opt= ['boc1','boc2','boc3','boc4','boc5','valboc']
    rn = ReaxFF(libfile='ffield',
                direcs=direcs, 
                dft='siesta',
                atomic=True,
                InitCheck=True,
                optword='nocoul',
                VariablesToOpt=vto,
                nn=False,
                bo_layer=[9,2],
                pkl=True,
                batch_size=batch,
                losFunc='n2',
                bo_penalty=10.0)

    # tc.session(learning_rate=1.0e-4,method='AdamOptimizer')
    # GradientDescentOptimizer AdamOptimizer AdagradOptimizer RMSPropOptimizer
    rn.sa(total_step=1000,step=100000,astep=3000,
          print_step=10,writelib=1000) 
    rn.close()


def z():
    opt=['atomic','ovun5','Desi','Depi','Depp','Devdw','Dehb'],
    rn = ReaxFF(libfile='ffield',
                direcs=direcs, 
                dft='siesta',
                atomic=True,
                optword='nocoul',
                opt=['atomic'],
                nn=False,
                cons=None,
                pkl=True,
                batch_size=batch,
                losFunc='n2',
                bo_penalty=10000.0)

    # tc.session(learning_rate=1.0e-4,method='AdamOptimizer')
    # GradientDescentOptimizer AdamOptimizer AdagradOptimizer RMSPropOptimizer
    rn.run(learning_rate=100,
              step=1000,
              print_step=10,writelib=1000) 
    rn.close()



if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [t,r,z])
   argh.dispatch(parser)

