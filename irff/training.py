#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import argh
import argparse
import numpy as np
from .reax import ReaxFF
from .mpnn import MPNN
from .init_check import Init_Check
from .dingtalk import send_msg
import json as js


# direcs v={'ethw':'/home/gfeng/siesta/train/case1',
#           'ethw1':'/home/gfeng/siesta/train/case1/run_1'}
# batch = 50


def train(direcs=None,step=5000,batch=None,convergence=0.97,
          nanv={'pen3':-2.0,'pen4':-2.0,'tor2':-5.0,'tor4':-5.0,
                'ovun3':-2.0,'ovun4':-2.0,
                'lp1':-3.0,'vdw1':-3.0},
          writelib=1000,nn=True,bo_layer=[8,1],mpopt=1):
    ffield = 'ffield.json' if nn else 'ffield'
    rn = ReaxFF(libfile=ffield,
                direcs=direcs, 
                dft='siesta',
                spec=['C','H','O','N'],
                optword='nocoul',
                atomic=True,
                cons=None,
                opt=None,
                nn=nn,
                bo_layer=bo_layer,
                pkl=False,
                batch_size=batch,
                losFunc='n2',
                bo_penalty=100000.0,
                nanv=nanv,
                convergence=convergence) # Loss Functon can be n2,abs,mse,huber

    loss,accu,accMax,i =rn.run(learning_rate=1.0e-4,
                      step=step,
                      print_step=10,
                      writelib=writelib) 

    libstep = int(i - i%writelib)

    if i==libstep:
       libstep = libstep - writelib
    if libstep<=0:
       ffd = 'ffield.json'
    else:
       ffd = 'ffield_' + str(libstep) +'.json'

    if loss==0.0 and accu==0.0:
       send_msg('-  Warning: the loss is NaN, parameters from %s changed auomatically ...' %ffd)
       with open(ffd,'r') as fj:
            j = js.load(fj)
            ic = Init_Check(nanv=nanv)
            j['p'] = ic.auto(j['p'])
            ic.close()
       with open('ffield.json','w') as fj:
            js.dump(j,fj,sort_keys=True,indent=2)

    p   = rn.p_
    zpe = rn.zpe_

    rn.close()
    return loss,accu,accMax,p,zpe,i


def train_mpnn(direcs=None,step=5000,batch=None,convergence=0.97,
          nanv={'pen3':-2.0,'pen4':-2.0,'tor2':-5.0,'tor4':-5.0,
                'ovun3':-2.0,'ovun4':-2.0,
                'lp1':-3.0,'vdw1':-3.0},
          writelib=1000,nn=True,bo_layer=[9,2],mpopt=3):
    ''' train the massage passing model '''
    ffield = 'ffield.json' if nn else 'ffield'
    opt_ = None
    if mpopt==1:
       mpopt_=[True,True]
       massages=1
    elif mpopt==2:
       mpopt_=[False,True,True]
       massages=2
       opt_=[]
    elif mpopt==3:
       mpopt_=[True,True,True]
       massages=2

    rn = MPNN(libfile=ffield,
                direcs=direcs, 
                dft='siesta',
                spec=['C','H','O','N'],
                optword='nocoul',
                atomic=True,
                cons=None,
                opt=opt_,
                nn=nn,
                bo_layer=bo_layer,
                massages=massages,
                mpopt=mpopt_,
                pkl=False,
                batch_size=batch,
                losFunc='n2',
                bo_penalty=100000.0,
                nanv=nanv,
                convergence=convergence) # Loss Functon can be n2,abs,mse,huber

    loss,accu,accMax,i =rn.run(learning_rate=1.0e-4,
                               step=step,
                               print_step=10,
                               writelib=writelib) 

    libstep = int(i - i%writelib)

    if i==libstep:
       libstep = libstep - writelib
    if libstep<=0:
       ffd = 'ffield.json'
    else:
       ffd = 'ffield_' + str(libstep) +'.json'

    if loss==0.0 and accu==0.0:
       send_msg('-  Warning: the loss is NaN, parameters from %s changed auomatically ...' %ffd)
       with open(ffd,'r') as fj:
            j = js.load(fj)
            ic = Init_Check(nanv=nanv)
            j['p'] = ic.auto(j['p'])
            ic.close()
       with open('ffield.json','w') as fj:
            js.dump(j,fj,sort_keys=True,indent=2)

    p   = rn.p_
    zpe = rn.zpe_

    rn.close()
    return loss,accu,accMax,p,zpe,i


def sa(direcs=None,step=5000,batch=None,total_step=2):
    rn = ReaxFF(libfile='ffield',
              direcs=direcs, 
              dft='siesta',
              spec=['C','H','O','N'],
              ro_scale=0.1,
              optword='nocoul',
              atomic=True,
              cons=None,
              opt=None,
              pkl=False,
              batch_size=batch,
              losFunc='n2',
              bo_penalty=100.0) # Loss Functon can be n2,abs,mse,huber


    rn.sa(total_step=total_step,
           step=step,astep=1000,zstep=500,
           print_step=10,writelib=1000) 

    p   = rn.p_
    zpe = rn.zpe_
    rn.close()
    return loss,accu,p,zpe


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [z,sa,train])
   argh.dispatch(parser)

