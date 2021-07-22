#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import argh
import argparse
import numpy as np
from .reax import ReaxFF
from .mpnn import MPNN
from .initCheck import Init_Check
from .dingtalk import send_msg
import json as js


# direcs v={'ethw':'/home/gfeng/siesta/train/case1',
#           'ethw1':'/home/gfeng/siesta/train/case1/run_1'}
# batch = 50


def train_reax(direcs=None,step=5000,batch=None,convergence=0.97,lossConvergence=1000.0,
          nanv={'pen3':-2.0,'pen4':-2.0,'tor2':-5.0,'tor4':-5.0,
                'ovun3':-2.0,'ovun4':-2.0,
                'lp1':-3.0,'vdw1':-3.0},
          spec=[],optword='nocoul',
          writelib=1000,
          nn=False,vdwnn=False,
          lambda_me=0.1,regularize=False,
          EnergyFunction=3,MessageFunction=2,mpopt=3,
          bo_layer=None,
          mf_layer=None,
          be_layer=None,
          vdw_layer=None,
          be_univeral_nn='all',
          bo_univeral_nn='all',
          mf_univeral_nn='all',
          vdw_univeral_nn='all',
          spv_be=False,
          spv_bm=False,
          spv_ang=False,
          bore={'others':0.45},
          bom={'others':1.20},
          weight={'others':10.0},
          lambda_bd=10000.0,lambda_reg=0.0001,lambda_ang=0.02,
          learning_rate=1.0e-4,
          ffield='ffield.json'):
    ''' training the force field '''
    rn = ReaxFF(libfile=ffield,
                direcs=direcs, 
                dft='siesta',
                spec=spec,
                optword=optword,
                opt=None,
                nn=nn,
                pkl=False,
                batch_size=batch,
                losFunc='n2',
                lambda_bd=lambda_bd,
                nanv=nanv,
                weight=weight,
                bore=bore,
                lambda_me=lambda_me,
                convergence=convergence,
                lossConvergence=lossConvergence) # Loss Functon can be n2,abs,mse,huber
 
    loss,accu,accMax,i,zpe =rn.run(learning_rate=learning_rate,
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
       send_msg('-  Warning: the loss is NaN!')
       return -1,1,accMax,None,zpe,i

    p   = rn.p_
    rn.close()
    return loss,accu,accMax,p,zpe,i


def train_mpnn(direcs=None,step=5000,batch=None,convergence=0.97,lossConvergence=1000.0,
               nanv={'pen3':-2.0,'pen4':-2.0,'tor2':-5.0,'tor4':-5.0,
                     'ovun3':-2.0,'ovun4':-2.0,
                     'lp1':-3.0,'vdw1':-3.0},
               optword='nocoul',
               spec=[],
               writelib=1000,nn=True,vdwnn=True,
               bo_layer=[4,1],mf_layer=[9,2],be_layer=[6,1],vdw_layer=[6,1],
               be_univeral_nn='all',bo_univeral_nn='all',mf_univeral_nn='all',
               vdw_univeral_nn='all',
               EnergyFunction=3,MessageFunction=2,mpopt=3,
               spv_be=False,bore={'others':0.45},
               spv_bm=False,bom={'others':1.0},
               lambda_me=0.1,lambda_ang=0.02,lambda_bd=1000.0,
               weight={'others':10.0},
               spv_ang=False,
               regularize=True,lambda_reg=0.0001,
               learning_rate=1.0e-4,
               ffield = 'ffield.json'):
    ''' train the massage passing model '''
    opt_ = None
    if mpopt==1:
       mpopt_=[True,True,True,True]
       messages=1
    elif mpopt==2:
       mpopt_=[False,True,True]
       messages=2
       opt_=[]
    elif mpopt==3:
       mpopt_=[True,True,True,True]
       messages=2

    rn = MPNN(libfile=ffield,
              direcs=direcs, 
              dft='siesta',
              spec=spec,
              optword=optword,opt=opt_,
              nn=nn,vdwnn=vdwnn,
              bo_layer=bo_layer,mf_layer=mf_layer,
              be_layer=be_layer,vdw_layer=vdw_layer,
              be_univeral_nn=be_univeral_nn,
              bo_univeral_nn=bo_univeral_nn,
              mf_univeral_nn=mf_univeral_nn,
              vdw_univeral_nn=vdw_univeral_nn,
              EnergyFunction=EnergyFunction,
              MessageFunction=MessageFunction,
              messages=messages,
              mpopt=mpopt_,
              pkl=False,
              batch_size=batch,
              losFunc='n2',
              regularize=regularize,lambda_reg=lambda_reg,lambda_bd=lambda_bd,
              nanv=nanv,
              bom=bom,bore=bore,spv_bm=spv_bm,spv_be=spv_be,
              spv_ang=spv_ang,lambda_ang=lambda_ang,
              weight=weight,lambda_me=lambda_me,
              convergence=convergence,
              lossConvergence=lossConvergence) # Loss Functon can be n2,abs,mse,huber

    loss,accu,accMax,i,zpe =rn.run(learning_rate=learning_rate,
                                   step=step,
                                   print_step=10,
                                   writelib=writelib) 

    libstep = int(i - i%writelib)
    if i==libstep:
       libstep = libstep - writelib
    if libstep<=0:
       ffd = 'ffield.json'
    else:
       ffd = 'ffield_' + str(libstep) + '.json'

    p   = rn.p_
    if loss==0.0 and accu==0.0:
       send_msg('-  Warning: the loss is NaN!')
       # with open(ffd,'r') as fj:
       #      j = js.load(fj)
       #      ic = Init_Check(nanv=nanv)
       #      j['p'] = ic.auto(j['p'])
       #      ic.close()
       # with open('ffield.json','w') as fj:
       #      js.dump(j,fj,sort_keys=True,indent=2)
       return -1,1,accMax,p,zpe,i

    rn.close()
    return loss,accu,accMax,p,zpe,i


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [train,train_mpnn])
   argh.dispatch(parser)

