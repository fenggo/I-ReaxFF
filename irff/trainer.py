#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import argh
import argparse
import numpy as np
from .reax import ReaxFF
from .reax_nn import ReaxFF_nn
from .mpnn import MPNN
# from .intCheck import Intelligent_Check 
from .dingtalk import send_msg
import json as js


# dataset ={'ethw':'/home/gfeng/siesta/train/case1',
#           'ethw1':'/home/gfeng/siesta/train/case1/run_1'}
# batch = 50


def train_reax(dataset=None,step=5000,batch=None,convergence=0.97,lossConvergence=1000.0,
          spec=[],optword='nocoul',cons=None,clip={},
          writelib=1000,
          nn=False,vdwnn=False,
          lambda_me=0.1,# regularize=False,
          EnergyFunction=3,MessageFunction=2,
          messages=1,mpopt=3,
          bo_layer=None,
          mf_layer=None,
          be_layer=None,
          vdw_layer=None,
          VdwFunction=0,
          BOFunction=0,
          be_universal_nn='all',
          bo_universal_nn='all',
          mf_universal_nn='all',
          vdw_universal_nn='all',
          spv_bo=False,#boc={},#boup={},
          spv_be=False,belo={},beup={},
          spv_vdw=False,vlo={},vup={},
          spv_ang=False,
          bore={'others':0.45},
          bom={'others':1.20},
          weight={'others':10.0},
          lambda_bd=10000.0,lambda_reg=0.0001,lambda_ang=0.02,
          learning_rate=1.0e-4,
          ffield='ffield.json',**kwargs):
    ''' training the force field '''
    rn = ReaxFF(libfile=ffield,
                dataset=dataset, 
                dft='siesta',
                spec=spec,
                optword=optword,
                opt=None,clip=clip,
                nn=nn,
                batch_size=batch,
                losFunc='n2',
                lambda_bd=lambda_bd,
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


def train_mpnn(dataset=None,step=5000,batch=None,convergence=0.97,lossConvergence=1000.0,
               optword='nocoul',cons=None,clip={},
               spec=[],
               writelib=1000,nn=True,vdwnn=True,VdwFunction=1,
               bo_layer=[4,1],mf_layer=[9,2],be_layer=[6,1],vdw_layer=[6,1],
               be_universal_nn='all',bo_universal_nn='all',mf_universal_nn='all',
               vdw_universal_nn='all',
               BOFunction=0,EnergyFunction=3,MessageFunction=2,
               messages=1,mpopt=[1,1,1,1],
               spv_bo=False,#boc={},#boup={},
               spv_be=False,belo={},beup={},
               spv_vdw=False,vlo={},vup={},
               lambda_me=0.1,lambda_ang=0.02,lambda_bd=1000.0,
               weight={'others':10.0},
               spv_ang=False,
               lambda_reg=0.0001, # regularize=True,
               learning_rate=1.0e-4,
               ffield = 'ffield.json',**kwargs):
    ''' train the massage passing model '''
    regularize =True if lambda_reg>0.0001 else False
    cons_=['val','vale',
           'ovun1','ovun2','ovun3','ovun4',
           'ovun5','ovun6','ovun7','ovun8',
           'lp2','lp3',#'lp1',
           'cot1','cot2',
           'coa1','coa2','coa3','coa4',
           'pen1','pen2','pen3','pen4',
           'Depi','Depp',#'Desi','Devdw',
           #'bo1','bo2','bo3','bo4','bo5','bo6',
           #'rosi','ropi','ropp',
           'cutoff','hbtol','acut' ] #'val8','val9','val10' 
    if cons is None:
       cons = cons_
    else:
       cons = cons + cons_
       
    rn = MPNN(libfile=ffield,
              dataset=dataset, 
              dft='siesta',
              spec=spec,
              optword=optword,clip=clip,
              cons=cons,
              nn=nn,vdwnn=vdwnn,VdwFunction=VdwFunction,
              bo_layer=bo_layer,mf_layer=mf_layer,
              be_layer=be_layer,vdw_layer=vdw_layer,
              be_universal_nn=be_universal_nn,
              bo_universal_nn=bo_universal_nn,
              mf_universal_nn=mf_universal_nn,
              vdw_universal_nn=vdw_universal_nn,
              BOFunction=BOFunction,EnergyFunction=EnergyFunction,
              MessageFunction=MessageFunction,
              messages=messages,
              mpopt=mpopt,
              batch_size=batch,
              losFunc='n2',
              regularize_be=regularize,regularize_bo=regularize,regularize_mf=regularize,
              lambda_reg=lambda_reg,lambda_bd=lambda_bd,
              spv_be=spv_be,belo=belo,beup=beup,
              spv_bo=spv_bo,#boup=boup,
              spv_vdw=spv_vdw,vlo=vlo,vup=vup,
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

def train_nn(dataset=None,step=5000,batch=None,convergence=0.97,lossConvergence=1000.0,
               optword='nocoul',cons=None,clip={},
               spec=[],
               writelib=1000,nn=True,vdwnn=True,VdwFunction=1,
               bo_layer=[4,1],mf_layer=[9,2],be_layer=[6,1],vdw_layer=[6,1],
               be_universal_nn='all',bo_universal_nn='all',mf_universal_nn='all',
               vdw_universal_nn='all',
               BOFunction=0,EnergyFunction=3,MessageFunction=2,
               messages=1,mpopt=[1,1,1,1],
               spv_bo=False,#boc={},#boup={},
               spv_be=False,belo={},beup={},
               spv_vdw=False,vlo={},vup={},
               lambda_me=0.1,lambda_ang=0.02,lambda_bd=1000.0,
               weight={'others':10.0},
               spv_ang=False,
               lambda_reg=0.0001, # regularize=True,
               learning_rate=1.0e-4,
               ffield = 'ffield.json',**kwargs):
    ''' train the massage passing model '''
    regularize =True if lambda_reg>0.0001 else False
    cons_=['val','vale',
           'ovun1','ovun2','ovun3','ovun4',
           'ovun5','ovun6','ovun7','ovun8',
           'lp2','lp3',#'lp1',
           'cot1','cot2',
           'coa1','coa2','coa3','coa4',
           'pen1','pen2','pen3','pen4',
           'Depi','Depp',#'Desi','Devdw',
           #'bo1','bo2','bo3','bo4','bo5','bo6',
           #'rosi','ropi','ropp',
           'cutoff','hbtol','acut' ] #'val8','val9','val10' 
    if cons is None:
       cons = cons_
    else:
       cons = cons + cons_
       
    rn = ReaxFF_nn(libfile=ffield,
              dataset=dataset, 
              dft='siesta',
              spec=spec,
              optword=optword,clip=clip,
              cons=cons,
              nn=nn,vdwnn=vdwnn,VdwFunction=VdwFunction,
              bo_layer=bo_layer,mf_layer=mf_layer,
              be_layer=be_layer,vdw_layer=vdw_layer,
              be_universal_nn=be_universal_nn,
              bo_universal_nn=bo_universal_nn,
              mf_universal_nn=mf_universal_nn,
              vdw_universal_nn=vdw_universal_nn,
              BOFunction=BOFunction,EnergyFunction=EnergyFunction,
              MessageFunction=MessageFunction,
              messages=messages,
              mpopt=mpopt,
              batch_size=batch,
              losFunc='n2',
              regularize_be=regularize,regularize_bo=regularize,regularize_mf=regularize,
              lambda_reg=lambda_reg,lambda_bd=lambda_bd,
              spv_be=spv_be,belo=belo,beup=beup,
              spv_bo=spv_bo,#boup=boup,
              spv_vdw=spv_vdw,vlo=vlo,vup=vup,
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
       return -1,1,accMax,p,zpe,i

    rn.close()
    return loss,accu,accMax,p,zpe,i

if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [train,train_mpnn,train_nn])
   argh.dispatch(parser)

