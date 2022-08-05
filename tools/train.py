#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import sys
import argh
import argparse
from irff.reax import ReaxFF
from irff.mpnn import MPNN
from irff.dingtalk import send_msg
import json as js
from irff.initCheck import Init_Check
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol


dataset  = {'s1':'s1.traj',
            's2':'s2.traj',}

# getdata = ColData()
# strucs  = [] 
batchs = {'others':50}

# for mol in strucs:
#     b = batchs[mol] if mol in batchs else batchs['others']
#     trajs = getdata(label=mol,batch=b)
#     dataset.update(trajs)

check_emol(dataset)


def train_reax(writelib=1000,print_step=10,
              step=500,opt=None,cons=None,lr=1.0e-4,
              convergence=0.97,lossConvergence=100.0,batch=50):
    rn = ReaxFF(libfile='ffield.json',
                dataset=dataset, 
                dft='qe',
                optword='nocoul',
                opt=None,
                cons=None,
                batch_size=batch,
                losFunc='n2',
                lambda_bd=10000.0,
                lambda_me=0.01,
                weight={'others':2.0},
                convergence=convergence,
                lossConvergence=lossConvergence) # Loss Functon can be n2,abs,mse,huber

    # GradientDescentOptimizer AdamOptimizer AdagradOptimizer RMSPropOptimizer
    loss,accu,accMax,i,zpe =rn.run(learning_rate=lr,
                      step=step,
                      print_step=print_step,
                      writelib=writelib) 
    libstep = int(i - i%writelib)

    if i==libstep:
       libstep = libstep - writelib
    if libstep<=0:
       ffd = 'ffield.json'
    else:
       ffd = 'ffield_' + str(libstep) +'.json'

    if loss==0.0 and accu==0.0:
       send_msg('-  Warning: the loss is NaN %s stop now!' %ffd)
       return 0.0,1.0,1.0,None,None,i
       # with open(ffd,'r') as fj:
       #     j = js.load(fj)
       #     ic = Init_Check(nanv=nanv)
       #     j['p'] = ic.auto(j['p'])
       #     ic.close()
       #with open('ffield.json','w') as fj:
       #     js.dump(j,fj,sort_keys=True,indent=2)

    p   = rn.p_
    ME = rn.MolEnergy_

    rn.close()
    return loss,accu,accMax,p,ME,i


def run(step=50000,convergence=0.99,loss_conver=20.0,writelib=1000,
        opt=None,
        cons = ['val','vale','lp3','cutoff'],
        print_step=10,
        lr=1.0e-4,
        maxcycle=20,
        batch=50):
    accu,loss = 0.0,100    
    c = 0

    while accu<convergence and c<maxcycle:   
          loss,accu,accMax,p,zpe,i=train_reax(step=step,opt=opt,cons=cons,
                                              convergence=convergence,lr=lr,
                                              writelib=writelib,print_step=print_step)
          if loss>loss_conver and accu>convergence:
             convergence = accu + 0.0003
          # system('cp ffield.json ffield%dt%de%df%d.json' %(int(loss),messages,ef,fm))
          c += 1

    if loss>0.0:
       send_msg('-  Convergence reached, the loss %7.4f and accuracy %7.4f.' %(loss,accu))


if __name__ == '__main__':
   ''' use commond like ./train.py --s=50000 --l=0.0001 to run it
                     or ./train.py to use default parameters 
       s:   steps
       l:   learning rate
   '''
   parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation')
   parser.add_argument('--step',default=50000,type=int, help='training steps')
   parser.add_argument('--lr',default=0.0001,type=float, help='learning rate')
   parser.add_argument('--writelib',default=1000,type=int, help='every this step to write parameter file')
   parser.add_argument('--print',default=10,type=int,help='every this step to print')
   parser.add_argument('--loss',default=30,type=float,help='the convergence criter of loss')
   parser.add_argument('--convergence',default=0.95,type=float,help='the convergence criter of accuracy')
   parser.add_argument('--maxcycle',default=1,type=int,help='the max training cycle')
   parser.add_argument('--batch',default=50,type=int,help='the batch size of every configuration')
   args = parser.parse_args(sys.argv[1:])

   run(step=args.step,
       convergence=args.convergence,
       loss_conver=args.loss,
       lr=args.lr,
       print_step=args.print,
       batch=args.batch,
       maxcycle=args.maxcycle)
