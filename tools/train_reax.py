#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
import json as js
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import sys
import argh
import argparse
from irff.reax import ReaxFF
from irff.mpnn import MPNN
from irff.dingtalk import send_msg
from irff.initCheck import Init_Check
from irff.data.ColData import ColRawData
from irff.dft.CheckEmol import check_emol
from irff.tools.fluctuation import bo_fluct,make_fluct

parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation')
parser.add_argument('--step',default=50000,type=int, help='training steps')
parser.add_argument('--lr',default=0.0001,type=float, help='learning rate')
parser.add_argument('--writelib',default=1000,type=int, help='every this step to write parameter file')
parser.add_argument('--print',default=10,type=int,help='every this step to print')
parser.add_argument('--loss',default=300,type=float,help='the convergence criter of loss')
parser.add_argument('--convergence',default=0.95,type=float,help='the convergence criter of accuracy')
parser.add_argument('--maxcycle',default=1,type=int,help='the max training cycle')
parser.add_argument('--batch',default=50,type=int,help='the batch size of every configuration')
parser.add_argument('--boc',default=1,type=int,help='F(f1,f2,f3,f4,f5)  optimize flag')
parser.add_argument('--bo',default=1,type=int,help='bond-order parameters optimize flag')
args = parser.parse_args(sys.argv[1:])
getdata = ColRawData()

rawdata = {'Fe2O3_1':'mdout/Fe2O3_1.traj',
           'Fe3O4_1':'mdout/Fe3O4_1.traj',
           'Fe2O3_1':'mdout/Fe2O3_1.traj',
           'H2O_1':'mdout/H2O_1.traj',
           'H2_1':'mdout/H2_1.traj',}
check_emol(rawdata)
dataset = getdata(batch=100,rawdata=rawdata)

data  = dataset.keys()
data_ = random.sample(data,ntraj)
dataset_ = {}

for key in data_:
    dataset_[key] = dataset[key]


def train_reax(writelib=10000,print_step=100,
              step=50000,opt=None,lr=1.0e-4,
              convergence=0.97,lossConvergence=100.0,batch=50):
    cons = ['val','vale','lp3','cutoff','boc1']
    if not args.boc:
       cons += ['valboc','boc1','boc2','boc3','boc4','boc5']
    if not args.bo:
       cons += ['bo1','bo2','bo3','bo4','bo5','bo6','rosi','ropi','ropp']

    # belo,beup,vlo,vup = make_fluct(fluct=0.2,bond=['C-C'],csv='fluct')

    rn = ReaxFF(libfile='ffield.json',
                dataset=dataset,
                dft='siesta',
                optword='nocoul',
                opt=None,cons=cons,
                clip={'boc1':(10.0,30.0),'boc2':(3.0,9.0),'boc3':(0.1,19.9),'boc4':(0.5,9.9),
                      'bo1':(-0.4,-0.02),'bo2':(4.0,15.0),
                      'bo3':(-0.4,-0.02),'bo4':(4.0,16.0),
                      'bo5':(-0.4,-0.02),'bo6':(4.0,16.0),
                      'Desi':(10.0,160.0),'Depi':(30.0,90.0),'Depp':(30.0,100.0),
                      'be1':(0.01,0.5),'be2':(0.01,0.2),
                      'ovun1':(0.1,0.9),'ovun3':(0.01,30.0),'ovun4':(0.5,10.0),
                      'ovun5':(0.1,50.0),'ovun7':(0.1,20.0),'ovun8':(1.0,18.0),
                      'lp1':(5.0,18.0),'lp2':(0.0,20.0),
                      'Devdw':(0.025,0.2),'alfa':(10.0,14.0),'rvdw':(1.9,2.3),'vdw1':(1.5,2.0),'gammaw':(2.5,5.0),
                      'val2':(0.21,2.0),'val3':(0.1,5.0),
                      'val9':(1.0,2.0),
                      'pen1':(9.0,11.0),'pen2':(1.0,9.0),'pen3':(0.0,1.0),'pen4':(1.0,6.0),
                      'coa1':(-1.0,0.0),'cot1':(-1.0,0.0),
                      'V2':(0.0,10.0),'V3':(0.0,10.0),'V1':(0.0,10.0)},
                batch_size=batch,
                losFunc='n2',
                spv_vdw=False,# vup=vup,#vlo=vlo,
                lambda_bd=10000.0,
                lambda_me=0.01,
                atol=0.002,hbtol=0.002,
                weight={'al4':2.0,'others':2.0},
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

    if loss==9999999.9 and accu==-1.0:
       send_msg('-  Warning: the loss is NaN, please check parameters!')
       return 0.0,1.0,1.0,None,None,i

    p   = rn.p_
    ME = rn.MolEnergy_

    rn.close()
    return loss,accu,accMax,p,ME,i


def run(step=50000,convergence=0.99,loss_conver=20.0,writelib=1000,
        opt=None,
        print_step=10,
        lr=1.0e-4,
        maxcycle=20,
        batch=50):
    accu,loss = 0.0,100
    c = 0

    while accu<convergence and c<maxcycle:
          loss,accu,accMax,p,zpe,i=train_reax(step=step,opt=opt,
                                              convergence=convergence,lr=lr,
                                              writelib=writelib,print_step=print_step)
          if loss>loss_conver and accu>convergence:
             convergence = accu + 0.0003
          c += 1

    if loss>0.0:
       send_msg('-  Convergence reached, the loss %7.4f and accuracy %7.4f.' %(loss,accu))


if __name__ == '__main__':
   ''' use commond like ./train.py --s=50000 --l=0.0001 to run it
                     or ./train.py to use default parameters
       s:   steps
       l:   learning rate
   '''

   run(step=args.step,
       convergence=args.convergence,
       loss_conver=args.loss,
       lr=args.lr,
       print_step=args.print,
       batch=args.batch,
       maxcycle=args.maxcycle,
       writelib=args.writelib)
