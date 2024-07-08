#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
import json as js
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import sys
import argparse
from irff.reax import ReaxFF
from irff.mpnn import MPNN
from irff.dingtalk import send_msg
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
from irff.ml.train import train

'''
Optimze the parameters of ReaxFF with gradient based optimizer
'''

parser = argparse.ArgumentParser(description='nohup ./train.py --s=10000 > py.log 2>&1 &')
parser.add_argument('--step',default=10000,type=int, help='training steps')
parser.add_argument('--lr',default=0.0001,type=float, help='learning rate')
parser.add_argument('--writelib',default=1000,type=int, help='every this step to write parameter file')
parser.add_argument('--pr',default=10,type=int,help='every this step to print')
parser.add_argument('--pi',default=0,type=int,help='regularize BO pi component')
parser.add_argument('--loss',default=1.0,type=float,help='the convergence criter of loss')
parser.add_argument('--convergence',default=0.999,type=float,help='the convergence criter of accuracy')
parser.add_argument('--maxepoch',default=100,type=int,help='the max training epoch')
parser.add_argument('--batch',default=50,type=int,help='the batch size of every configuration')
parser.add_argument('--zpe',default=1,type=int,help='update the zero point energy')
args = parser.parse_args(sys.argv[1:])


getdata = ColData()
dataset = {}
strucs  = ['tkx','tkx2']
weight  ={'nomb':0.0,'others':2.0}


batchs  = {'others':args.batch}
batch   = args.batch

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)
# check_emol(dataset)

clip = {'Desi':(125.0,748.0),
        'bo1':(-0.089,-0.02),'bo2':(5.0,9.9),
        'bo3':(-0.089,-0.02),'bo4':(5.0,9.9), 
        'bo5':(-0.089,-0.02),'bo6':(5.0,9.9), 
        'rosi_C':(1.2,1.499),'rosi_H-N':(0.8,1.11),
        'rosi_N':(1.08,1.52),
        'rosi_C-H':(1.0,1.25),'ropi_C-H':(1.0,1.2),
        'rosi_C-N':(1.0,1.448),'ropi_C-N':(1.0,1.3),'ropp_C-N':(1.0,1.3),
        'rosi_H-O':(0.82,1.2),'ropi_H-O':(0.81,1.2),'ropp_H-O':(0.81,1.2),
        'ovun1':(0.0,0.12), #'ovun1_N-N':(0.0012,0.177),'ovun1_O-O':(0.0012,0.177),
        'ovun3':(0.0,9.0),'ovun4':(-1.0,2.4),
        'ovun2':(-1.0,0),
        'rvdw':(1.48,2.1),# 'rvdw_C-O':(1.65,2.15),'rvdw_C':(1.65,2.161),'rvdw_O':(1.65,2.09),
        'Devdw':(0.001,0.3),'Devdw_O-N':(0.0001,0.251),'Devdw_C-O':(0.0001,0.1),'Devdw_N':(0.0001,0.1),
        'Devdw_C-H':(0.0001,0.321),'Devdw_C-N':(0.0001,0.154),
        'alfa':(5.0,16.0),
        'vdw1':(0.8,8.0),'gammaw':(1.5,12.0),'gammaw_C':(4.0,12.0),'gammaw_H':(3.5,12.0),'gammaw_N':(6.0,12.0),
        'valang_N':(0.0,4.8),'valang_C':(0.0,4.8),'valang_H':(0.0,2.7),'valang_O':(0.0,4.8),
        'val1':(0.0,62.0),'val1_C-C-O':(0.0,0.0),'val1_C-N-N':(0.0,48),
        'val1_C-N-C':(0.0,48),'val1_N-C-N':(0.0,27),
        'val2':(0.3,1.9),'val3':(0.0,6.3),
        'val4':(0.1,4.68),'val4_C-N-N':(0.1,2.24),'val4_H-C-H':(0.1,0.24),
        'val5':(0.3,3.6),'val7':(0.5,11.0),
        'tor1':(-9.0,-0.02),'tor2':(1.0,6.6),'tor4':(0.001,2.0),
        'V2':(-42.0,48.0),#'V3_N-C-N-N':(0.0,0.0),
        'V1_C-C-C-H':(0.0,0.0),'V3_C-C-C-H':(0.0,0.0),
        'V1_N-C-C-N':(0.0,0.0),'V3_N-C-C-N':(0.0,0.0),
        'hb1':(3.05,3.06),'hb2':(19.16,19.1628),
        'Dehb':(-4.0,0.0),'Dehb_C-H-O':(-3.5,-0.8),
        'rohb':(1.73,2.26),
        'acut':(0.0001,0.1)}

parameters = ['boc1','boc2','boc3','boc4','boc5',
              'rosi','ropi','ropp',
              'Desi','Depi','Depp',
              'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6',
              'atomic']


if args.pi:
    pi_clip={'C-C-C':[(7.8,9.1,1.63,1.81)],
            'C-C-H':[(8.0,8.8,1.63,1.8),(10.0,13,0,0.6)],
            #'H-C-H':[(8.0,8.8,1.65,1.78),(10,12,0.24,0.5)],
            'O-N-N':[(7.2,9,0.84,1.2)],
            'O-N-O':[(7.2,9,0.84,1.2)],
            'C-N-O':[(7.2,9,0.84,1.2)],
            #'H-N-H':[(7.8,8.4,0.0,0.55)],
            'H-O-H':[(3,5,0.57,0.77)],
            'C-C-N':[(7.8,8.6,1.61,1.78),(8.7,9.3,1.61,1.78),(10,12,0.3,0.6)],
            'N-C-N':[(7.8,8.4,1.61,1.78)],
            }
else:
    pi_clip= False

bo_clip = {'C-H':[(1.6,7.5,11,1.8,11,0.0,0.01)],
           'O-O':[(1.7,2.0,11,2.0,11,0,0)],
           'C-O':[(1.9,10,19,2.5,5,0,0)],
           'O-N':[(1.75,2.5,9,7.4,9,0.0,0.0)],
           'H-N':[(1.5,2.0,11,7.0,11,0.0,0.01)],
           }
# bo_clip = None
       
be_universal_nn = None #['C-H','O-O']

if __name__ == '__main__':
   ''' train ''' 
   rx = ReaxFF(libfile='ffield.json',
              dataset=dataset, 
              energy_term={'ecoul':False},
              opt=parameters,
              clip=clip,
              batch_size=batch,
              losFunc='n2',
              lambda_bd=100.0,
              lambda_me=0.001,
              weight={'tkx':3.0,'others':2.0},
              convergence=1.0,
              lossConvergence=0.0)  # Loss Functon can be n2,abs,mse,huber
   rx.initialize()
   rx.session(learning_rate=0.0001, method='AdamOptimizer') 
   if args.zpe:
      rx.update(reset_emol=True)      
      rx.get_zpe()
      rx.update() 

   #evdw = 0.06
   #while evdw >0.01:
   #      evdw -= 0.001
   #      rn.update(p={'Devdw_N':evdw})
   rx.run(learning_rate=args.lr,
          step=args.step,
          print_step=args.pr,
          writelib=args.writelib,
          method='AdamOptimizer',
          close_session=True)
