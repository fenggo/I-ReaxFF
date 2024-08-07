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
from irff.ml.fluctuation import morse

parser = argparse.ArgumentParser(description='nohup ./train.py --v=1 --h=0> py.log 2>&1 &')
parser.add_argument('--step',default=30000,type=int, help='training steps')
parser.add_argument('--lr',default=0.0001,type=float, help='learning rate')
parser.add_argument('--writelib',default=1000,type=int, help='every this step to write parameter file')
parser.add_argument('--pr',default=10,type=int,help='every this step to print')
parser.add_argument('--pi',default=1,type=int,help='regularize BO pi component')
parser.add_argument('--loss',default=1.0,type=float,help='the convergence criter of loss')
parser.add_argument('--convergence',default=0.999,type=float,help='the convergence criter of accuracy')
parser.add_argument('--maxepoch',default=100,type=int,help='the max training epoch')
parser.add_argument('--batch',default=50,type=int,help='the batch size of every configuration')
parser.add_argument('--t',default=0,type=int,help='optimize the three-boday term flag')
parser.add_argument('--h',default=0,type=int,help='optimize the hydrogen bond term flag')
parser.add_argument('--a',default=0,type=int,help='surpvise the angle term flag')
parser.add_argument('--f',default=0,type=int,help='optimize the four-boday term flag')
parser.add_argument('--bo',default=1,type=int,help='optimize the bond term flag')
parser.add_argument('--zpe',default=0,type=int,help='optimize the zero point energy')
parser.add_argument('--vdw',default=1,type=int,help='optimize the vdw energy')
args = parser.parse_args(sys.argv[1:])


getdata = ColData()
 
strucs = ['h2o2','ch4w2','h2o16']  


weight={'h2o2':2.0,
        'others':2.0}

dataset = {'h22-v':'aimd_h22/h22-v.traj',
           }
batchs  = {'h2o16':50,'others':args.batch}
for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

clip = {'Desi':(125.0,750.0),
        'bo1':(-0.077,-0.02),'bo2':(5.0,8.77),
        'bo3':(-0.077,-0.02),'bo4':(5.0,8.77), 
        'bo5':(-0.067,-0.028),'bo6':(5.0,8.77), 
        'rosi_C':(1.2,1.50),'rosi_H-N':(0.8,1.11),
        'rosi_N':(1.08,1.52),
        'rosi_C-H':(1.0,1.15),'ropi_C-H':(1.0,1.15),
        'rosi_C-N':(1.0,1.41),'ropi_C-N':(1.0,1.21),'ropp_C-N':(1.0,1.21),
        'rosi_H-O':(0.82,1.11),'ropi_H-O':(0.81,1.11),'ropp_H-O':(0.81,1.11),
        'rvdw_C-N':(1.69,2.4),'rvdw_C':(1.757,2.34),'rvdw_H':(1.53,2.0),'rvdw_H-N':(1.6,2.2),
        'rvdw_N':(1.74,2.2),'rvdw_O-N':(1.775,2.35),'rvdw_H-O':(1.6,2.0),'rvdw_O':(1.73,2.2),
        'rvdw_C-O':(1.79,2.4),
        'Devdw':(0.01,1.0),
        'alfa':(6.0,16.0),
        'vdw1':(0.8,8.0),'gammaw':(1.5,12.0),'gammaw_C':(5.09,12.0),'gammaw_H':(4.25,12.0),'gammaw_N':(7.025,12.0),
        'valang_N':(0.0,4.8),'valang_C':(0.0,4.8),'valang_H':(0.0,2.7),'valang_O':(0.0,4.8),
        'val1':(0.0,60.0),'val1_C-C-O':(0.0,0.0),'val1_H-N-O':(0.0,0.0),'val1_H-O-N':(0.0,0.0),
        'val2':(0.5,2.0),'val4':(0.5,2.0),'val5':(0.0,5.6),
        'tor2':(1.0,6.6),'tor4':(0.0004,2.0),
        'V2':(-42.0,64.0),#'V3_N-C-N-N':(0.0,0.0),
        'V1_C-C-C-H':(0.0,0.0),'V2_C-C-C-H':(0.0,0.0),'V3_C-C-C-H':(0.0,0.0),
        'V1_N-C-C-N':(0.0,0.0),'V3_N-C-C-N':(0.0,0.0),
        'rohb':(1.68,4.0),'hb1':(-1.6,9.0),'hb2':(0.0,26.0),
        'acut':(0.0001,0.01)}

cons = ['ovun2','ovun3','ovun4','ovun1',
        'ovun6','ovun7','ovun8','ovun5',
        # 'rosi','ropi','ropp',
        'lp2','lp1',
        #'theta0',
        #'val9','valang',
        #'val10','val8',
        'valboc','cot1','cot2','coa1','coa2','coa3','coa4',
        'pen1','pen2','pen3','pen4',
        #'vdw1','gammaw','rvdw','alfa','Devdw',
        #'rohb','Dehb','hb1','hb2',
        'val1_C-C-O','val1_H-N-O','val1_H-O-N',
        'V1_X-N-N-X','V2_X-N-N-X','V3_X-N-N-X',
        'V1_X-C-N-X','V2_X-C-N-X','V3_X-C-N-X',
        #'V1_N-C-N-N','V2_N-C-N-N','V3_N-C-N-N',
        #'V1_C-C-N-N','V2_C-C-N-N','V3_C-C-N-N',
        'val','lp3','cutoff',#'acut',
        ]    # 不进行局域优化

if not args.f:
   cons += [ 'tor1','tor2','tor3','tor4',      # Four-body
             'V1','V2','V3',]
if not args.t:
   cons += ['val1','val2','val3','val6','val7'] #
if not args.a:
   #cons += ['theta0']
   cons += ['theta0','val9','val10','val8','vale','valang','val4','val5'] # 
if not args.vdw:
   cons += ['vdw1','gammaw','rvdw','alfa','Devdw']
if not args.h:
   cons += ['rohb','Dehb','hb1','hb2']

if not args.bo:
   cons += ['Depi','Depp','Desi',
            'rosi','ropi','ropp',
            'bo1','bo2','bo3','bo4','bo5','bo6',
            'vdw1','gammaw','rvdw','alfa','Devdw',
            ] ### 
   mpopt = [0,0,0,0] # neural network for BO,MF,BE,VDW

       
if __name__ == '__main__':
   ''' train ''' 
   rn = MPNN(libfile='ffield.json',
             dataset=dataset,            
             weight=weight,
             cons=cons,clip=clip,
             regularize_mf=1,regularize_be=1,regularize_bias=1,
             lambda_reg=0.001,lambda_bd=100.0,lambda_me=0.001,lambda_pi=0.003,
             mf_layer=[9,1],be_layer=[9,1],
             EnergyFunction=1,MessageFunction=3,
             mf_universal_nn=None,be_universal_nn=['C-H','O-O'], # share the same weight and bias matrix
             messages=1,
             bdopt=None,    # ['H-N'], 
             mfopt=None,    # ['N'], 
             batch_size=args.batch,
             fixrcbo=False,
             losFunc='n2',  # n2, mse, huber,abs
             convergence=0.95) 

   rn.run(learning_rate=args.lr,
          step=args.step,
          print_step=args.pr,
          writelib=args.writelib,
          method='AdamOptimizer')
