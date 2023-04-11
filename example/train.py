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
parser.add_argument('--print',default=10,type=int,help='every this step to print')
parser.add_argument('--loss',default=1.0,type=float,help='the convergence criter of loss')
parser.add_argument('--convergence',default=0.999,type=float,help='the convergence criter of accuracy')
parser.add_argument('--maxepoch',default=100,type=int,help='the max training epoch')
parser.add_argument('--batch',default=50,type=int,help='the batch size of every configuration')
parser.add_argument('--t',default=0,type=int,help='optimize the three boday term flag')
parser.add_argument('--h',default=0,type=int,help='optimize the hydrogen bond term flag')
# parser.add_argument('--bo',default=0,type=int,help='not optimize the bond term flag')
parser.add_argument('--zpe',default=0,type=int,help='optimize the zero point energy')
parser.add_argument('--vdw',default=1,type=int,help='optimize the vdw energy')
args = parser.parse_args(sys.argv[1:])

dataset = {'h22-v':'aimd_h22/h22-v.traj',
           #'hmx-s':'hmx.traj',
           }

getdata = ColData()
strucs = ['ch3no2',
          'nm2',
          'nmc',
          'hmx1',
          'hmx2',
          'n2h4',
          'ch3nh2',
          'nh3',
          'c2h6',
          'c2h4',
          'c3h8',
          'no2',
          'o2n',
          'cn2',
          'n22',
          'co2',
          #'h2o',
          'h2o2',
          'ch4w2',
          'oh2',
          'hmx',
          'hmxc',
          'cl1',
          'cbd',
          #'cl20',
          'fox',
          ]

batchs = {'others':args.batch}

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

check_emol(dataset)

clip = {'Desi':(125.0,750.0),
        'bo1':(-0.09,-0.02),'bo2':(4.0,9.15),
        'bo3':(-0.09,-0.02),'bo4':(4.0,9.15),
        'bo5':(-0.0765,-0.0275),'bo6':(4.0,9.15),
      #   'bo1_H-O':(-0.072,-0.01),'bo2_H-O':(4.0,7.0),'bo3_H-O':(-0.072,-0.01),'bo4_H-O':(4.0,7.0),
      #   'bo5_H-O':(-0.05,-0.03),'bo6_H-O':(4.0,5.5),
      #   'bo3_O-O':(-0.075,-0.01),'bo4_O-O':(4.0,8.0),
      #   'bo1_C-C':(-0.092,-0.01),'bo2_C-C':(4.0,9.3),
        'rosi_C':(1.35,1.546),
        'ropi_C-N':(1.0,1.3),
        'rvdw_C-N':(1.67,2.4),'rvdw_C':(1.755,2.34),'rvdw_H':(1.53,2.0),'rvdw_H-N':(1.6,2.2),
        'rvdw_N':(1.71,2.2),'rvdw_O-N':(1.755,2.35),'rvdw_H-O':(1.6,2.0),'rvdw_O':(1.725,2.2),
        'rvdw_C-O':(1.79,2.4),
        'Devdw':(0.01,1.0),
        'alfa':(6.5,14.0),
        'vdw1':(0.8,8.0),'gammaw':(1.5,12.0),
        'valang_N':(2.0,4.0),'valang_C':(3.2,4.1),'valang_H':(0.8,1.1),'valang_O':(1.5,4.0),
        'val1':(0.0,80.0),'val1_C-C-O':(0.0,0.0),
        'val2':(0.0,4.0)}

cons = ['ovun2','ovun3','ovun4','ovun1',
        'ovun6','ovun7','ovun8','ovun5',
        # 'rosi','ropi','ropp',
        'lp2','lp1','vale',
        #'theta0',
        #'val9','valang',
        #'val10','val8',
        'valboc','cot1','cot2','coa1','coa2','coa3','coa4',
        'pen1','pen2','pen3','pen4',
        #'vdw1','gammaw','rvdw','alfa','Devdw',
        #'rohb','Dehb','hb1','hb2',
        'val','lp3','cutoff']    # 不进行局域优化
if not args.t:
   cons += ['theta0','val9','valang','val10','val8']
if not args.vdw:
   cons += ['vdw1','gammaw','rvdw','alfa','Devdw']
if not args.h:
   cons += ['rohb','Dehb','hb1','hb2']

if args.zpe:
   cons += ['Depi','Depp','Desi',
            'rosi','ropi','ropp',
            'bo1','bo2','bo3','bo4','bo5','bo6',
            'vdw1','gammaw','rvdw','alfa','Devdw',
            'val1','val2','val3','val4','val5','val6','val7',
            'val8','val9','val10','theta0',
            'valboc','valang','vale',
            'cot1','cot2',
            'coa1','coa2','coa3','coa4',
            'pen1','pen2','pen3','pen4',
            'lp2',
            'tor2','tor3','tor4','tor1',
            'V1','V2','V3'
            ] ### 
   mpopt = [0,0,0,0] # neural network for BO,MF,BE,VDW

def train_reax(writelib=10000,print_step=100,
               mpopt = [1,1,1,1],cons=cons,
               step=50000,opt=None,om=1,lr=0.001,
               convergence=0.97,lossConvergence=100.0,batch=50):
    ''' train '''
    rn = MPNN(libfile='ffield.json',
              dataset=dataset,            
              spv_ang=False,lambda_ang=0.001,
              spv_bo=None,
              weight={'hmx-r':20.0,'h2o2':2.0,'others':2.0},
              optword='nocoul',mpopt=mpopt,
              opt=opt,cons=cons,clip=clip,
              regularize_mf=1,regularize_be=1,regularize_bias=1,
              lambda_reg=0.005,lambda_bd=1000.0,lambda_me=0.001,
              mf_layer=[9,1],be_layer=[9,1],
              EnergyFunction=1,MessageFunction=3,
              mf_universal_nn=None,be_universal_nn=None,#['N-N','C-O'],
              bdopt=None,    # ['H-N'], 
              mfopt=None,    # ['N'], 
              batch_size=batch,
              fixrcbo=False,
              losFunc='n2',  # n2, mse, huber,abs
              convergence=convergence) 

    loss,accu,accMax,i,zpe =rn.run(learning_rate=lr,
                      step=step,
                      print_step=print_step,
                      writelib=writelib,
                      method='AdamOptimizer')
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
       # with open(ffd,'r') as fj:
       #     j = js.load(fj)
       #     ic = Init_Check(nanv=nanv)
       #     j['p'] = ic.auto(j['p'])
       #     ic.close()
       #with open('ffield.json','w') as fj:
       #     js.dump(j,fj,sort_keys=True,indent=2)
    p   = rn.p_
    ME  = rn.MolEnergy_

    rn.close()
    return loss,accu,accMax,p,ME,i


def run(step=50000,convergence=0.99,loss_conver=20.0,writelib=1000,
        opt=None,om=1,
        print_step=10,
        lr=1.0e-4,
        maxepoch=20,
        batch=50):

    accu,loss = 0.0,100    
    epoch = 0

    while accu<convergence and epoch<maxepoch:   
          loss,accu,accMax,p,zpe,i=train_reax(step=step,opt=opt,om=om,
                                              convergence=convergence,lr=lr,
                                              writelib=writelib,print_step=print_step)
          if loss>loss_conver and accu>convergence:
             convergence = accu + 0.0003
          # if loss==9999999.9 and accu==-1.0:
          #   send_msg('-  The loss is NaN, please check parameters!')
          #   return
          # system('cp ffield.json ffield%dt%de%df%d.json' %(int(loss),messages,ef,fm))
          epoch += 1

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
       maxepoch=args.maxepoch,
       writelib=args.writelib)


