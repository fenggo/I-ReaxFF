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
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
from irff.tools.fluctuation import bo_fluct,make_fluct

parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation')
parser.add_argument('--step',default=50000,type=int, help='training steps')
parser.add_argument('--lr',default=0.0001,type=float, help='learning rate')
parser.add_argument('--writelib',default=1000,type=int, help='every this step to write parameter file')
parser.add_argument('--print',default=10,type=int,help='every this step to print')
parser.add_argument('--loss',default=1.0,type=float,help='the convergence criter of loss')
parser.add_argument('--convergence',default=0.999,type=float,help='the convergence criter of accuracy')
parser.add_argument('--maxcycle',default=1,type=int,help='the max training cycle')
parser.add_argument('--batch',default=50,type=int,help='the batch size of every configuration')
parser.add_argument('--boc',default=1,type=int,help='F(f1,f2,f3,f4,f5)  optimize flag')
parser.add_argument('--bo',default=1,type=int,help='bond-order parameters optimize flag')
parser.add_argument('--om',default=1,type=int,help='optimize the molecular energy')
args = parser.parse_args(sys.argv[1:])

dataset = {#'gpu-0':'data/gpu-0.traj',
           #'gpu-1':'data/gpu-1.traj'
           'gpd-0':'data/gpd-0.traj',
           'gpd-1':'data/gpd-1.traj',
           #'gpd-2':'data/gpd-2.traj',
           }
getdata = ColData()
strucs = [#'c2',
          # 'c2c6',
          # 'c3',
          #'c32',
          'c4',
          # 'c5',
          'c6',
          # 'c62',
          #'c8',
          #'c10',
          # 'c12',
          # 'c14',
          # 'c16',
          # 'c18',
          'dia',
          'gpu',
          'gp',
          'gp-1',
          # 'gpd',
          # 'gpe',
          ]
batchs = {'others':50}

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

check_emol(dataset)

clip = {'be1':(-1.0,1.0),'be2':(0.01,0.9),
        'bo1':(-0.4,-0.01),'bo2':(4.0,10.0),
        'bo3':(-0.4,-0.03),'bo4':(4.0,16.0),
        'bo5':(-0.4,-0.04),'bo6':(4.0,16.0),
        'rosi':(1.13,1.4),
        'Desi':(100.0,360.0),'Depi':(60.0,160.0),'Depp':(60.0,150.0),
        'Devdw':(0.03,0.2),'alfa':(9.0,14.0),'rvdw':(1.75,2.2),'vdw1':(1.5,8.0),
        'gammaw':(2.5,12.0),
        'ovun1':(0.0,0.2),'ovun3':(0.01,35.0),'ovun4':(0.5,10.0),
        'ovun5':(0.0,50.0),'ovun7':(0.5,16.0),'ovun8':(1.0,16.0),
        'lp1':(6.0,18.0),'lp2':(0.0,0.01),
        'theta0':(60,86.0),'val1':(10.0,100.0),'val2':(0.22,2.0),'val3':(0.1,5.0),'val5':(1.0,16.0),
        'val8':(0.5,4.0),'val9':(0.5,3.0),'val10':(0.5,3.0),
        'pen1':(9.0,11.0),'pen2':(1.0,9.0),'pen3':(0.0,1.0),'pen4':(1.0,6.0),
        'coa1':(-1.0,0.0),'cot1':(-1.0,0.0),'cot2':(0.0,5.0),
        'V1':(0.0,30.0),'V2':(0.0,30.0),'V3':(0.0,30.0)}

def train_reaxff_nn(writelib=10000,print_step=100,
              step=50000,opt=None,om=1,lr=1.0e-4,
              convergence=0.97,lossConvergence=100.0,batch=50):
    cons=['val','vale',
          #'ovun1','ovun2','ovun3','ovun4',
          #'ovun5','ovun6','ovun7','ovun8',
          #'lp2', # 'lp1',
          'lp3',
          'cot1','cot2',
          'coa1','coa2','coa3','coa4',
          'pen1','pen2','pen3','pen4',
          'hbtol',
          'Depi','Depp','cutoff','acut']
    if not args.boc:
       cons += ['valboc','boc1','boc2','boc3','boc4','boc5']
    if not args.bo:
       cons += ['bo1','bo2','bo3','bo4','bo5','bo6','rosi','ropi','ropp']
    

    belo,beup,vlo,vup = make_fluct(fluct=0.3,bond=['C-C'],csv='fluct') 
    # opt=['atomic']

    rn = MPNN(libfile='ffield.json',
              dataset=dataset,            
              spv_ang=False,lambda_ang=0.02,
              spv_vdw=False,spv_bo=False,spv_be=False,
              beup=beup,#vlo=vlo,#vup=vup,
              weight={'c6':2.0,'others':2.0},
              optword='nocoul',
              opt=opt,optmol=om,cons=cons,clip=clip,
              regularize_mf=1,regularize_be=1,
              lambda_reg=0.002,lambda_bd=10000.0,lambda_me=0.05,
              mf_layer=[6,1],be_layer=[6,0],
              BOFunction=0,EnergyFunction=1,MessageFunction=4,
              mf_univeral_nn=None,
              vdwnn=False,vdw_layer=[9,1],
              bdopt=None,    # ['N-N'],# ['H-H','O-O','C-C','C-H','N-N','C-N','C-O'],
              mfopt=None,    # ['N'],# ['H','O','C','N'],
              batch_size=batch,
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
    ME = rn.MolEnergy_

    rn.close()
    return loss,accu,accMax,p,ME,i


def run(step=50000,convergence=0.99,loss_conver=20.0,writelib=1000,
        opt=None,om=1,
        print_step=10,
        lr=1.0e-4,
        maxcycle=20,
        batch=50):
    accu,loss = 0.0,100    
    c = 0

    while accu<convergence and c<maxcycle:   
          loss,accu,accMax,p,zpe,i=train_reaxff_nn(step=step,opt=opt,om=om,
                                              convergence=convergence,lr=lr,
                                              writelib=writelib,print_step=print_step)
          if loss>loss_conver and accu>convergence:
             convergence = accu + 0.0003
          # if loss==9999999.9 and accu==-1.0:
          #   send_msg('-  The loss is NaN, please check parameters!')
          #   return
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

   run(step=args.step,
       convergence=args.convergence,
       loss_conver=args.loss,
       om=args.om,
       lr=args.lr,
       print_step=args.print,
       batch=args.batch,
       maxcycle=args.maxcycle,
       writelib=args.writelib)
