#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
import json as js
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import sys
import argparse
from irff.reax_nn import ReaxFF_nn
#from irff.mpnn import MPNN
from irff.dingtalk import send_msg
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
from irff.ml.fluctuation import make_fluct

parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation')
parser.add_argument('--step',default=30000,type=int, help='training steps')
parser.add_argument('--lr',default=0.0001,type=float, help='learning rate')
parser.add_argument('--writelib',default=10000,type=int, help='every this step to write parameter file')
parser.add_argument('--print',default=10,type=int,help='every this step to print')
parser.add_argument('--loss',default=1.0,type=float,help='the convergence criter of loss')
parser.add_argument('--convergence',default=0.999,type=float,help='the convergence criter of accuracy')
parser.add_argument('--maxepoch',default=100,type=int,help='the max training epoch')
parser.add_argument('--batch',default=1000,type=int,help='the batch size of every configuration')
# parser.add_argument('--boc',default=1,type=int,help='F(f1,f2,f3,f4,f5)  optimize flag')
parser.add_argument('--bo',default=0,type=int,help='only bond-order parameters optimization flag')
#parser.add_argument('--om',default=1,type=int,help='optimize the molecular energy')
#parser.add_argument('--zpe',default=0,type=int,help='only optimize the zero point energy')
parser.add_argument('--vdw',default=0,type=int,help='only optimize the vdw energy')
args = parser.parse_args(sys.argv[1:])

dataset = {'h22-v':'aimd_h22/h22-v.traj',
         #   'dia-0':'data/dia-0.traj',
         #   'gp2-0':'data/gp2-0.traj',
         #   'gp2-1':'data/gp2-1.traj',
           }

getdata = ColData()
strucs = ['ch3no2',
          'nm2',
          'hmx1',
          'hmx2',
          'n2h4',
          'ch3nh2',
          'nh3',
          'c2h6',
          'c2h4',
          'no2',
          'o2n',
          'cn2',
          'co2',
          'hmx',
          'cl1',
          #'cl20',
          ]

batchs = {'others':args.batch}

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

check_emol(dataset)

clip = {'Desi':(100.0,745.0),
        #'ovun1':(0.0,0.0),'ovun5':(0.0, 0.0),'lp2':(0.0,0.01),
        'bo1':(-0.1,-0.01),'bo2':(4.0,12.0),
        'bo3':(-0.10,-0.01),'bo4':(4.0,12.0),
        'bo5':(-0.08,-0.02),'bo6':(4.0,12.0),
        'rosi_O':(1.25,1.39),'rosi_O-N':(1.25,1.55),
        'rvdw_C-N':(1.65,2.4),'rvdw_C':(1.75,2.34),'rvdw_H':(1.53,2.0),'rvdw_H-N':(1.6,2.2),
        'rvdw_N':(1.65,2.2),'rvdw_O-N':(1.75,2.35),'rvdw_H-O':(1.6,2.0),'rvdw_O':(1.7,2.2),
        'Devdw':(0.01,1.0),
        'alfa':(6.5,14.0),
        'vdw1':(0.8,8.0),'gammaw':(1.5,12.0),
        'theta0':(60,98.0),'theta0_N-C-N':(67.8,71.6), 'theta0_C-N-C':(67.8,71.6),####### Angle 
        'theta0_C-N-N':(60,72),'theta0_H-N-H':(60,72),                            ####### Angle
        'valang_N':(3.0,4.0),'valang_C':(3.9,4.1),'valang_H':(0.9,1.1),'valang_O':(2.0,4.0),
        'val1':(0.0,80.0),'val1_N-C-N':(0.0,40.0), 
        'val2':(0.0,4.0),'val2_C-C-H':(0.6,2.0),'val2_C-C-C':(0.4,2.0),'val2_H-C-H':(0.8,1.9),
        'val3':(0.5,5.0),'val4':(0.5,16.0),'val5':(1.0,6.0),'val6':(1.0,16.0),
        'val7':(1.0,16.0),'val8':(1.0,8.0),'val9':(0.5,5.0),'val10':(0.5,5.0),
        'lp2':(0.0,0.0),
        'pen1':(0.0,0.0),'pen2':(1.0,9.0),'pen3':(0.0,1.0),'pen4':(1.0,6.0),
        'coa1':(-0.0,0.0),'cot1':(-0.0,0.0),'cot2':(0.0,5.0),
        #'V1':(0.0,100.0),'V2':(0.0,100.0),'V3':(0.0,100.0),
        'tor3':(0,5.0),'tor4':(0,9.0),
        'Dehb_C-H-C':(0.0,0.0),'Dehb_C-H-N':(0.0,0.0),'Dehb_N-H-C':(0.0,0.0),
        'Dehb_N-H-N':(-5.0,0.0),'Dehb_C-H-C':(0.0,0.0) }

def train_reax(writelib=10000,print_step=100,
               mpopt = [1,1,1,1],
               step=50000,opt=None,om=1,lr=0.0001,
               convergence=0.97,lossConvergence=100.0,batch=50):
    cons = ['ovun2','ovun3','ovun4','ovun1',
            'ovun6','ovun7','ovun8','ovun5',
            'lp1',# 'lp2','boc1','boc2'
            'rohb','Dehb','hb1','hb2',
            'vdw1','gammaw','rvdw','alfa','Devdw',
            'theta0','valang','val','val1','val2','val3', # three-body
            'val4','val5','val7','valboc','valang','pen1','coa1',
            'tor1','V1','V2','V3','cot1',                 # four-body
            'tor2','tor3','tor4',
            'val','lp3','cutoff']                         # 不进行局域优化
    if args.bo:
       cons += ['vdw1','gammaw','rvdw','alfa','Devdw',
                'Devdw',
                #'rosi','ropi','ropp',
                #'bo1','bo2','bo3','bo4','bo5','bo6',
                'val8','val9','val10','theta0',
                'cot1','cot2',
                'coa1','coa2','coa3','coa4',
                'pen1','pen2','pen3','pen4',
                ] ### 
       mpopt = [0,0,0,0] # neural network for BO,MF,BE,VDW
    if args.vdw:
       cons = ['ovun2','ovun3','ovun4','ovun1',
               'ovun6','ovun7','ovun8','ovun5',
               'lp1',# 'lp2','boc1','boc2'
               'rohb','Dehb','hb1','hb2',
               'vdw1','gammaw','rvdw','alfa','Devdw',
               'theta0','valang','val','val1','val2','val3', # three-body
               'val4','val5','val7','valboc','valang','pen1','coa1',
               'tor1','V1','V2','V3','cot1',                 # four-body
               'tor2','tor3','tor4',
               'val','lp3','cutoff']    
       # mpopt = [0,0,0,0] # neural network for BO,MF,BE,VDW
    # opt=['atomic']

    rn = ReaxFF_nn(libfile='ffield.json',
                   dataset=dataset,            
                   weight={'cl20':2.0,'others':2.0},
                   optword='nocoul',mpopt=mpopt,
                   opt=opt,cons=cons,clip=clip,
                   regularize_mf=1,regularize_be=1,regularize_bias=0,
                   lambda_reg=0.005,lambda_bd=1000.0,lambda_me=0.001,
                   mf_layer=[9,1],be_layer=[9,0],
                   bdopt=['H-N'],   
                   mfopt=['N'],    
                   batch=args.batch,
                   fixrcbo=False,
                   losFunc='n2',  # n2, mse, huber,abs
                   convergence=convergence) 

    loss,accu,accMax,i =rn.run(learning_rate=lr,
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
       # with open('ffield.json','w') as fj:
       #     js.dump(j,fj,sort_keys=True,indent=2)

    p   = rn.p_
    ME  = rn.MolEnergy_
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
   ''' use commond like ./train_nn.py --s=50000 --l=0.0001 to run it
                     or ./train_nn.py to use default parameters 
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


