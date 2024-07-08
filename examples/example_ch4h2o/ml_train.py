.......#!/usr/bin/env python
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import sys
import argparse
import matplotlib
import json as js
matplotlib.use('Agg')
from irff.reax import ReaxFF
from irff.mpnn import MPNN
from irff.dingtalk import send_msg
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
from irff.ml.train import train

parser = argparse.ArgumentParser(description='nohup ./train.py --v=1 --h=0> py.log 2>&1 &')
parser.add_argument('--step',default=30000,type=int, help='training steps')
parser.add_argument('--estep',default=0,type=int, help='evaluate training steps')
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
parser.add_argument('--bo',default=0,type=int,help='optimize the bond term flag')
parser.add_argument('--zpe',default=0,type=int,help='optimize the zero point energy')
parser.add_argument('--vdw',default=1,type=int,help='optimize the vdw energy')
parser.add_argument('--i',default=10,type=int,help='the population size generated in the begining')
args = parser.parse_args(sys.argv[1:])


getdata = ColData()
 
strucs = ['h2o2','ch4w2','h2o16']  


weight={'h2o2':2.0,
        'others':2.0}


dataset = {'h22-v':'aimd_h22/h22-v.traj',
           }
batchs  = {'others':args.batch}
for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

clip = {'Desi':(125.0,750.0),
        'bo1':(-0.08,-0.02),'bo2':(5.0,9.0),
        'bo3':(-0.08,-0.02),'bo4':(5.0,9.0), 
        'bo5':(-0.067,-0.028),'bo6':(5.0,9.0), 
        'rosi_C':(1.2,1.50),'rosi_H-N':(0.8,1.11),
        'rosi_N':(1.08,1.52),
        'rosi_C-H':(1.0,1.15),'ropi_C-H':(1.0,1.15),
        'rosi_C-N':(1.0,1.41),'ropi_C-N':(1.0,1.21),'ropp_C-N':(1.0,1.21),
        'rosi_H-O':(0.82,1.11),'ropi_H-O':(0.81,1.11),'ropp_H-O':(0.81,1.11),
        'rvdw':(1.5,2.01),# 'rvdw_C-O':(1.65,2.15),'rvdw_C':(1.65,2.161),'rvdw_O':(1.65,2.09),
        'Devdw':(0.001,0.3),'Devdw_O-N':(0.0001,0.251),'Devdw_C-O':(0.0001,0.097),'Devdw_N':(0.0001,0.1),
        'Devdw_C-H':(0.0001,0.321),'Devdw_C-N':(0.0001,0.154),
        'alfa':(6.0,16.0),
        'vdw1':(0.8,8.0),'gammaw':(1.5,12.0),'gammaw_C':(4.0,12.0),'gammaw_H':(3.5,12.0),'gammaw_N':(6.0,12.0),
        'valang_N':(0.0,4.8),'valang_C':(0.0,4.8),'valang_H':(0.0,2.7),'valang_O':(0.0,4.8),
        'val1':(0.0,79.0),'val1_C-C-O':(0.0,0.0),'val1_H-N-O':(0.0,0.0),'val1_H-O-N':(0.0,0.0),#'val1_C-N-N':(0.0,47.1),
        'val2':(0.3,2),
        'val4':(0.1,0.68),'val4_C-N-N':(0.1,0.24),'val4_H-C-H':(0.1,0.24),
        'val5':(0.3,3.6),'val7':(0.5,12.0),
        'tor2':(1.0,6.6),'tor4':(0.001,2.0),
        'V2':(-42.0,48.0),#'V3_N-C-N-N':(0.0,0.0),
        'V1_C-C-C-H':(0.0,0.0),'V2_C-C-C-H':(0.0,0.0),'V3_C-C-C-H':(0.0,0.0),
        'V1_N-C-C-N':(0.0,0.0),'V3_N-C-C-N':(0.0,0.0),
        'Dehb':(-3.2,3.0),'rohb':(1.68,2.28),'hb1':(3.05,3.06),'hb2':(19.16,19.1628),
        'Dehb_N-H-N':(-3.5,0.0),'Dehb_C-H-O':(-3.5,0.0),'Dehb_C-H-N':(-3.5,0.0),
        'acut':(0.0001,0.0085)}
         
cons = ['ovun2','ovun3','ovun4','ovun1',
        'ovun6','ovun7','ovun8','ovun5',
        # 'rosi','ropi','ropp',
        'lp2','lp1',
        #'theta0',
        #'val9','valang',
        #'val10','val8',
        'valboc','cot1','cot2','coa1','coa2','coa3','coa4',
        'pen1','pen2','pen3','pen4',
        'Devdw','rvdw',#'vdw1','gammaw','alfa',
        # 'hb1','hb2',#'rohb','Dehb',
        'val1_C-C-O','val1_H-N-O','val1_H-O-N',
        'V1_X-N-N-X','V2_X-N-N-X','V3_X-N-N-X',
        'V1_X-C-N-X','V2_X-C-N-X','V3_X-C-N-X',
        #'V1_N-C-N-N','V2_N-C-N-N','V3_N-C-N-N',
        #'V1_C-C-N-N','V2_C-C-N-N','V3_C-C-N-N',
        'val','lp3','cutoff',#'acut',
        'tor1','tor2','tor3','tor4','V1','V2','V3',    # Four-body
        'val1','val2','val3','val4','val5','val6','val7',
        'theta0','val9','val10','val8','vale','valang',
        ]    # 不进行局域优化

if args.f:
   parameters = [ 'tor1','tor2','tor3','tor4',      # Four-body
                  'V1','V2','V3']
   scale      = {'tor1':0.001,'tor2':0.001,'tor3':0.001,'tor4':0.001,
                 'V1':0.1,'V2':0.1,'V3':0.1}
   fcsv       = 'fourbody.csv'
elif  args.t:
   parameters = ['val1','val2',
                 'val3','val4','val5','val7',
                 'val6']
   scale      = {'val1':0.2,'val2':0.01,'val3':0.001,'val4':0.01,
                 'val5':0.001,'val6':0.001,'val7':0.001}
   fcsv       = 'threebody.csv'
elif args.a:
   parameters = ['theta0','val9','val10','val8','vale','valang'] # 
   scale      = {'theta0':0.0001}
elif args.bo:
   parameters = ['rosi','ropi','ropp',
                 'bo1','bo2','bo3','bo4','bo5','bo6'] ### 
   scale      = {'bo1':0.0001}
   fcsv  = 'bo.csv'
elif args.h:
   parameters = ['rohb','Dehb',# 'hb1','hb2',
                 ] ### 
   scale      = {'rohb':0.001,'Dehb':0.01}
   fcsv  = 'hbond.csv'
else:
   parameters = ['atomic']
   cons      += ['atomic']
   scale      = {'atomic':0.0001}
   fcsv       = 'atomic.csv'   

# if not args.h:
#    cons += ['rohb','Dehb','hb1','hb2']
   
if not args.vdw:
   parameters = ['Devdw_N','Devdw_C-O','Devdw_O-N','rvdw_C-O']
   cons      += parameters
   scale      = {'Devdw_N':0.0001,'Devdw_C-O':0.0001,'Devdw_O-N':0.0001,'rvdw_C-O':0.0001}
       
be_universal_nn = ['C-H','O-O'] # share the same weight and bias matrix

if __name__ == '__main__':
   ''' train ''' 
   rn = MPNN(libfile='ffield.json',
             dataset=dataset,            
             weight=weight,
             cons=cons,clip=clip,
             regularize_mf=1,regularize_be=1,regularize_bias=1,
             lambda_reg=0.001,lambda_bd=100.0,lambda_me=0.0002,lambda_pi=0.003,
             mf_layer=[9,2],be_layer=[9,1],
             EnergyFunction=1,MessageFunction=3,
             mf_universal_nn=None,be_universal_nn=be_universal_nn,
             messages=1,
             bdopt=None,    # ['H-N'], 
             mfopt=None,    # ['N'], 
             eaopt=parameters,
             batch_size=args.batch,
             fixrcbo=False,
             losFunc='n2',  # n2, mse, huber,abs
             convergence=0.9,lossConvergence=0.001) 
   
   rn.initialize()
   rn.session(learning_rate=0.0001, method='AdamOptimizer') 
   
   train(step=args.step,print_step=10,
         writelib=args.writelib,
         to_evaluate=-1.0,
         fcsv=fcsv,
         evaluate_step=args.estep,
         lossConvergence=0.0,
         max_generation=100,
         max_ml_iter=1000,
         max_data_size=200,             # 保持的数据参数组数量
         size_pop=500,                  # 用于推荐的参数组数量
         init_pop=args.i,               # 最初生成的参数组数量
         n_clusters=4,
         prob_mut=0.3,
         potential=rn,
         parameters=parameters,
         scale=scale,
         variable_scale=1)

