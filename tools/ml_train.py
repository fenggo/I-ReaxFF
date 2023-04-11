#!/usr/bin/env python
import sys
import argparse
from irff.ml.train import train
from irff.data.ColData import ColData
from irff.ml.fluctuation import morse
from irff.reax_nn import ReaxFF_nn


help_ = 'nohup ./ml_train.py --t=1 --p=10 --s=3000>py.log 2>&1 &'
#                            --t=1 优化三体作用项 
#                            --f=1 优化四体作用项 
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--step',default=20000,type=int, help='training steps')
parser.add_argument('--estep',default=20000,type=int, help='evaluate training steps')
parser.add_argument('--writelib',default=1000,type=int, help='every this step to write parameter file')
parser.add_argument('--p',default=10,type=int,help='every this step to print')
#parser.add_argument('--maxepoch',default=100,type=int,help='the max training epoch')
parser.add_argument('--batch',default=5000,type=int,help='the batch size of every configuration')
parser.add_argument('--t',default=0,type=int,help='GA optimize the three-body energy')
parser.add_argument('--f',default=0,type=int,help='GA optimize the four-body energy')
parser.add_argument('--h',default=0,type=int,help='GA optimize the hydrogen-bond energy')
parser.add_argument('--m',default=0,type=int,help='GA optimize the many-body energy')
parser.add_argument('--n',default=1,type=int,help='optimize the neural network')
parser.add_argument('--i',default=100,type=int,help='the population size generated in the begining')
args = parser.parse_args(sys.argv[1:])


dataset = {'h22-v':'aimd_h22/h22-v.traj',
           #'hmx-s':'hmx.traj',
           }

getdata = ColData()
strucs = ['ch3no2',
          'nm2',
          'nmc']

batchs  = {'others':args.batch}

clip = {'Desi':(100.0,745.0),
        'bo1':(-0.1,-0.01),'bo2':(4.0,10.0),
        'bo3':(-0.10,-0.01),'bo4':(4.0,10.0),
        'bo3_O-O':(-0.075,-0.01),'bo4_O-O':(4.0,8.0),
        'bo5':(-0.08,-0.02),'bo6':(4.0,10.0),
        'rvdw_C-N':(1.67,2.4),'rvdw_C':(1.755,2.34),'rvdw_H':(1.53,2.0),'rvdw_H-N':(1.6,2.2),
        'rvdw_N':(1.67,2.2),'rvdw_O-N':(1.755,2.35),'rvdw_H-O':(1.6,2.0),'rvdw_O':(1.725,2.2),
        'rvdw_C-O':(1.79,2.4),
        'Devdw':(0.01,1.0),
        'alfa':(6.5,14.0),
        'vdw1':(0.8,8.0),'gammaw':(1.5,12.0),
        'valang_N':(2.75,4.0),'valang_C':(3.8,4.1),'valang_H':(0.8,1.1),'valang_O':(2.0,4.0),
        'val1':(0.0,80.0),'val2':(0.0,4.0)}


cons = ['ovun2','ovun3','ovun4','ovun1',
        'ovun6','ovun7','ovun8','ovun5',
        'rosi','ropi','ropp',
        'bo1','bo2','bo3','bo4','bo5','bo6'
        'lp1','lp2','lp3','vale','val','cutoff',
        'val1','val2','val3','val4','val5','val6','val7',
        'val8','val9','val10',
        'theta0','valang','valboc',
        'coa1','coa2','coa3','coa4',
        'pen1','pen2','pen3','pen4',
        'tor1','tor2','tor3','tor4',                         # Four-body
        'V1','V2','V3',
        'cot1','cot2', 'acut',#'atomic',
        'rohb','Dehb','hb1','hb2']  # H-drogen bond          # 不进行局域优化

if args.t: 
   scale = {'theta0':0.0001,'valang':0.0001,                 # scale (标准差) for Gaussian distribution          
            'val1':0.01,'val2':0.01,'val3':0.01,'val4':0.01,                 
            'val5':0.01,'val6':0.01,'val7':0.01,
            'val8':0.01,'val9':0.01,'val10':0.001}                           
   parameters = [#'acut',
                 'theta0','valang',                          # three-body 
                 'val8','val9','val10',  
                 'val1','val2','val3','val4',                           
                 'val5','val6','val7',
                 #'pen1','pen2','pen3','pen4',
                 #'coa1','coa2','coa3','coa4','valboc',
                 ]
   fcsv = 'threebody.csv'
elif args.f:
   scale      = {'tor1':0.01,'tor2':0.01,'tor3':0.01,'tor4':0.01,
                 'V1':0.01,'V2':0.01,'V3':0.01}
   parameters = ['tor1','tor2','tor3','tor4',                # Four-body
                 'V1','V2','V3',
                 #'cot1','cot2',
                 'acut']                                     # 进行遗传算法优化
   fcsv = 'fourbody.csv' 
elif args.m:
   parameters = ['theta0','valang',                          # three-body 
                 'val1','val2','val3','val4',                           
                 'val5','val6','val7','val8','val9','val10',  
                 'tor1','tor2','tor3','tor4',                # Four-body
                 'V1','V2','V3',
                 #'cot1','cot2',
                 'acut']                                     # 进行遗传算法优化
   fcsv = 'manybody.csv' 
elif args.h:
   scale      = {'rohb':0.001,'Dehb':0.001,'hb1':0.001,'hb2':0.001}
   parameters = ['rohb','Dehb','hb1','hb2']                        # Hydrogen-bond 遗传算法优化
   fcsv       =  'hbond.csv' 
else:
   scale      = {'atomic':0.01}
   cons      += ['atomic']
   parameters = ['atomic']
   fcsv = 'atomic.csv'

getdata = ColData()
for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)


if args.t or args.f:
   cons += ['vdw1','gammaw','rvdw','alfa','Devdw',
            'Desi','rosi','ropi','ropp',
            'bo1','bo2','bo3','bo4','bo5','bo6']   

nnopt = [1,1,1,1]  if args.n else [0,0,0,0] 
                    

reax_nn = ReaxFF_nn(libfile='ffield.json',
                    dataset=dataset,
                    weight={'hmx-s':20.0,'others':2.0},
                    optword='nocoul',mpopt=nnopt,eaopt=parameters,
                    optmol=True,cons=cons,clip=clip,
                    regularize_mf=1,regularize_be=1,regularize_bias=0,
                    lambda_reg=0.01,lambda_bd=1000.0,lambda_me=0.01,
                    mf_layer=[9,1],be_layer=[9,0],
                    bdopt=None,#['C-O'],
                    mfopt=None,#['C','O'],
                    batch=args.batch,
                    fixrcbo=False,
                    losFunc='n2',  # n2, mse, huber,abs
                    convergence=0.999) 

reax_nn.initialize()
# GradientDescentOptimizer AdamOptimizer
reax_nn.session(learning_rate=0.0001, method='AdamOptimizer') 


train(step=args.step,print_step=args.p,
      writelib=args.writelib,
      fcsv=fcsv,
      evaluate_step=args.estep,
      lossConvergence=0.0,
      max_ml_iter=1000,
      max_data_size=1000,        # 保持的数据参数组数量
      size_pop=1600,             # 用于推荐的参数组数量
      init_pop=args.i,               # 最初生成的参数组数量
      prob_mut=0.3,
      potential=reax_nn,
      parameters=parameters,
      scale=scale)

