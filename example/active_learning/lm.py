#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
from irff.LearningMachine import LearningMachine
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol


dataset = {} #{'c64-s1':'s1.traj',
             # 'c64-s2':'s2.traj'}

getdata = ColData()
strucs = [#'c2',
          # 'c2c6',
          # 'c3',
          # 'c32',
          'c4',
          # 'c5',
          'c6',
          # 'c62',
          # 'c8',
          # 'c10',
          # 'c12',
          # 'c14',
          # 'c16',
          # 'c18',
          'dia',
          #'gpu',
          #'gp',
          'gpd',
          # 'gpe',
          ]
# strucs = ['c64']
batch  = 50
batchs = {'others':batch }

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)
check_emol(dataset)

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
      'V2':(0.0,10.0),'V3':(0.0,10.0),'V1':(0.0,10.0)}


lm = LearningMachine(initConfig='gpd.gen',
                     nn=False,                       # True: ReaxFF-MPNN模型 False: ReaxFF
                     dataset=dataset,ncpu=8,batch=batch,
                     maxiter=70,
                     step=20000,md_step=100,MinMDstep=35,MaxMDstep=100,mom_step=100,colFrame=20,
                     T=2000,
                     angmax=20.0,
                     lossCriteria=1.650,accCriteria=0.5,
                     CheckZmat=False,
                     lambda_reg=0.0003,lambda_bd=1000.0,lambda_me=0.1,lambda_ang=0.01,
                     learnWay=3,dft_step=48,
                     beta=0.9, # FirstAtom=6,FreeAtoms=[6],
                     EngTole=0.01,dEtole=0.1,dEstop=2.0,
                     spv_be=False,
                     spv_ang=False,
                     weight={'others':2.0},
                     optword='nocoul',               # 不优化库仑相互作用
                     clip=clip,
                     writelib=1000,convergence=0.04,
                     dft='siesta',xcf='GGA',xca='PBE',basistype='split')
lm.run()
lm.close()

