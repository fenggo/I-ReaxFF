#!/usr/bin/env python
from irff.ml.train import train
from irff.data.ColData import ColData
# from irff.reax import ReaxFF
from irff.mpnn import MPNN

dataset = {#'gpu-0':'data/gpu-0.traj',
           #'gpu-1':'data/gpu-1.traj'
           'gpd-0':'data/gpd-0.traj',
           'gpd-1':'data/gpd-1.traj',
           'gpd-2':'data/gpd-2.traj',
           }

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

batch   = 50
batchs  = {'others':50}

clip = {'boc1':(10.0,26.0),'boc2':(3.0,10.0),'boc3':(0.1,19.9),'boc4':(0.5,9.9),
        'bo1':(-0.4,-0.01),'bo2':(4.0,10.0),'rosi':(1.3,1.4),
        'bo3':(-0.4,-0.03),'bo4':(4.0,16.0),
        'bo5':(-0.4,-0.04),'bo6':(4.0,16.0),
        'Desi':(150.0,350.0),'Depi':(30.0,120.0),'Depp':(30.0,120.0),
        'be1':(0.01,0.6),'be2':(0.01,0.5),
        'ovun1':(0.1,0.9),'ovun3':(0.01,35.0),'ovun4':(0.5,10.0),
        'ovun5':(0.5,50.0),'ovun7':(0.5,16.0),'ovun8':(1.0,16.0),
        'lp1':(5.0,18.0),'lp2':(0.0,0.01),
        'Devdw':(0.02,0.2),'alfa':(9.0,14.0),'rvdw':(1.75,2.3),'vdw1':(1.5,2.0),'gammaw':(2.5,8.0),
        'theta0':(60.0,90.0),
        'val2':(0.21,2.0),'val3':(0.1,5.0),
        'val6':(0.5,36.0),'val8':(0.5,4.0),'val9':(1.0,2.0),
        'pen1':(9.0,11.0),'pen2':(1.0,9.0),'pen3':(0.0,1.0),'pen4':(1.0,6.0),
        'coa1':(-1.0,0.0),'cot1':(-1.0,0.0),'cot2':(0.0,5.0),
        'V2':(0.0,10.0),'V3':(0.0,10.0),'V1':(0.0,10.0)}

parameters = [# 'boc1','boc2','boc3','boc4','boc5',
              'rosi','ropi','ropp','Desi',#'Depi','Depp',
              'bo1','bo2','bo3','bo4','bo5','bo6',#'be1','be2',
              'theta0',#'valang','valboc',
              'val1','val2','val3','val4',
              'val5','val6','val7','val8','val9','val10',
              'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
              'lp1','lp2',
              'coa2','coa3','coa4',
              'pen1','pen2','pen3','pen4',
              'tor2','tor3','tor4','cot1','cot2',
              'V1','V2','V3',
              'rvdw','gammaw','Devdw','alfa','vdw1',
              'rohb','Dehb','hb1','hb2']  # 进行机器学习优化

cons = ['val','vale','lp3','cutoff','boc1','boc2',
        'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
        'lp1','lp2']                                  # 不进行局域优化

getdata = ColData()
for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)

mpnn = MPNN(libfile='ffield.json',
            dataset=dataset,            
            spv_ang=False,lambda_ang=0.02,
            spv_vdw=False,spv_bo=False,
            #vlo=vlo,#vup=vup,
            weight={'c6':2.0,'others':2.0},
            optword='nocoul',
            cons=cons,
            clip=clip,
            regularize_bo=1,
            lambda_reg=0.002,lambda_bd=10000.0,lambda_me=0.05,
            mf_layer=[6,1],be_layer=[6,0],
            BOFunction=0,EnergyFunction=1,MessageFunction=4,
            mf_univeral_nn=None,
            vdwnn=False,vdw_layer=[9,1],
            bdopt=None,    # ['N-N'],# ['H-H','O-O','C-C','C-H','N-N','C-N','C-O'],
            mfopt=None,    # ['N'],# ['H','O','C','N'],
            batch_size=batch,
            losFunc='n2',  # n2, mse, huber,abs
            convergence=0.999)

mpnn.initialize()
mpnn.session(learning_rate=0.0001, method='AdamOptimizer')


train(step=30000,print_step=100,
      fcsv='ffield_ml.csv',
      to_evaluate=-100,
      evaluate_step=2000,
      lossConvergence=0.0,
      max_ml_iter=100,
      max_generation=100,
      max_data_size=150,
      size_pop=100,
      potential=mpnn,
      parameters=parameters)

