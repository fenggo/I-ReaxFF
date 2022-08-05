#!/usr/bin/env python
from irff.ml.train import train
from irff.data.ColData import ColData
from irff.reax import ReaxFF


getdata = ColData()
strucs = ['c22',
          # 'c2c6',
          'c3',
          'c32',
          'c4',
          'c5',
          'c8',
          'c10',
          'c12',
          'c14',
          ] 

batchs  = {'others':50}
dataset = {}

for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)


mp = MPNN(libfile='ffield.json',
          dataset=dataset, 
          dft='qe',
          spv_be=True,bore={'others':0.12}, # 
          spv_bm=False,bom={'others':1.95}, # 'O-N':(1.28,0.6),'C-N':(1.55,0.6),
          spv_pi=True,pim={ 'C-C-C':0.1,'others':5.0},lambda_pi=2.0,
          spv_ang=False,lambda_ang=0.02,
          weight={'others':2.0},
          optword='nocoul',
          opt=opt,optmol=True,cons=cons,
          VariablesToOpt=None,#['gammaw_H','vdw1','Devdw_H','rvdw_H','Desi_H-H','alfa_H'],
          resetDeadNeuron=False,
          regularize=True,
          lambda_reg=0.003,lambda_bd=lambda_bd,lambda_me=0.03,
          messages=messages,mpopt=mpopt,
          bo_layer=[4,1],mf_layer=[9,2],be_layer=[6,1],
          bo_univeral_nn=None,#['C-H','H-O'], 
          be_univeral_nn=None,#['C-H','H-O'], 
          mf_univeral_nn=None,#['C'],
          vdw_univeral_nn=None,#['C-H','H-O'],
          EnergyFunction=1,
          vdwnn=True,vdw_layer=[6,1],
          bdopt=None,# ['N-N'],# ['H-H','O-O','C-C','C-H','N-N','C-N','C-O'],
          mfopt=None,#['N'],# ['H','O','C','N'],
          batch_size=batch,
          convergence=convergence)
mp.initialize()
mp.session(learning_rate=0.0001, method='AdamOptimizer')


parameters = ['gamma','chi','mu']

train(step=10000,print_step=100,
      fcsv='ffield_bo.csv',
      to_evaluate=1000,
      evaluate_step=1000,
      lossConvergence=30.0,
      max_ml_iter=10,
      max_generation=100,
      potential=mp,
      parameters=parameters)

