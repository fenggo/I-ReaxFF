#!/usr/bin/env python
from irff.ml.train import train
from irff.data.ColData import ColData
# from irff.reax import ReaxFF
from irff.mpnn import MPNN

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

def train_mpnn(writelib=1000,print_step=10,
               step=10000,opt=None,
               messages=1,mpopt=[True,True,True,True],lr=1.0e-4,
               lambda_bd=1000.0):
    cons=['val','vale',
          'ovun1','ovun2','ovun3','ovun4',
          'ovun5','ovun6','ovun7','ovun8',
          'lp1',#'lp2',
          'lp3',
          'cot1','cot2','coa1','coa2','coa3','coa4',
          'pen1','pen2','pen3','pen4',
          'Dehb','rohb','hb1','hb2','hbtol',
          'Depi','Depp','cutoff','acut']
    rn = MPNN(libfile='ffield.json',
              dataset=dataset, 
              dft='qe',
              spv_be=False,bore={'others':0.12}, # 
              spv_bm=False,bom={'others':1.95}, # 'O-N':(1.28,0.6),'C-N':(1.55,0.6),
              spv_pi=False,pim={ 'C-C-C':0.1,'others':5.0},lambda_pi=2.0,
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
              batch_size=50,
              convergence=0.999)

    loss,accu,accMax,i,zpe =rn.run(learning_rate=lr,
                      step=step,
                      print_step=print_step,
                      writelib=writelib) 

    p   = rn.p_
    ME = rn.MolEnergy_
    rn.close()
    return loss,p

parameters = ['gamma','chi','mu']

train(step=5000,print_step=100,
      fcsv='ffield_coulomb.csv',
      to_evaluate=-100.0,
      evaluate_step=1000,
      lossConvergence=30.0,
      max_ml_iter=10,
      max_generation=100,
      trainer=train_mpnn,
      parameters=parameters)

