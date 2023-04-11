#!/usr/bin/env python
from irff.ml.train import train
from irff.data.ColData import ColData
from irff.ml.fluctuation import morse
from irff.reax_nn import ReaxFF_nn

dataset = {}

getdata = ColData()
strucs = [#'gpdm',
          'gpp',
          'gphit3',
          #'cnt32-1',
          'cnt32',
          #'c60',
          #'cnt333-2'
          ]

batch   = 2000
batchs  = {'others':2000}

clip = {'boc1':(10.0,26.0),'boc2':(3.0,10.0),'boc3':(0.1,19.9),'boc4':(0.5,9.9),
        'bo1':(-0.4,-0.01),'bo2':(4.0,10.0),'rosi':(1.3,1.4),
        'bo3':(-0.4,-0.03),'bo4':(4.0,16.0),
        'bo5':(-0.4,-0.04),'bo6':(4.0,16.0),
        'Desi':(150.0,350.0),'Depi':(30.0,120.0),'Depp':(30.0,120.0),
        'Devdw':(0.02,0.2),'alfa':(9.0,14.0),'rvdw':(1.75,2.3),'vdw1':(1.5,2.0),'gammaw':(2.5,8.0),
        'theta0':(60.0,90.0)}

parameters = [#'theta0','valang',                          # 'valboc',
              #'val1','val2','val3','val4',
              #'val5','val6','val7','val8','val9','val10', 
              # 'pen1','pen2','pen3','pen4',
              #'coa1','coa2','coa3','coa4',  
              'tor2','tor3','tor4',                        # Four-body
              'V1','V2','V3',
              'cot1','cot2']                               # 进行遗传算法优化

getdata = ColData()
for mol in strucs:
    b = batchs[mol] if mol in batchs else batchs['others']
    trajs = getdata(label=mol,batch=b)
    dataset.update(trajs)


def trainer(writelib=1000,print_step=10,
            batch=2000,
            step=10000,opt=None,
            mpopt=[True,True,True,True],lr=1.0e-4):
    ''' a trainer for ReaxFF-nn '''
    cons = ['ovun2','ovun3','ovun4','ovun1',
            'ovun6','ovun7','ovun8','ovun5',
            'lp1','lp2','lp3','vale','val','cutoff',
            'val1','val2','val3','val4','val5','val6','val7',
            'val8','val9','val10','theta0','valang','valboc',
            'coa1','coa2','coa3','coa4',
            'pen1','pen2','pen3','pen4',
            'tor1','tor2','tor3','tor4',                              # Four-body
            'V1','V2','V3',
            'cot1','cot2' ,
            'rohb','Dehb','hb1','hb2']  # H-drogen bond        # 不进行局域优化
    if step<5001:
       cons += ['vdw1','gammaw','rvdw','alfa','Devdw','Desi',
                'rosi','ropi','ropp',
                'bo1','bo2','bo3','bo4','bo5','bo6']  
       mpopt = [0,0,0,0] # neural network for BO,MF,BE,VDW  
       lr = 0.001

    rn = ReaxFF_nn(libfile='ffield.json',
                   dataset=dataset,
                   weight={'gphit3':4.0,'others':2.0},
                   optword='nocoul',
                   mpopt=mpopt,
                   opt=opt,optmol=True,cons=cons,clip=clip,
                   regularize_mf=1,regularize_be=1,regularize_bias=1,
                   lambda_reg=0.01,lambda_bd=1000.0,lambda_me=0.001,
                   mf_layer=[9,1],be_layer=[9,1],
                   bdopt=None,    # ['N-N'],
                   mfopt=None,    # ['N'],
                   batch=batch,
                   fixrcbo=False,
                   losFunc='n2',  # n2, mse, huber,abs
                   convergence=0.999) 

    loss,accu,accMax,i,zpe =rn.run(learning_rate=lr,
                      step=step,
                      print_step=print_step,
                      writelib=writelib,
                      method='AdamOptimizer')

    p   = rn.p_
    rn.close()
    return loss,p


train(step=10000,print_step=100,
      fcsv='fourbody.csv',
      to_evaluate=-100,
      evaluate_step=2000,
      lossConvergence=0.0,
      max_ml_iter=100,
      max_generation=100,
      max_data_size=150,
      size_pop=100,
      trainer=trainer,
      parameters=parameters)

