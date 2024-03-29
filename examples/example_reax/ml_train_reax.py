#!/usr/bin/env python
from irff.ml.train import train
from irff.data.ColData import ColData
from irff.reax import ReaxFF

'''
A machine learning based optimze method to optimze the parameters of ReaxFF
'''

dataset = {}
strucs  = ['tkx','tkx2']
batch   = 50

getdata = ColData()
for mol in strucs:
    trajs = getdata(label=mol,batch=batch)
    dataset.update(trajs)


clip = {'boc1':(0.0,50.0),
        'V2':(0.0,10.0),'V3':(0.0,10.0),'V1':(0.0,10.0),
        'boc2':(0.0,50.0)} # if you not sure about the parameters range, remove it.

# ** All parameters, you can chose a part of them to optimze.
# parameters = ['boc1','boc2','boc3','boc4','boc5',
#               'rosi','ropi','ropp','Desi','Depi','Depp',
#               'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6',
#               'theta0',#'valang','valboc',
#               'val1','val2','val3','val4',
#               'val5','val6','val7','val8','val9','val10',
#               'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
#               'lp1','lp2','coa2','coa3','coa4',
#               'pen1','pen2','pen3','pen4',
#               'tor2','tor3','tor4','cot1','cot2',
#               'V1','V2','V3',
#              'rvdw','gammaw','Devdw','alfa','vdw1',
#              'rohb','hb1','hb2','Dehb' ]   

parameters = ['boc1','boc2',
              'rosi','ropi','ropp','Desi','Depi','Depp',
              'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6',
              'boc3','boc4','boc5']

reax = ReaxFF(libfile='ffield.json',
              dataset=dataset, 
              optword='nocoul',
              opt=['atomic'],
              eaopt=parameters,
              clip=clip,
              batch_size=batch,
              losFunc='n2',
              lambda_bd=100.0,
              lambda_me=0.001,
              weight={'tkx':3.0,'others':2.0},
              convergence=1.0,
              lossConvergence=0.0)  # Loss Functon can be n2,abs,mse,huber
reax.initialize()
reax.session(learning_rate=0.0001, method='AdamOptimizer')

# scale: The width of the Gaussian distribution used for randomly generating parameters, default to 0.001, 
#        default to 0.001, can be adjusted slightly larger, with a larger search range centered on the
#        current parameter and a variance of scale, to generate a set of Gaussian distribution parameters
scale   = {'boc1':0.0001,'boc2':0.01,
           'rosi':0.01,'ropi':0.01,'ropp':0.01,
           'Desi':1.0,'Depi':1.0,'Depp':1.0} 

train(step=0,print_step=10,
      fcsv='param.csv',
      to_evaluate=-100.0,
      evaluate_step=0,
      lossConvergence=10.0,
      max_ml_iter=1000,
      max_data_size=2000,   # The number of data parameter to be maintained
      max_generation=100,   # The number of iterations of genetic algorithm
      init_pop=100,         # Number of initially generated parameter set
      n_clusters=20,        # The number of cores used for clustering algorithms
      size_pop=1000,        # Number of parameter groups used for recommendation
      potential=reax,
      scale=scale,     
      parameters=parameters)
