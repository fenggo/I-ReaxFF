#!/usr/bin/env python
from irff.ml.train import train
from irff.data.ColData import ColData
from irff.reax import ReaxFF

'''
A machine learning based optimze method to optimze the parameters of ReaxFF
'''

dataset = {#'gpu-0':'data/gpu-0.traj',
           #'gpu-1':'data/gpu-1.traj'
           'gpd-0':'data/gpd-0.traj',
           'gpd-1':'data/gpd-1.traj',
           'gpd-2':'data/gpd-2.traj',
           }


batch   = 50
batchs  = {'others':50}


clip = {'boc1':(0.0,50.0),
        'V2':(0.0,10.0),'V3':(0.0,10.0),'V1':(0.0,10.0)} # if you do not sure with the value range can be left unset.


reax = ReaxFF(libfile='ffield.json',
              dataset=dataset, 
              opt=['atomic']
              clip=clip,
              batch_size=batch,
              losFunc='n2',
              lambda_bd=100.0,
              lambda_me=0.001,
              atol=0.002,hbtol=0.002,
              weight={'h2o2-1':50.0,'others':2.0},
              convergence=1.0,
              lossConvergence=0.0)  # Loss Functon can be n2,abs,mse,huber
reax.initialize()
reax.session(learning_rate=0.0001, method='AdamOptimizer')
'''
** All parameters, you can chose a part of them to optimze.
parameters = ['boc1','boc2','boc3','boc4','boc5',
              'rosi','ropi','ropp','Desi','Depi','Depp',
              'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6',
              'theta0',#'valang','valboc',
              'val1','val2','val3','val4',
              'val5','val6','val7','val8','val9','val10',
              'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
              'lp1','lp2','coa2','coa3','coa4',
              'pen1','pen2','pen3','pen4',
              'tor2','tor3','tor4','cot1','cot2',
              'V1','V2','V3',
              'rvdw','gammaw','Devdw','alfa','vdw1',
              'rohb','hb1','hb2','Dehb' ]   
'''
parameters = ['boc1','boc2','boc3','boc4','boc5',
              'rosi','ropi','ropp','Desi','Depi','Depp',
              'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6']
# scale: The width of the Gaussian distribution used for randomly generating parameters,
#        default to 0.001, can be setted slightly larger value, with a larger search range 
#        centered on the current parameter and a variance of scale, to generate a set of 
#        Gaussian distribution of the parameters
scale      = {'rosi':0.01,'ropi':0.01,'ropp':0.01} 

train(step=0,print_step=10,
      fcsv='param.csv',
      to_evaluate=-1000.0,
      evaluate_step=0,
      lossConvergence=10.0,
      max_ml_iter=1000,
      max_data_size=2000,   # The number of data parameter groups to be maintained, the larger the computing power allowed, the better
      max_generation=100,   # The number of iterations of genetic algorithm for parameter group recommendation
      init_pop=100,         # Number of initially generated parameter groups
      n_clusters=20,        # The number of cluster used for clustering algorithms, with more parameters, takes more cores, 
      #                       and fewer parameters. Conversely, for one parameter, at most one core can be used
      size_pop=3000,        # The number of parameter groups used for recommendation, > max_data_size
      prob_mut=0.3,  
      potential=reax,
      scale=scale,     
      parameters=parameters)

