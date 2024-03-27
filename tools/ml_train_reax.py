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
        'V2':(0.0,10.0),'V3':(0.0,10.0),'V1':(0.0,10.0)} # 对取值范围不确定可以不设置


reax = ReaxFF(libfile='ffield.json',
              dataset=dataset, 
              optword='nocoul',
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
# scale: 用于随机产生参数的高斯分布的宽度，默认为0.001, 可以适当的调大一些，搜索范围更大
#        以当前参数为中心，方差为scale,产生一组高斯分布的参数
scale      = {'rosi':0.01,'ropi':0.01,'ropp':0.01} 

train(step=0,print_step=10,
      fcsv='param.csv',
      to_evaluate=-1000.0,
      evaluate_step=0,
      lossConvergence=10.0,
      max_ml_iter=1000,
      max_data_size=2000,   # 保持的数据参数组数量,算力允许越大越好
      max_generation=100,   # 用于推荐的参数组遗传算法的迭代次数
      init_pop=100,         # 最初生成的参数组数量
      n_clusters=20,        # 用于聚类算法的核心数量,参数多，核心取多些，参数少，反之，1个参数，最多只能1个核心
      size_pop=2000,        # 用于推荐的参数组数量
      prob_mut=0.3,  
      potential=reax,
      scale=scale,     
      parameters=parameters)

