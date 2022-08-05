from os import popen #,system,
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from .evolution import Evolution
from .ffield import update_ffield
from .evaluate_data import evaluate
from ..data.ColData import ColData
from ..reax import ReaxFF


def train(step=5000,print_step=100,
             evaluate_step=100,
                      fcsv='ffield.csv',
               to_evaluate=-2000,
           evaluate_ffield=False,
           lossConvergence=30.0,
               max_ml_iter=2000,
            max_generation=100,
                  size_pop=100,
              create_ratio=0.1,
                 potential=None,
                   trainer=None,
                parameters=['boc1','boc2'],
             max_data_size=1000,
             GAThreshold=0.01):
    ''' Using machine learing model to assis the training of ReaxFF'''
    sizepop = int(size_pop*(1.0-create_ratio))
    scoreConvergence = - lossConvergence

    d = evaluate(model=potential,trainer=trainer,
                 fcsv=fcsv,to_evaluate=to_evaluate,
                 step=evaluate_step,print_step=print_step,
                 evaluate_ffield=evaluate_ffield,
                 parameters=parameters)
    d.sort_values(axis=0,by='score',ascending=False,inplace=True)

    X       = d.values[:, : -1]
    Y       = d.values[:, -1]
    columns = d.columns

    new_row = {}
    zrow = d.index[0]
    for col in columns:
        new_row[col] = [d.loc[zrow, col]]
        # print(col,d.loc[0, col])
    new_row = pd.DataFrame(new_row)

    ### 训练机器学习模型  ### MLP RF GMM
    ml_model = RandomForestRegressor(n_estimators=100,max_depth=10,oob_score=True).fit(X,Y)
    score    = ml_model.score(X,Y) # cross_val_score(rfr,x,y,cv=10).mean()

    def func(x):                        ## 用于遗传算法评估的函数
        x = np.expand_dims(x,axis=0)
        y = ml_model.predict(x)
        return -np.squeeze(y)

    with open('ML.log','w') as galog:
         print('---------------------------------------------------------------',file=galog)
         print('-              Machine Learning Parameterization              -',file=galog)
         print('---------------------------------------------------------------\n',file=galog)
    it_     = 0
    score   = scoreConvergence - 1.0
    do_gen  = True
    while score<scoreConvergence and it_< max_ml_iter:
        # d   = evaluate(model=potential,trainer=trainer,fcsv=fcsv,
        #                to_evaluate=to_evaluate, # Y[-1]+0.0001,
        #                step=evaluate_step,print_step=print_step,
        #                parameters=parameters)
        X   = d.values[:, : -1]
        Y   = d.values[:, -1]
        ml_model.fit(X,Y)
        score_ = ml_model.score(X,Y) # cross_val_score(rfr,x,y,cv=10).mean()
        galog = open('ML.log','a')
        print('The accuraccy of the mathine learning model: {:f}'.format(score_),file=galog)

        size_ = d.shape[0]
        if size_ > sizepop:
           size_ = sizepop
        X_ = d.values[:size_, : -1]

        ## if PSO DE GMM
        if do_gen:
           print('Do genetic evolute ...',file=galog)

           lb=0.9*np.min(X_,axis=0)
           ub=1.1*np.max(X_,axis=0)
           de = Evolution(func,n_dim=X_.shape[1], F=0.5,size_pop=size_pop,
                          max_iter=max_generation, 
                          prob_mut=0.1,lb=lb, 
                          ub=ub,X_input=X_)             
           best_x,best_y = de.run(logfile=galog)                     ###   PSO GMM
           print('The guessed score of best candidate: {:f}'.format(float(-best_y)),file=galog)
        else:
           print('The score of current parameters set looks good, need not do the genetic step.',file=galog)
        
        if do_gen:
           for i,key in enumerate(columns):
               if key != 'score':
                  new_row.loc[0,key] = best_x[i]

        if not potential is None:                  #### 两种引入方式 potential or trainer
           potential.update(p=new_row.loc[0])
           potential.run(learning_rate=1.0e-4,step=step,print_step=print_step,
                          writelib=1000,close_session=False)
           p_      = potential.p_ 
           score   = -potential.loss_
        elif not trainer is None:
           update_ffield(new_row.loc[0],'ffield.json')    #### update parameters
           loss,p_ = trainer(step=step,print_step=print_step)
           score   = -loss
        else:
           raise RuntimeError('-  At least one of potential or trainer function is defind!')
        
        zrow = d.index[0]
        ratio = score/d.loc[zrow, 'score']
        if ratio<0.999:
           popen('cp ffield.json ffield_best.json')
        do_gen = False if ratio< (1.0-GAThreshold) else True

        if score>-9999999.0:
           for i,key in enumerate(columns):
               if key:
                  if key != 'score':
                     new_row.loc[0,key] = p_[key]       
           new_row.loc[0,'score'] = score 

        if len(new_row.loc[0])>1:
           l2 = 0.0
           for key in columns:
               if key != 'score':
                  l2 += np.square(new_row.loc[0,key] - d.loc[zrow, key])
           if l2>0.00001:
              d = pd.concat([new_row,d],ignore_index=True)
           else:
              d.loc[zrow, 'score']  = score
           print('The score after gradient descent: {:f}'.format(score),file=galog)
           d.sort_values(axis=0,by='score',ascending=False,inplace=True)

        nrow = d.shape[0]
        if nrow>max_data_size:
           nrow = d.index[nrow-1]
           sc = d.loc[nrow, 'score']  
           d.drop(nrow,axis=0,inplace=True) 
           print('row index {:d} in data: {:f} has been deleted beacuse reach maxium datasize'.format(nrow-1,
                 sc),file=galog)
            
        print('Saving the data to {:s} ...'.format(fcsv),file=galog)
        d.to_csv(fcsv)
        it_ += 1
        galog.close()

# getdata = ColData()
# strucs  = ['al4',
#            'c4',
#            'c2al2',
#            'c2f2',
#            'f4',
#            ]

# batchs  = {'others':50}
# dataset = {}

# for mol in strucs:
#     b = batchs[mol] if mol in batchs else batchs['others']
#     trajs = getdata(label=mol,batch=b)
#     dataset.update(trajs)

# reax = ReaxFF(libfile='ffield.json',  
#               dataset=dataset,
#               dft='siesta',
#               # cons=parameters+['val','vale','lp3','cutoff','hbtol'],
#               optword='nocoul',
#               opt=None,
#               batch_size=batch,
#               losFunc='n2',
#               lambda_bd=10000.0,
#               lambda_me=0.01,
#               weight={'others':2.0},
#               convergence=1.0,
#               lossConvergence=0.0) # Loss Functon can be n2,abs,mse,huber
# reax.initialize()
# reax.session(learning_rate=0.0001, method='AdamOptimizer')

