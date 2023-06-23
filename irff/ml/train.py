from os import popen 
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from .evolution import Evolution
from .ffield import update_ffield
from .evaluate_data import evaluate
# from ..data.ColData import ColData
# from ..reax import ReaxFF


def train(step=5000,print_step=100,writelib=500,
             evaluate_step=100,
                      fcsv='ffield.csv',
               to_evaluate=-998,
           evaluate_ffield=True,
           lossConvergence=30.0,
               max_ml_iter=2000,
            max_generation=500,
                  size_pop=500,
                  init_pop=10,
                n_clusters=1,
                  prob_mut=0.5,
                 potential=None,
                   trainer=None,
                parameters=['boc1','boc2'],
                     scale={},
            variable_scale=0,
             max_data_size=1000,
                regenerate=False,
          end_search_ratio=0.999999,
            end_search_nan=False,
               GAThreshold=0.5):
    ''' Using machine learing model to assis the training of ReaxFF'''
    scoreConvergence = - lossConvergence

    d = evaluate(model=potential,trainer=trainer,
                 fcsv=fcsv,to_evaluate=to_evaluate,
                 step=evaluate_step,print_step=print_step,writelib=writelib,
                 evaluate_ffield=evaluate_ffield,pop=init_pop,scale=scale,
                 n_clusters=n_clusters,parameters=parameters)
    d.sort_values(axis=0,by='score',ascending=False,inplace=True)

    X       = d.values[:, : -1]
    Y       = d.values[:, -1]
    columns = d.columns
    new_row = {}
    zrow    = d.index[0]
    
    scale_  = []
    for col in columns:
        new_row[col] = [d.loc[zrow, col]]
        key = col.split('_')[0]
        if key!='score':
           if key in scale:
              scale_.append(scale[key])
           else:
              scale_.append(0.01)

    scale_  = np.array(scale_)
    new_row = pd.DataFrame(new_row)

    ### 训练机器学习模型  ### MLP RF GMM
    ml_model = RandomForestRegressor(n_estimators=100,max_depth=10,oob_score=True).fit(X,Y)
    score    = ml_model.score(X,Y)      # cross_val_score(rfr,x,y,cv=10).mean()

    def func(x):                        ## 用于遗传算法评估的函数
        # x = np.expand_dims(x,axis=0)
        y = ml_model.predict(x)
        return -y # -np.squeeze(y)

    with open('evolution.log','w') as galog:
         print('----------------------------------------------------------------',file=galog)
         print('-          Machine Learning Parameter optimization             -',file=galog)
         print('----------------------------------------------------------------\n',file=galog)
         print('  Initial parameter vector: ',file=galog)
         print(new_row,file=galog)

    it_       = 0
    score     = scoreConvergence - 0.1
    do_gen    = True
    keep_best = 0
    while score<scoreConvergence and it_< max_ml_iter:
        size_ = d.shape[0]
        zrow  = d.index[0]
        sizepop = int(size_pop/2)
        if size_ > sizepop:
           size_ = sizepop
        X_ = d.values[:size_, : -1]
        X   = d.values[:, : -1]
        Y   = d.values[:, -1]

        ml_model.fit(X,Y)
        score_ = ml_model.score(X,Y) 
        feature_importances = ml_model.feature_importances_
        max_ = max(feature_importances)

        if variable_scale==2 and score_>0.98:
           _scale = scale_*feature_importances*feature_importances/(max_*max_) # *np.random.choice([0.1,1.0,10.0])
        elif variable_scale>=1 and score_>0.96:
           _scale = scale_*feature_importances/max_ # *np.random.choice([0.1,1.0,10.0])
        else:
           _scale = scale_ # *np.random.choice([0.1,1.0,10.0])

        galog = open('evolution.log','a')
        print('\n           ----------------------------------',file=galog)
        print('                   Iteration: {:6d} '.format(it_),file=galog)
        print('           ----------------------------------\n',file=galog)

        print('\n  The accuraccy of the mathine learning model: {:f}'.format(score_),file=galog)
        print('  The feature importances of the mathine learning model: ',file=galog)
        for i_,x_ in enumerate(feature_importances):
            if i_%3==0:
               print(' ',file=galog)
            print('{:16s} {:9.6f} |'.format(columns[i_],x_),end=' ',file=galog)
        print('\n ',file=galog)

        ## PSO DE GMM
        if do_gen:
           print('  Do genetic recommendation ...\n',file=galog)
           # gentic_start_time = time.time()
           # lb=0.9*np.min(X_,axis=0)
           # ub=1.1*np.max(X_,axis=0)
           de = Evolution(func,n_dim=X_.shape[1], F=0.5,size_pop=size_pop,
                          scale=_scale,max_iter=max_generation, n_clusters=n_clusters,
                          prob_mut=prob_mut,X_input=X_)    

           best_x,best_y = de.run(logfile=galog)                     ###   PSO GMM
           print('  The guessed score of best candidate: {:f} '.format(float(-best_y)),file=galog) 
           print('  The score of last best             : {:f} '.format(d.loc[zrow, 'score']),file=galog) 
           print('\n  The parameter vector: ',file=galog)
           for i_,x_ in enumerate(best_x):
               if i_%3==0:
                  print('\n--------------------------------------------------------------------------------------',file=galog)
               cn = columns[i_]
               best_x[i_] = x_ = float('{:.6f}'.format(x_))
               print('{:16s} {:9.6f} |'.format(cn,x_),end=' ',file=galog)
           print('\n--------------------------------------------------------------------------------------',file=galog)
           keep_ = True # if step==0 else False
           cycle = 0 
           while keep_ and cycle<30:
               for i,key in enumerate(columns):
                   if key != 'score':
                      if abs(new_row.loc[0,key] - best_x[i])>=0.000001:
                         keep_ = False
                         print('                {:16s} {:9.6f} -> {:9.6f}'.format(key,new_row.loc[0,key],best_x[i]),file=galog)
               if keep_: 
                  # print(' The parameter vector keep best, a random parameter set is chosen ...',file=galog)
                  print(' The parameter vector keep best, a second best parameter set is chosen ...',file=galog)
                  # i = np.random.choice(de.size_pop)
                  cycle += 1
                  best_x = de.X[de.global_best_index[cycle]]
                  best_y = de.Y[de.global_best_index[cycle]]
        else:
           print('  The score of current parameters set looks good, need not do the genetic step.',file=galog)
        print('--------------------------------------------------------------------------------------',file=galog)
        # gentic_end_time = time.time()
        # print('\n  The time usage of genetic recommendation: {:f}'.format(gentic_end_time-gentic_start_time),file=galog)
        if do_gen:
           for i,key in enumerate(columns):
               if key != 'score':
                  new_row.loc[0,key] = best_x[i]

        if not potential is None:                                  #### 两种引入方式 potential or trainer
           # potential.initialize()
           # potential.session(learning_rate=0.0001, method='AdamOptimizer') 
           potential.update(p=new_row.loc[0],reset_emol=True)      ### --------------------------------
           potential.get_zpe()
           potential.update(p=new_row.loc[0],reset_emol=False) 
           potential.run(learning_rate=1.0e-4,step=step,print_step=print_step,
                          writelib=writelib,close_session=False)
           p_      = potential.p_ 
           score   = -potential.loss_ if 'atomic' not in parameters else -potential.ME_
        elif not trainer is None:
           update_ffield(new_row.loc[0],'ffield.json')    #### update parameters
           loss,p_ = trainer(step=step,print_step=print_step)
           score   = -loss
        else:
           raise RuntimeError('-  At least one of potential or trainer function is defind!')
        
        ratio = score/d.loc[zrow, 'score']
        print('\n  The current ratio of the evolution algrithm: {:9.7f}'.format(ratio),file=galog)
        if ratio<0.999999:
           popen('cp ffield.json ffield_best.json')
        do_gen = False if ratio< (1.0-GAThreshold) else True

        if not end_search_nan or score>-99999999999.0: 
           for i,key in enumerate(columns):
               if key:
                  if key != 'score':
                     new_row.loc[0,key] = p_[key]       
           new_row.loc[0,'score'] = score 
        else:
           print('\n  Error: the score of current parameter vector is NaN, the search is end.',file=galog)
           break

        if len(new_row.loc[0])>1:
           x_   = new_row.loc[0].values[:-1]
           irow = -1
           for i,x in enumerate(X):
               # print('x_: \n',x_,file=galog)
               # print('x: \n',x,file=galog)
               if np.array_equal(x_,x):
                  irow = i

           if irow!=0 or ratio<end_search_ratio:
              if irow<0:
                 d = pd.concat([new_row,d],ignore_index=True)
              else:
                 d.loc[irow, 'score']  = score
              keep_best  = 0
           else:
              d.loc[zrow, 'score']  = score
              keep_best += 1

           print('  The score after evaluate: {:f}\n'.format(score),file=galog)
           d.sort_values(axis=0,by='score',ascending=False,inplace=True)

        nrow = d.shape[0]
        if nrow>max_data_size:
           nrow = d.index[nrow-1]
           sc = d.loc[nrow, 'score']  
           d.drop(nrow,axis=0,inplace=True) 
           print('row index {:d} in data: {:f} has been deleted beacuse maxium datasize reached'.format(nrow-1,
                 sc),file=galog)
            
        print('  Saving the data to {:s} ...'.format(fcsv),file=galog)
        if keep_best>4 and ratio>=end_search_ratio:
           print('\n  The current parameter vector keep best for iterations, the search is end.',file=galog)
           break
           
        d.to_csv(fcsv)
        it_ += 1
        if it_>= max_ml_iter:
           print('\n  The maximum iterations have reached, the search is end.',file=galog)
        galog.close()

