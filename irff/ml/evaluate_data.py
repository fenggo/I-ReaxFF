import pandas as pd
import random
from os.path import isfile
from os import listdir,getcwd
import numpy as np
from sklearn.cluster import KMeans
from .ffield import ffield_to_csv,update_ffield #,get_csv_data
from ..data.ColData import ColData
from ..reax import ReaxFF 


def init_ffield_csv(fcsv,parameters=['boc1','boc2','boc3','boc4','boc5',
                    'rosi','ropi','ropp','Desi','Depi','Depp',
                    'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6',
                    'valboc','val3','val5','val6','val8','val9','val10',
                    'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
                    'lp1','lp2','coa2','coa3','coa4','pen2','pen3','pen4',
                    'tor2','tor3','tor4','cot2',
                    'rvdw','gammaw','Devdw','alfa','vdw1']):
    files=listdir(getcwd())
    for f in files:
        if f.startswith('ffield') and f.endswith('.json'):
           ffield_to_csv(ffield=f,fcsv=fcsv,
                         parameters=parameters) # 'valang','val','vale',

def evaluate(model=None,trainer=None,fcsv='ffield_bo.csv',to_evaluate=-9999.0,
             parameters=['boc1','boc2','boc3','boc4','boc5',
                    'rosi','ropi','ropp','Desi','Depi','Depp',
                    'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6',
                    'valboc','val3','val5','val6','val8','val9','val10',
                    'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
                    'lp1','lp2','coa2','coa3','coa4','pen2','pen3','pen4',
                    'tor2','tor3','tor4','cot2',
                    'rvdw','gammaw','Devdw','alfa','vdw1'],
             evaluate_ffield=True,scale=1.0,pop=20,n_clusters=1,
             step=1000,print_step=100,writelib=500):
    ''' evaluate the score of the parameter set in csv file '''
    if not isfile(fcsv):
       # init_ffield_csv(fcsv,parameters=parameters)
       pna,row = ffield_to_csv(ffield='ffield.json',fcsv=fcsv,parameters=parameters) 
       
       scale_  = []                                                 # 创建初始参数分布
       for col in pna:
           if col=='':  continue
           key = col.split('_')[0]
           if key!='score':
              if key in scale:
                 scale_.append(scale[key])
              else:
                 scale_.append(0.01)

       ndim  = len(row)-2
       r_ = []
       for i,r in enumerate(row):
           if i!= 0 and i <= ndim:
              r_.append(float(r))  

       X = np.random.normal(loc=r_, scale=scale_, size=(pop, ndim))  # 初始参数分布
       with open(fcsv,'a') as f:
            for i,x in enumerate(X):
                print(i+1,end=',',file=f)
                for x_ in x:
                    print(x_,end=',',file=f)
                print(-99999999999.9,file=f)                         # 得分<-999，需要重新评估
    else:
       if n_clusters>1:
          d   = pd.read_csv(fcsv)
          columns        = d.columns
          for c in columns:                                ### Check Data
              col_ = c.split()
              if len(col_)>0:
                 col = col_[0]
                 if col == 'Unnamed:':
                    d.drop(c,axis=1,inplace=True)          ### Delete Unnamed column
          X   = d.values[:,:-1]
          #Y  = d.values[:, -1]
          random.seed()
          len_ = X.shape[0]
          n_   = n_clusters if len_>n_clusters else len_
          kmeans = KMeans(n_clusters=n_, random_state=random.randint(0,10)).fit(X)
          #print(kmeans.labels_)
          #print(kmeans.cluster_centers_)
          
          pna,row = ffield_to_csv(ffield='ffield.json',fcsv=fcsv,parameters=parameters,mode='w') 
          with open(fcsv,'a') as f:
             for i,_x in enumerate(kmeans.cluster_centers_):
                 index_ = np.squeeze(np.where(kmeans.labels_==i))
                 if len(index_.shape)==0 or isinstance(index_,int):
                    x      = X[index_]
                 else:
                    x      = X[index_[0]]
                 print(i+1,end=',',file=f)
                 for x_ in x:
                     print(x_,end=',',file=f)
                 print(-99999999999.9,file=f)

    d              = pd.read_csv(fcsv)
    columns        = d.columns

    for c in columns:                                                ### Check Data
        col_ = c.split()
        if len(col_)>0:
           col = col_[0]
           if col == 'Unnamed:':
              d.drop(c,axis=1,inplace=True)                          ### Delete Unnamed column

    columns = d.columns
    if evaluate_ffield:                                              ### whether evaluate the current ffield
       if not model is None:                                         ### evaluate the score of current ffield
          model.update(p=None,reset_emol=True)
          model.get_zpe()
          model.update(p=None,reset_emol=False)
          model.run(learning_rate=1.0e-4,step=step,print_step=print_step,
                    writelib=writelib,close_session=False)
          loss    = model.loss_ if 'atomic' not in parameters else model.ME_
          p_      = model.p_ 
       elif not trainer is None:
          loss,p_ = trainer(step=step,print_step=print_step)
       else:
          raise RuntimeError('-  At least one of potential or trainer function is defind!')

       if np.isnan(loss) or np.isinf(loss) or loss>99999999999.0:
          loss = 99999999999.9
       else:
          new_row = {}
          for item in columns:                                       ## 对所有列遍历
              if item:
                 if item != 'score':
                    new_row[item]= float('{:.6f}'.format(p_[item]))  ## 参数有所变化，重新更新值
          new_row['score'] = [-loss]    
          new_row = pd.DataFrame(new_row)
          d = pd.concat([new_row,d],ignore_index=True)               ## 评估当前ffield得分，并加入到数据集中

    d_row,d_column = d.shape
    for j in range(d_row):
        p=d.loc[j]
        if d.loc[j, 'score'] <= to_evaluate:
           if not model is None: 
              model.update(p=p,reset_emol=True)
              model.get_zpe()
              model.update(p=p,reset_emol=False)
              model.run(learning_rate=1.0e-4,step=step,print_step=print_step,
                        writelib=writelib,close_session=False)
              loss = model.loss_ if 'atomic' not in parameters else model.ME_
              p_   = model.p_
           elif not trainer is None:
              update_ffield(p,'ffield.json')   #### update parameters
              loss,p_ = trainer(step=step,print_step=print_step)
           else:
              raise RuntimeError('-  At least one of potential or trainer function is defind!')

           if np.isnan(loss) or np.isinf(loss) or loss>99999999999.9:
              loss = 99999999999.9
           else:
              for item in columns:                       ## 对所有列遍历
                  if item:
                     if item != 'score':
                        d.loc[j, item]= float('{:.6f}'.format(p_[item]))         ## 参数有所变化，重新更新值  
           d.loc[j, 'score'] = -loss
        d.to_csv(fcsv)
    # model.close()
    return d



