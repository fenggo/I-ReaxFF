# import csv
import pandas as pd
from os.path import isfile
from os import listdir,getcwd
import numpy as np
from .ffield import ffield_to_csv,update_ffield #,get_csv_data
from ..data.ColData import ColData
from ..reax import ReaxFF 
#from ..dft.CheckEmol import check_emol

# parameters = ['boc1','boc2','boc3','boc4','boc5',
#               'rosi','ropi','ropp','Desi','Depi','Depp',
#               'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6',
#               'valboc','val3','val5','val6','val8','val9','val10',
#               'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
#               'lp1','lp2','coa2','coa3','coa4','pen2','pen3','pen4',
#               'tor2','tor3','tor4','cot2',
#               'rvdw','gammaw','Devdw','alfa','vdw1',
#               'rohb','Dehb','hb1','hb2']

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
             evaluate_ffield=True,scale=1.0,pop=20,regenerate=False,
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
                print(-99999999999.9,file=f)                                   # 得分为-999，需要重新评估

    d              = pd.read_csv(fcsv)
    columns        = d.columns

    for c in columns:                                                ### Check Data
        col_ = c.split()
        if len(col_)>0:
           col = col_[0]
           if col == 'Unnamed:':
              d.drop(c,axis=1,inplace=True)                          ### Delete Unnamed column


    #from sklearn.cluster import KMeans
    #import numpy as np
    # X = np.array([[1, 2], [1, 4], [1, 0],
    #               [10, 2], [10, 4], [10, 0]])
    #kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    #kmeans.labels_
    # array([1, 1, 1, 0, 0, 0], dtype=int32)
    #kmeans.predict([[0, 0], [12, 3]])
    # array([1, 0], dtype=int32)
    #kmeans.cluster_centers_
    # array([[10.,  2.],
    #        [ 1.,  2.]])

    columns        = d.columns
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



