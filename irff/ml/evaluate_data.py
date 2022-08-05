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
             evaluate_ffield=False,
             step=1000,print_step=100):
    ''' evaluate the score of the parameter set in csv file '''
    if not isfile(fcsv):
       init_ffield_csv(fcsv,parameters=parameters)
     
    ffield_to_csv(ffield='ffield.json',fcsv=fcsv,
                  parameters=parameters) # 'val

    d              = pd.read_csv(fcsv)
    columns        = d.columns

    for c in columns:                            ### Check Data
        col_ = c.split()
        if len(col_)>0:
           col = col_[0]
           if col == 'Unnamed:':
              d.drop(c,axis=1,inplace=True)         ### Delete Unnamed column

    d_row,d_column = d.shape
    columns        = d.columns
    
    if evaluate_ffield:                          ### whether evoluate the current ffield
       if not model is None:                     ### evoluate the score of current ffield
          model.run(learning_rate=1.0e-4,step=step,print_step=print_step,
                    writelib=1000,close_session=False)
          loss    = model.loss_
          p_      = model.p_
       elif not trainer is None:
          loss,p_ = trainer(step=step,print_step=print_step)
          score   = -loss
       else:
          raise RuntimeError('-  At least one of potential or trainer function is defind!')

       if np.isnan(loss) or np.isinf(loss) or loss>9999999.0:
          loss = 9999999.0
       else:
          new_row = {}
          for item in columns:                    ## 对所有列遍历
              if item:
                 if item != 'score':
                    new_row[item]= [p_[item]]          ## 参数有所变化，重新更新值
          new_row['score'] = [-loss]    
          new_row = pd.DataFrame(new_row)
          d = pd.concat([new_row,d],ignore_index=True) ## 评估当前ffield得分，并加入到数据集中

       for j in range(d_row):
           p=d.loc[j]
           if d.loc[j, 'score'] <= to_evaluate:
              if not model is None: 
                 model.update(p=p)
                 model.run(learning_rate=1.0e-4,step=step,print_step=print_step,
                            writelib=1000,close_session=False)
                 loss = model.loss_
                 p_   = model.p_
              elif not trainer is None:
                 update_ffield(p,'ffield.json')   #### update parameters
                 loss,p_ = trainer(step=step,print_step=print_step)
                 score   = -loss
              else:
                 raise RuntimeError('-  At least one of potential or trainer function is defind!')

              if np.isnan(loss) or np.isinf(loss) or loss>9999999.0:
                 loss = 9999999.0
              else:
                 for item in columns:                    ## 对所有列遍历
                     if item:
                        if item != 'score':
                           d.loc[j, item]= p_[item]         ## 参数有所变化，重新更新值
              d.loc[j, 'score'] = -loss
    d.to_csv(fcsv)
    # model.close()
    return d



