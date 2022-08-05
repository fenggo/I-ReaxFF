#!/usr/bin/env python
#from irff.reaxfflib import read_ffield,write_ffield
from irff.ml.ffield import ffield_to_csv

# def init_ffield_csv():
#     # path='/home/miao/PycharmProjects/scsslh'
#     files=os.listdir(path)
#     for f in files:
#         if f.startswith('ffield') and f.endswith('.json'):
#            ffield_to_csv(ffield=f, fcsv='ffield_bo.csv',
#                          parameters=['boc1','boc2',
#                                      'boc3','boc4','boc5',
#                                      'rosi','ropi','ropp','Desi','Depi','Depp',
#                                      'be1','be2','bo1','bo2','bo3','bo4','bo5','bo6',
#                                      'valboc','val3','val5','val6','val8','val9','val10',
#                                      'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
#                                      'lp1','lp2','coa2','coa3','coa4','pen2','pen3','pen4',
#                                      'tor2','tor3','tor4','cot2',
#                                      'rvdw','gammaw','Devdw','alfa','vdw1']) # 'valang','val','vale',

if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   parameters = [# 'boc1','boc2','boc3','boc4','boc5',
              'rosi','ropi','ropp','Desi','Depi','Depp',
              'bo1','bo2','bo3','bo4','bo5','bo6',#'be1','be2',
              'theta0',#'valang','valboc',
              'val1','val2','val3','val4',
              'val5','val6','val7','val8','val9','val10',
              'ovun1','ovun2','ovun3','ovun4','ovun5','ovun6','ovun7','ovun8',
              'lp1','lp2',
              'coa2','coa3','coa4',
              'pen1','pen2','pen3','pen4',
              'tor2','tor3','tor4','cot1','cot2',
              'V1','V2','V3',
              'rvdw','gammaw','Devdw','alfa','vdw1']  # 进行机器学习优化
   ffield_to_csv(ffield='ffield.json',parameters=parameters,fcsv='ffield_ml.csv')
   # get_csv_data('ffield.csv')

