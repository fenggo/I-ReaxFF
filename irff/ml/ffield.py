#!/usr/bin/env python
#from irff.reaxfflib import read_ffield,write_ffield
import json as js
import csv
import pandas as pd
from os.path import isfile
from os import listdir


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


def update_ffield(p_,ffield):
    unit   = 4.3364432032e-2
    punit  = ['Desi','Depi','Depp','lp2','ovun5','val1',
              'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb'] # ,'hb1'
    with open(ffield,'r') as lf:
         j = js.load(lf)

    if type(p_)==dict:
       keys = p_.keys()
    else:
       keys = p_.index

    for key in keys:
        k = key.split('_')[0]
        if key in p_:
           p = p_[key]*unit if k in punit else p_[key]
           j['p'][key] = p         

    with open('ffield.json','w') as fj:
         js.dump(j,fj,sort_keys=True,indent=2)


def ffield_to_csv(ffield='ffield.json',parameters=None,fcsv='ffield.csv'):
    with open(ffield,'r') as lf:
         j = js.load(lf)
         p_ = j['p']
         m_ = j['m']
         bo_layer = j['bo_layer']
    spec,bonds,offd,angs,torp,hbs = init_bonds(p_)
    # cons   = ['val','vale','lp3','cutoff','acut','hbtol']
    p_g    = ['boc1','boc2','coa2','ovun6','lp1', # 'lp3',
              'ovun7','ovun8','val6','val9','val10','tor2',
              'tor3','tor4','cot2','coa4','ovun4', 
              'ovun3','val8','coa3','pen2','pen3','pen4','vdw1']
              # 'cutoff','acut','hbtol'
              
    p_spec = ['valboc','ovun5',
              'lp2','boc4','boc3','boc5','rosi','ropi','ropp',
              'ovun2','val3','val5', 'atomic',
              'gammaw','gamma','chi','mu', # 'mass' ,
              'Devdw','rvdw','alfa','valang','val','vale']
    p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
    p_bond = ['Desi','Depi','Depp','bo5','bo6','ovun1',
              'be1','be2','bo3','bo4','bo1','bo2','corr13','ovcorr'] 
    p_ang  = ['theta0','val1','val2','coa1','val7','val4','pen1']
    p_tor  = ['V1','V2','V3','tor1','cot1']
    p_hb   = ['Dehb','rohb','hb1','hb2']

    p_name = ['']
    for g in p_g: 
        if parameters is None:
           p_name.append(g)
        else:
           if g in parameters:
              p_name.append(g)

    for s in spec:
        for k in p_spec:
            if parameters is None:
               p_name.append(k+'_'+s)
            else:
               if k in parameters:
                  p_name.append(k+'_'+s)

    for bd in bonds:
        for k in p_bond:
            if parameters is None:
               p_name.append(k+'_'+bd)
            else:
               if k in parameters:
                  p_name.append(k+'_'+bd)

    for bd in offd:
        for k in p_offd:
            if parameters is None:
               p_name.append(k+'_'+bd)
            else:
               if k in parameters:
                  p_name.append(k+'_'+bd)

    for ang in angs:
        a = ang.split('-')
        if a[1]!='H':
           for k in p_ang:
               if parameters is None:
                  p_name.append(k+'_'+ang)
               else:
                  if k in parameters:
                     p_name.append(k+'_'+ang)

    for tor in torp:
        a = tor.split('-')
        if a[1]!='H' and a[2]!='H':
           for k in p_tor:
               if parameters is None:
                  p_name.append(k+'_'+tor)
               else:
                  if k in parameters:
                     p_name.append(k+'_'+tor)

    for hb in hbs:
        for k in p_hb:
            if parameters is None:
               p_name.append(k+'_'+hb)
            else:
               if k in parameters:
                  p_name.append(k+'_'+hb)
                  
    p_name.append('score')

    already_exist = False
    if isfile(fcsv):
       already_exist = True

    if already_exist: 
       fcsv_ = open(fcsv,'a')
    else:
       fcsv_ = open(fcsv,'w')
    csv_write = csv.writer(fcsv_)

    if not already_exist:
       csv_write.writerow(p_name)
    
    row = ['0'] # if already_exist else []
    for key in p_name:
        if key:
           if key=='score':
              s = '{:.4f}'.format(float(-10000.0))
           else:
              s = '{:.4f}'.format(p_[key])
           row.append(s)

    # row.append('{:.4f}'.format(-100000.0))
    csv_write.writerow(row)
    fcsv_.close()


def get_csv_data(fcsv):
    d = pd.read_csv(fcsv)
    # print(d['boc1'])
    return d


def init_bonds(p_):
    spec,bonds,offd,angs,torp,hbs = [],[],[],[],[],[]
    for key in p_:
        # key = key.encode('raw_unicode_escape')
        # print(key)
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           # print(kk)
           if len(kk)==2:
              if kk[0]!=kk[1]:
                 offd.append(k[1])
           elif len(kk)==1:
              spec.append(k[1])
        elif k[0]=='theta0':
           angs.append(k[1])
        elif k[0]=='tor1':
           torp.append(k[1])
        elif k[0]=='rohb':
           hbs.append(k[1])
    return spec,bonds,offd,angs,torp,hbs


# if __name__ == '__main__':
#    ''' use commond like ./gmd.py nvt --T=2800 to run it'''
#    # ffield_to_csv()
#    get_csv_data('ffield.csv')

