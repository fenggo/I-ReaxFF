#!/usr/bin/env python
from absl import app
from absl import flags
import argh
import argparse
import json as js
from irff.reaxfflib import read_ffield #,write_lib


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

def select_elements(elements=['O','H']):
    # p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield-C')
    elements.append('X')
    with open('ffield.json','r') as fj:
         j = js.load(fj)
         p = {}
         for key in j['p']:
             k = key.split('_')
             lk = len(k)
             if lk==1:
                p[key] = j['p'][key] 
             elif lk>1:
                kk = k[1]
                k_ = kk.split('-')
                
                app = True
                for kkk in k_:
                    if kkk not in elements:
                       app = False
                if app:
                   p[key] = j['p'][key] 
             else:
                raise RuntimeError('-  Unepected error occured!')
         j['p'] = p
    with open('ffield_new.json','w') as fj:
         js.dump(j,fj,sort_keys=True,indent=2)


def add_elements(element,ffield='ffield'):
    ''' add elements from another ReaxFF ffield file'''
    p_,zpe,spec_,bonds_,offd_,angs_,tors_,hbs_= read_ffield(libfile=ffield)
    # mass = {'C':12.0,'H':1.0,'O':16.0,'N':14.0}
    pspec =['rosi','val','mass','rvdw','Devdw','gamma','ropi','vale',
           'alfa','gammaw','valang','ovun5','chi','mu','atomic',
           'ropp','lp2','boc4','boc3','boc5',
           'ovun2','val3','valboc','val5']
    pbond = ['Desi','Depi','Depp','be1','bo5','corr13','bo6','ovun1',
             'be2','bo3','bo4','bo1','bo2','ovcorr']
    pofd  = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
    pang  = ['theta0','val1','val2','coa1','val7','pen1','val4']
    ptor  = ['V1','V2','V3','tor1','cot1']
    phb   = ['rohb','Dehb','hb1','hb2']

    with open('ffield.json','r') as fj:
         j = js.load(fj)
         spec,bonds,offd,angs,tors,hbs = init_bonds(j['p'])

    sp_all = spec + [element]
    for key in pspec:
        k = key+'_'+element
        j['p'][k] = p_[k] 

    bonds_new = set()
    offd_new  = set()
    e = element
    for sp in spec:
        if sp==e:
           continue
        bd  = sp +'-' + e
        bd_ = e +'-' + sp
        if bd in bonds_: 
           bonds_new.add(bd)
        elif bd_ in bonds_: 
           bonds_new.add(bd_)
        else:
           print('Error: {:s} or {:s} not found in ffield!'.format(bd,bd_))

        if bd in offd_: 
           offd_new.add(bd)
        elif bd_ in bonds_: 
           offd_new.add(bd_)
        else:
           print('Error: {:s} or {:s} not found in ffield!'.format(bd,bd_)) 
        bonds_new.add(e +'-' + e)

    for bd in bonds_new:
        for key in pbond:
            k = key+'_'+bd
            j['p'][k] = p_[k] 

    #print(offd_new)
    for bd in offd_new:
        for key in pofd:
            k = key+'_'+bd
            #print(k)
            j['p'][k] = p_[k] 

    angs_new = set()
    for sp1 in sp_all:
        for sp2 in sp_all:
            for sp3 in sp_all:
                if element in [sp1,sp2,sp3]:
                   ang = sp1 + '-' +sp2 + '-' +sp3
                   ang_= sp3 + '-' +sp2 + '-' +sp1
                   if ang in angs_:
                      angs_new.add(ang)
                   elif ang_ in angs_:
                      angs_new.add(ang_)
                   else:
                      angs_new.add(ang_)
                      #  print(sp2)
                      #  if sp2=='H':
                      #     angs_new.add(ang_)
                      #  else:
                      #     print('Error: {:s} or {:s} not found in ffield!'.format(ang,ang_)) 

    for ang in angs_new:
        for key in pang:
            k = key+'_'+ang
            if k in p_:
               j['p'][k] = p_[k] 
            else:
               j['p'][k] = 0.0 

    tors_new = set()
    for sp1 in sp_all:
        for sp2 in sp_all:
            for sp3 in sp_all:
                for sp4 in sp_all:
                    if element in [sp1,sp2,sp3,sp4]:
                       tor  = sp1 + '-' +sp2 + '-' +sp3 +'-'+sp4
                       tor_ = sp4 +'-'+sp3 + '-' +sp2 + '-' +sp1
                       torx = 'X' + '-' +sp2 + '-' +sp3 +'-'+'X'
                       torx_= 'X' + '-' +sp3 + '-' +sp2 +'-'+'X'
                       if tor in tors_:
                          tors_new.add(tor)
                       elif tor_ in tors_:
                          tors_new.add(tor_)
                       elif torx in tors_:
                          tors_new.add(torx)
                       elif torx_ in tors_:
                          tors_new.add(torx_)
                       else:
                          tors_new.add(torx)
                          #   if sp2=='H' or sp3=='H':
                          #      tors_new.add(torx)
                          #   else:
                          #      print('Error: {:s} or {:s} not found in ffield!'.format(tor,tor_)) 
    for tor in tors_new:
        for key in ptor:
            k = key+'_'+tor
            if k in p_:
               j['p'][k] = p_[k] 
            else:
               j['p'][k] = 0.0   

 
    hbs_new = set()
    for sp1 in sp_all:
        for sp2 in sp_all:
            for sp3 in sp_all:
                if element in [sp1,sp2,sp3] and sp2=='H' and sp1!='H' and sp3!='H':
                   hb = sp1 + '-' +sp2 + '-' +sp3
                   #if hb in hbs_:
                   hbs_new.add(hb)
                   #else:
                      #print('Error: {:s} not found in ffield!'.format(hb)) 
    hbdic = {'rohb':2.0,'Dehb':-5.0,'hb1':2.9784,'hb2':2.8122}
    for hb in hbs_new:
        for key in phb:
            k = key+'_'+hb
            if k in p_:
               j['p'][k] = p_[k] 
            else:
               j['p'][k] = hbdic[key] 

    with open('ffield.json','w') as fj:
         js.dump(j,fj,sort_keys=True,indent=2)


if __name__ == "__main__":
  # app.run(select_elements)
  select_elements(['O','H'])
