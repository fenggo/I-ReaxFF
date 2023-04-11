#!/usr/bin/env python
from __future__ import print_function
from irff.reaxfflib import read_lib,write_lib
# from irff.irnnlib_new import write_lib
from irff.qeq import qeq
from ase.io import read
import argh
import argparse
import json as js
from os import environ,system


def i():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')

    fj = open('ffield.json','w')
    # j = {'p':p,'m':[],'bo_layer':[],'zpe':[]}
    j = {'p':p,'m':None,
         'EnergyFunction':0,
         'MessageFunction':0, 
         'messages':1,
         'bo_layer':None,
         'bf_layer':None,
         'be_layer':None,
         'vdw_layer':None,
         'MolEnergy':None}
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()


def setwb():
    lf = open('ffield.json','r')
    j  = js.load(lf)
    p  = j['p']
    m  = j['m']
    mf = j['MessageFunction']
    ef = j['EnergyFunction']
    messages = j['messages']

     
    bo_layer = j['bo_layer']
    lf.close()
    spec,bonds,offd,angs,torp,hbs = init_bonds(p)

    for bd in bonds:
        j['m']['f1b_'+bd]  = j['m']['f1b']
        j['m']['f1bi_'+bd] = j['m']['f1bi']
        j['m']['f1bo_'+bd] = j['m']['f1bo']
        j['m']['f1w_'+bd]  = j['m']['f1w']
        j['m']['f1wi_'+bd] = j['m']['f1wi']
        j['m']['f1wo_'+bd] = j['m']['f1wo']

        j['m']['feb_'+bd]  = j['m']['feb']
        j['m']['febi_'+bd] = j['m']['febi']
        j['m']['febo_'+bd] = j['m']['febo']
        j['m']['few_'+bd]  = j['m']['few']
        j['m']['fewi_'+bd] = j['m']['fewi']
        j['m']['fewo_'+bd] = j['m']['fewo']

        j['m']['fsib_'+bd]  = j['m']['fsib']
        j['m']['fsibi_'+bd] = j['m']['fsibi']
        j['m']['fsibo_'+bd] = j['m']['fsibo']
        j['m']['fsiw_'+bd]  = j['m']['fsiw']
        j['m']['fsiwi_'+bd] = j['m']['fsiwi']
        j['m']['fsiwo_'+bd] = j['m']['fsiwo']

        j['m']['fpib_'+bd]  = j['m']['fpib']
        j['m']['fpibi_'+bd] = j['m']['fpibi']
        j['m']['fpibo_'+bd] = j['m']['fpibo']
        j['m']['fpiw_'+bd]  = j['m']['fpiw']
        j['m']['fpiwi_'+bd] = j['m']['fpiwi']
        j['m']['fpiwo_'+bd] = j['m']['fpiwo']

        j['m']['fppb_'+bd]  = j['m']['fppb']
        j['m']['fppbi_'+bd] = j['m']['fppbi']
        j['m']['fppbo_'+bd] = j['m']['fppbo']
        j['m']['fppw_'+bd]  = j['m']['fppw']
        j['m']['fppwi_'+bd] = j['m']['fppwi']
        j['m']['fppwo_'+bd] = j['m']['fppwo']
        
    system('mv ffield.json ffield_.json')
    fj = open('ffield.json','w')
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()


 

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


if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   setwb()
 

