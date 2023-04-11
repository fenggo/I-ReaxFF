#!/usr/bin/env python
from irff.reaxfflib import read_lib,write_lib
# from irff.irnnlib_new import write_lib
from irff.qeq import qeq
from ase.io import read
import argh
import argparse
import json as js
from os import environ,system


def ffieldtojson():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')

    fj = open('ffield.json','w')
    # j = {'p':p,'m':[],'bo_layer':[],'zpe':[]}
    j = {'p':p,'m':None,
         'EnergyFunction':0,
         'BOFunction':0,
         'VdwFunction':0,
         'MessageFunction':0, 
         'messages':1,
         'bo_layer':None,
         'mf_layer':None,
         'be_layer':None,
         'vdw_layer':None,
         'MolEnergy':{} ,
         'rcut':None,
         'rEquilibrium':None,
         'rcutBond':None}
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
   ffieldtojson()

