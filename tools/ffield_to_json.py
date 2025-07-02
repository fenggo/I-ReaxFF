#!/usr/bin/env python
import sys
import argparse
import numpy as np
import json as js
from os import environ,system
from ase.io import read
from irff.reaxfflib import read_ffield,write_lib


def ffieldtojson(ffield):
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_ffield(libfile=ffield)

    fj = open('ffield.json','w')
    # j = {'p':p,'m':[],'bo_layer':[],'zpe':[]}
    for key in p:
        # print(p[key])
        if isinstance(p[key],np.float32):
           p[key]= float(p[key])
    j = {'p':p,'m':None,
         'EnergyFunction':0,
         'VdwFunction':0,
         'MessageFunction':0, 
         'messages':1,
         'mf_layer':None,
         'be_layer':None,
         'vdw_layer':None,
         'MolEnergy':{} ,
         'rcut':None,
         'rEquilibrium':None,
         'rcutBond':None}
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()


if __name__ == '__main__':
   help_= 'Run with commond: ./ffield_to_json.py  --f=ffield '
   parser = argparse.ArgumentParser(description=help_)
   parser.add_argument('--f',default='ffield',type=str, help='ffield file name')
   args = parser.parse_args(sys.argv[1:])

   ffieldtojson(args.f)

