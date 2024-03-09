#!/usr/bin/env python
from irff.reaxfflib import read_ffield,write_lib
# from irff.irnnlib_new import write_lib
from irff.qeq import qeq
from ase.io import read
import argh
import argparse
import json as js
from os import environ,system


def ffieldtojson():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_ffield(libfile='ffield')

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


if __name__ == '__main__':
   ffieldtojson()

