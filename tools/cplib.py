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


def cpl():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield-FromPaperSort')
    p_ = {}
    specs = ['Al','C','H']
    for key in p:
        k = key.split('_')
        if len(k)==1:
           p_[key] = p[key]
        else:
           bd = k[1]
           b = bd.split('-')
           if len(b)==1:
              if b[0] in specs:
                 if b[0] == 'H':
                    p_[k[0]+'_'+'F'] = p[key]
                 else:
                    p_[key] = p[key]
           elif len(b)==2:
              if b[0] in specs and b[1] in specs:
                 if b[0] == 'H' or b[1] == 'H':
                    bd_ = bd.replace('H','F')
                    p_[k[0]+'_'+bd_] = p[key]
                 else:
                    p_[key] = p[key]
           elif len(b)==3:
              if b[0] in specs and b[1] in specs and b[2] in specs:
                 if b[0] == 'H' or b[1] == 'H' or b[2] == 'H':
                    bd_ = bd.replace('H','F')
                    p_[k[0]+'_'+bd_] = p[key]
                 else:
                    p_[key] = p[key]

           elif len(b)==4:
              if b[0] in specs and b[1] in specs and b[2] in specs and b[3] in specs:
                 if b[0] == 'H' or b[1] == 'H' or b[2] == 'H' or b[3] == 'H':
                    bd_ = bd.replace('H','F')
                    p_[k[0]+'_'+bd_] = p[key]
                 else:
                    p_[key] = p[key]
    spec_ = ['C','F','Al']
    # write_lib(p_,spec_,bonds,offd,angs,torp,hbs,libfile='ffield_')
    with open('ffield.json','w') as fj:
         j = {'p':p_,'m':None,
              'EnergyFunction':0,
              'MessageFunction':0, 
              'messages':1,
              'bo_layer':None,
              'bf_layer':None,
              'be_layer':None,
              'vdw_layer':None,
              'MolEnergy':None}
         js.dump(j,fj,sort_keys=True,indent=2)



if __name__ == '__main__':
   ''' use commond like ./cplib.py cpl --T=2800 to run it'''
   # parser = argparse.ArgumentParser()
   # argh.add_commands(parser, [q,cpl])
   # argh.dispatch(parser)
   cpl()
