#!/usr/bin/env python
from __future__ import print_function
from irff.reaxfflib import read_lib,write_lib
from ase.io import read
import argh
import argparse
import json as js
from os import environ,system
from irff.init_check import init_bonds
import numpy as np


def check_tors(p,spec,torp):
    p_tor  = ['V1','V2','V3','tor1','cot1']  
    tors = []          ### check torsion parameter
    if 'X' in spec:
       spec.remove('X')
    # print(spec)
    for spi in spec:
        for spj in spec:
            for spk in spec:
                for spl in spec:
                    tor = spi+'-'+spj+'-'+spk+'-'+spl
                    if tor not in tors:
                       tors.append(tor)
                       
    for key in p_tor:
        for tor in tors:
            if tor not in torp:
               [t1,t2,t3,t4] = tor.split('-')
               tor1 = t1+'-'+t3+'-'+t2+'-'+t4
               tor2 = t4+'-'+t3+'-'+t2+'-'+t1
               tor3 = t4+'-'+t2+'-'+t3+'-'+t1
               tor4 = 'X'+'-'+t2+'-'+t3+'-'+'X'
               tor5 = 'X'+'-'+t3+'-'+t2+'-'+'X'

               if tor1 in torp:
                  p[key+'_'+tor] = p[key+'_'+tor1]
               elif tor2 in torp:
                  p[key+'_'+tor] = p[key+'_'+tor2]
               elif tor3 in torp:
                  p[key+'_'+tor] = p[key+'_'+tor3]    
               elif tor4 in torp:
                  p[key+'_'+tor] = p[key+'_'+tor4]    
               elif tor5 in torp:
                  p[key+'_'+tor] = p[key+'_'+tor5]    
               else:
                  p[key+'_'+tor] = 0.0
    return p,tors


def resetff():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield.AlCHNOF')
    # print(torp)
    p_ = p.copy()
    Poffd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']

    for bd in bonds:
        b= bd.split('-')
        if b[0]!=b[1]: 
          if bd not in offd:
             # print(bd)
             offd.append(bd)
             for ofd in Poffd:
                 if p_[ofd+'_'+b[0]]>0 and p_[ofd+'_'+b[1]]>0:
                    p_[ofd+'_'+bd] = np.sqrt(p_[ofd+'_'+b[0]]*p_[ofd+'_'+b[1]])
                 else: 
                    p_[ofd+'_'+bd] = -1.0

    for key in p:
        k = key.split('_')
        kpre = k[0]        
        if kpre == 'ropi':
           pcon = k[1]
           pc_  = pcon.split('-')

           if p[key]<=0.0:
              if pcon!='X':
                 if len(pc_) == 2: 
                    p_['bo3_'+pcon] = -50.0
                    p_['bo4_'+pcon] = 0.0000
                 else:
                    p_['bo3_'+pcon+'-'+pcon] = -50.0
                    p_['bo4_'+pcon+'-'+pcon] = 0.0000

        if kpre == 'ropp':
           pcon = k[1]
           pc_  = pcon.split('-')

           if p[key]<=0.0:
              if pcon!='X':
                 if len(pc_) == 2: 
                    p_['bo5_'+pcon] = -50.0
                    p_['bo6_'+pcon] = 0.0000
                 else:
                    p_['bo5_'+pcon+'-'+pcon] = -50.0
                    p_['bo6_'+pcon+'-'+pcon] = 0.0000

    for key in p:
        k = key.split('_')
        kpre = k[0]    
        if kpre == 'ropi' or kpre=='ropp':
           pcon = k[1]
           if kpre=='ropi':
              p_[key] = 0.9*p['rosi_'+pcon]
           elif kpre=='ropp':
              p_[key] = 0.8*p['rosi_'+pcon]
                 
    p_,tors = check_tors(p_,spec,torp)
    write_lib(p_,spec,bonds,offd,angs,tors,hbs,libfile='ffield')



if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   # parser = argparse.ArgumentParser()
   # argh.add_commands(parser, [i,ii,jj])
   # argh.dispatch(parser)
   resetff()

