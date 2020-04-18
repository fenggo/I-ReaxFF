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
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield.AlCHNO')

    psoffd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
    # print(torp)
    p_ = p.copy()
    spec.append('F')
    bonds_ = bonds.copy()
    for bd in bonds:
        if 'H' in bd:
            b = bd.replace('H','F')
            bonds_.append(b)

    offd_ = offd.copy()
    for bd in offd:
        if 'H' in bd:
            b = bd.replace('H','F')
            offd_.append(b)

    angs_ = angs.copy()
    for ang in angs:
        if 'H' in ang:
            a = ang.replace('H','F')
            angs_.append(a)

    torp_ = torp.copy()
    for tor in torp:
        if 'H' in tor:
            t = tor.replace('H','F')
            torp_.append(t)


    for key in p:
        k = key.split('_')
        kpre = k[0]
        if len(k)>1:
           kc = k[1]
           # b  = kc.split('-')
           if 'H' in kc:
              kcc = kc.replace('H','F')
              p_[kpre+'_'+kcc] = p_[key]

              # if kpre in poffd:
              #    print(kpre+'_'+kcc)

    write_lib(p_,spec,bonds_,offd_,angs_,torp_,hbs,libfile='ffield')



if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   # parser = argparse.ArgumentParser()
   # argh.add_commands(parser, [i,ii,jj])
   # argh.dispatch(parser)
   resetff()
