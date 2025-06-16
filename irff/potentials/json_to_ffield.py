#!/usr/bin/env python
from irff.reaxfflib import read_ffield,write_ffield
import json as js
from os import environ,system
from os.path import isfile

''' run this script by:
      ./json_to_ffield.py
    command in bash to convert ffield.json file to LAMMPS ffield 
    and control file format.
'''

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

def jsontoffield():
    lf = open('ffield.json','r')
    j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    mf_layer  = j['mf_layer']
    be_layer  = j['be_layer']
    lf.close()

    spec,bonds,offd,angs,torp,hbs = init_bonds(p_)
    write_ffield(p_,spec,bonds,offd,angs,torp,hbs,
                 m=m_,mf_layer=mf_layer,be_layer=be_layer,
                 libfile='ffield')


if __name__ == '__main__':
   jsontoffield()

