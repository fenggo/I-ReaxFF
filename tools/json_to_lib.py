#!/usr/bin/env python
from irff.reaxfflib import read_ffield,write_lib,write_lib
from irff.qeq import qeq
from ase.io import read
import argh
import argparse
import json as js
from os import environ,system
import csv
import pandas as pd
from os.path import isfile



def ffieldtolib():
    lf = open('ffield.json','r')
    j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    vdw_layer = j['vdw_layer']
    mf_layer  = j['mf_layer']
    be_layer  = j['be_layer']
    lf.close()

    spec,bonds,offd,angs,torp,hbs = init_bonds(p_)
    write_lib(p_,spec,bonds,offd,angs,torp,hbs,m=m_,mf_layer=mf_layer,
              be_layer=be_layer,vdw_layer=vdw_layer,
              libfile='reaxff_nn.lib')


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
   ffieldtolib()

