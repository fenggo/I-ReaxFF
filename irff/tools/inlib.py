#!/usr/bin/env python
from __future__ import print_function
from irff.irnnlib import read_lib,write_lib
# from irff.irnnlib_new import write_lib
from irff.qeq import qeq
from ase.io import read
import argh
import argparse
import json as js
from os import environ,system
from irff.init_check import init_bonds



def q(gen='packed.gen'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')
    A = read(gen)
    q = qeq(p=p,atoms=A)
    q.calc()



def i():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')

    fj = open('ffield.json','w')
    j = {'p':p,'m':[],'eb_layer':[]}
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()


def ii():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')
    write_lib(p_,spec,bonds,offd,angs,torp,hbs,libfile='ffield_')



def j():
    lf = open('ffield.json','r')
    j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    bo_layer_ = j['bo_layer']
    lf.close()

    spec,bonds,offd,angs,torp,hbs = init_bonds(p_)
    write_lib(p_,spec,bonds,offd,angs,torp,hbs,libfile='ffield')


def jj():
    lf = open('ffield.json','r')
    j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    bo_layer = j['bo_layer']
    ea_layer = j['ea_layer']
    eb_layer = j['eb_layer']
    lf.close()

    spec,bonds,offd,angs,torp,hbs = init_bonds(p_)
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')

    system('mv ffield.json ffield_.json')
    fj = open('ffield.json','w')
    j = {'p':p,'m':m_,'bo_layer':bo_layer,'ea_layer':ea_layer,'eb_layer':eb_layer}
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()



def init_bonds(p_):
    spec,bonds,offd,angs,torp,hbs = [],[],[],[],[],[]
    for key in p_:
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           if len(kk)==2:
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
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [q,i,ii,j,jj])
   argh.dispatch(parser)

