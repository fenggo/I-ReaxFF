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
from irff.init_check import init_bonds



def q(gen='packed.gen'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')
    A = read(gen)
    q = qeq(p=p,atoms=A)
    q.calc()


def cp():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')
    p_ = p.copy()
    p_chon,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffieldCHON')


    for key in p_chon:
        if not key in p_:
           # print(key)
           p_[key] = p_chon[key]


    write_lib(p_,spec,bonds,offd,angs,torp,hbs,libfile='ffield_')




if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   # parser = argparse.ArgumentParser()
   # argh.add_commands(parser, [q,cp])
   # argh.dispatch(parser)
   cp()
