#!/usr/bin/env python
from __future__ import print_function
from irff.reaxfflib import read_lib
from irff.qeq import qeq
from ase.io import read
import argh
import argparse
from os import environ,system



def q(gen='packed.gen'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')
    A = read(gen)
    q = qeq(p=p,atoms=A)
    q.calc(A)
    # q.debug()
    print('\n-  Qeq Charges: \n',q.q[:-1])
    # q.debug_gam()



if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [q])
   argh.dispatch(parser)

