#!/usr/bin/env python
from irff.reaxfflib import read_lib,write_lib
# from irff.irnnlib_new import write_lib
from irff.qeq import qeq
from ase.io import read
import argh
import argparse
import json as js
from os import environ,system


def checkoffd():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')

    for key in p:             # check minus ropi ropp parameters
       k = key.split('_')
       if k[0] == 'ropi':
          bd = k[1]
          b  = bd.split('-')
          if len(b) == 1:
             bd_ = b[0]+'-'+b[0]
          else:
             bd_ = bd

          if p['ropi_'+bd] < 0.0:
             p['ropi_'+bd] = 0.3*p['rosi_'+bd]
             p['bo3_'+bd_] = -50.0
             p['bo4_'+bd_] = 0.0
          if p['ropp_'+bd] < 0.0:
             p['ropp_'+bd] = 0.2*p['rosi_'+bd]
             p['bo5_'+bd_] = -50.0
             p['bo6_'+bd_] = 0.0

    write_lib(p,spec,bonds,offd,angs,torp,hbs,libfile='ffield-new')



if __name__ == '__main__':
   checkoffd()

