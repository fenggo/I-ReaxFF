#!/usr/bin/env python
from irff.reaxfflib import read_lib,write_lib
from irff.plot.reax_plot import get_p,get_bo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
import argh
import argparse


def get_ebond(bo,be1,be2,De):
    powb = np.power(bo,be2)
    expb = np.exp(be1*(1.0-powb))
    eb   = De*bo*expb*4.3364432032e-2
    return eb


def peb(bd='C-C'):
    p,bonds = get_p('ffield.json')
    bo = np.arange(0.1,1.0,0.05)

    plt.figure()
    plt.ylabel( '%s Bond Energy (eV)' %bd)
    plt.xlabel(r'$Bond-Order$')
    # plt.xlim(0,2.5)
    # plt.ylim(0,1.01)
    # for i,bd in enumerate(bonds):
    
    print(bd,'be1:',p['be1_'+bd],'be2:',p['be2_'+bd])
    eb = get_ebond(bo,p['be1_'+bd],p['be2_'+bd],p['Desi_'+bd])

    plt.plot(bo,eb,label=r'$Bond-Energy$', 
             color='r', linewidth=2, linestyle='-')

    plt.legend()
    plt.savefig('ebond_%s.eps' %bd) 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./plbo.py pb --bd=C-C to run it
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [pbo,peb])
   argh.dispatch(parser)

