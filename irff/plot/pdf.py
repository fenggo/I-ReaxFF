#!/usr/bin/env python
from __future__ import print_function
from irff.mdtodata import MDtoData
from ase.io import read
import argh
import argparse
from os import getcwd
import matplotlib.pyplot as plt
import numpy as np


def pdf(traj='siesta.traj'):
    rcut = 4.0
    cwd = getcwd()
    d1 = MDtoData(structure='box',direc=cwd+'/siesta.traj',
                  dft='ase',atoms=None,
                  batch=100,minib=100,
                  nindex=[])
    bs1,hs1 = d1.pdf(bins=0.01,rcut=4.0,pdf_plot=False)

    d2 = MDtoData(structure='box',direc=cwd+'/gulp.traj',
                  dft='ase',atoms=None,
                  batch=100,minib=100,
                  nindex=[])
    bs2,hs2 = d2.pdf(bins=0.01,rcut=4.0,pdf_plot=False)

    for bd in bs2:
        plt.figure()   
        plt.ylabel('Pair Distribution Function')
        plt.xlabel(r'Radius $(\AA)$')
        plt.xlim(0,rcut)
        plt.ylim(0,max(np.max(hs1[bd]),np.max(hs2[bd]))+0.01)
        # plt.hist(rbd_,bin_=32,density=1,alpha=0.01,label='%s' %bd)

        ax = plt.gca()

        # ax.yaxis.set_ticks_position('right')
        # ax.xaxis.set_ticks_position('bottom')

        # ax.spines['left'].set_color('none')
        # ax.spines['top'].set_color('none')

        plt.plot(bs1[bd],hs1[bd],alpha=0.01,ls='-.',color='red',
                 label='%s by SIESTA' %bd)
        plt.plot(bs2[bd],hs2[bd],alpha=0.01,ls='--',color='blue',
                 label='%s by ReaxFF' %bd)

        plt.legend(loc='best',edgecolor='yellowgreen') # lower left
        plt.savefig('com_%s_bh.eps' %bd,transparent=True) 
        plt.close() 



if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   # parser = argparse.ArgumentParser()
   # argh.add_commands(parser, [h])
   # argh.dispatch(parser)
   pdf()

   