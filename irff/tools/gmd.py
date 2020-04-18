#!/usr/bin/env python
from __future__ import print_function
from irff.molecule import packmol,press_mol
from irff.gulp import write_gulp_in,xyztotraj
from irff.molecule import compress
from ase.io import read,write
from ase import Atoms
import argh
import argparse
from os import system,popen
import time


def nvt(T=350,time_step=0.1,tot_step=100,gen='packed.gen',mode='w'):
    A = read(gen,index=-1)
    write_gulp_in(A,runword='md qiterative conv',
                  T=T,
                  time_step=time_step,
                  tot_step=tot_step,
                  lib='reax')
    print('\n-  running gulp nvt ...')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode=mode)


def nvt_wt(T=350,time_step=0.1,tot_step=100,gen='packed.gen',mode='w',wt=25):
    A = read(gen,index=-1)
    write_gulp_in(A,runword='md conv qiterative ',
                  T=T,
                  time_step=time_step,
                  tot_step=tot_step,
                  lib='reax')
    print('\n-  running gulp nvt ...')
    system('nohup gulp<inp-gulp>gulp.out 2>&1 &')
    
    thread = popen('ps -aux | grep "gulp"')
    lines  = thread.readlines()
    time.sleep(wt)
    for line in lines:
        l = line.split()
        if l[7]=='R+' or l[7]=='R':
           system('kill %s' %l[1])

    xyztotraj('his.xyz',mode=mode)


def npt(T=350,time_step=0.1,tot_step=10.0):
    A = read('packed.gen')
    write_gulp_in(A,runword='md conp',
                  T=T,
                  time_step=time_step,
                  tot_step=tot_step,
                  lib='reax')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz')


def opt(T=350):
    A = read('packed.gen')
    # A = press_mol(A)
    write_gulp_in(A,runword='opti conv qiterative ',
                  T=T,
                  lib='reax')
    print('\n-  running gulp optimize ...')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode='w')


if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [opt,nvt,npt])
   argh.dispatch(parser)


