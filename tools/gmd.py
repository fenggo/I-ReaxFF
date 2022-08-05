#!/usr/bin/env python
from irff.md.gulp import write_gulp_in,xyztotraj
#from irff.molecule import compress
from ase.io import read,write
from ase import Atoms
import argh
import argparse
from os import system,popen
import time


def nvt(T=350,time_step=0.1,step=100,gen='poscar.gen',mode='w',lib='reax'):
    atoms = read(gen,index=-1)
    write_gulp_in(atoms,runword='md qiterative conv',
                  T=T,
                  time_step=time_step,
                  tot_step=step,
                  lib=lib)
    print('\n-  running gulp nvt ...')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode=mode)


def nvt_wt(T=350,time_step=0.1,tot_step=100,gen='poscar.gen',mode='w',wt=10):
    A = read(gen,index=-1)
    # A = press_mol(A)
    write_gulp_in(A,runword='md conv qiterative ',
                  T=T,
                  time_step=time_step,
                  tot_step=tot_step,
                  lib='reax')
    print('\n-  running gulp nvt ...')
    system('nohup gulp<inp-gulp>gulp.out 2>&1 &')
    nan_ = get_status(wt)
    xyztotraj('his.xyz',mode=mode)


def get_status(wt):
    while True:
          time.sleep(wt)
          line = popen('cat gulp.out | tail -3').read()
          if line.find('NaN')>=0:
             lines = popen('ps -aux | grep "gulp"').readlines()
             for line in lines:
                 l = line.split()
                 if l[7]=='R+' or l[7]=='R':
                    system('kill %s' %l[1])
             return True
          elif line.find('Job Finished')>=0:
             return False


def npt(T=350,time_step=0.1,tot_step=10.0):
    A = read('packed.gen')
    write_gulp_in(A,runword='md conp qiterative',
                  T=T,
                  time_step=time_step,
                  tot_step=tot_step,
                  lib='reax')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz')


def opt(T=350,gen='siesta.traj',mode='w'):
    A = read(gen,index=-1)
    A = press_mol(A)
    write_gulp_in(A,runword='opti conv qiterative',
                  T=T,
                  lib='reax')
    print('\n-  running gulp opt ...')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode='w')


def optl(T=350,gen='poscar.gen',mc=500):
    A = read(gen)
    # A = press_mol(A)
    write_gulp_in(A,runword='opti conp qiterative stre atomic_stress',
                  T=T,maxcyc=mc,
                  lib='reax')
    print('\n-  running gulp optimize ...')
    system('/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode='w',scale=False)


if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [opt,optl,nvt,npt])
   argh.dispatch(parser)


