#!/usr/bin/env python
import argh
import argparse
from os import system,popen
import time
from ase.io import read # ,write
# from ase import Atoms
from irff.md.gulp import write_gulp_in,xyztotraj,get_md_results,plot_md


def nvt(T=350,time_step=0.1,step=100,gen='poscar.gen',i=-1,mode='w',c=0,
        x=1,y=1,z=1,lib='reaxff_nn'):
    atoms = read(gen,index=i)*(x,y,z)
    write_gulp_in(atoms,runword='md qiterative conv',
                  T=T,
                  time_step=time_step,
                  tot_step=step,
                  lib=lib)
    print('\n-  running gulp nvt ...')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode=mode,traj='md.traj', checkMol=c,scale=False)


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


# def npt(T=350,time_step=0.1,tot_step=10.0):
#     A = read('packed.gen')
#     write_gulp_in(A,runword='md conp qiterative',
#                   T=T,
#                   time_step=time_step,
#                   tot_step=tot_step,
#                   lib='reax')
#     system('gulp<inp-gulp>gulp.out')
#     xyztotraj('his.xyz')


def opt(T=350,gen='siesta.traj',step=200,i=-1,l=0,c=0,
        x=1,y=1,z=1):
    A = read(gen,index=i)*(x,y,z)
    # A = press_mol(A)
    if l==0:
       runword='opti conv qiterative'
    elif l==1:
       runword= 'opti conp qiterative stre atomic_stress'

    write_gulp_in(A,runword=runword,
                  T=T,maxcyc=step,
                  lib='reaxff_nn')
    print('\n-  running gulp optimize ...')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode='w',traj='md.traj',checkMol=c,scale=False) 

def traj(inp='inp-gulp'):
    xyztotraj('his.xyz',inp=inp,mode='w',scale=False)

def plot(out='out'):
    E,Epot,T,P = get_md_results(out=out)
    plot_md(E,Epot,T,P,show=True)

def w(T=350,gen='siesta.traj',step=200,i=-1,l=0,c=0,
        x=1,y=1,z=1):
    A = read(gen,index=i)*(x,y,z)
    # A = press_mol(A)
    if l==0:
       runword='opti conv qiterative' # conjugate
    elif l==1:
       runword= 'opti conp qiterative stre atomic_stress'

    write_gulp_in(A,runword=runword,
                  T=T,maxcyc=step,
                  lib='reaxff_nn')
    print('\n-  write gulp input ...')

if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [opt,nvt,plot,traj,w])
   argh.dispatch(parser)


