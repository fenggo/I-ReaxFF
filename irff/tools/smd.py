#!/usr/bin/env python
from __future__ import print_function
import argh
import argparse
from os import environ,system,getcwd,chdir
from os.path import exists
from irff.siesta import siesta_md,siesta_opt
from irff.gulp import write_gulp_in,xyztotraj
from irff.molecule import compress,press_mol
from irff.mdtodata import MDtoData
from ase.io import read,write
from ase import Atoms
from ase.io.trajectory import Trajectory



def cmd(ncpu=4,T=350,comp=[0.99,1.0,0.999],us='F'):
    system('rm siesta.MDE siesta.MD_CAR')
    A = read('packed.gen')
    # A = press_mol(A)
    if not comp is None:
       if us=='T':
          fx = open('siesta.XV','r')
          lines = fx.readlines()
          fx.close()
          fx = open('siesta.XV','w')
          for i,line in enumerate(lines):
              if i<3:
                 l = line.split()
                 print(float(l[0])*comp[i],float(l[1])*comp[i],float(l[2])*comp[i],
                       l[3],l[4],l[5],file=fx)
              else:
                 print(line[:-1],file=fx)
          fx.close()
       else:
         A = compress(A,comp=comp)
    siesta_md(A,ncpu=ncpu,T=T,dt=0.1,tstep=2000,us=us)


def md(ncpu=20,T=2000,us='F',tstep=100,dt=1.0,gen='poscar.gen'):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=-1)
    # A = press_mol(A)
    print('\n-  running siesta md ...')
    siesta_md(A,ncpu=ncpu,T=T,dt=dt,tstep=tstep,us=us)


def mmd(ncpu=20,T=300,us='F',tstep=100,dt=1.0,gen='strucs.traj'):
    images = Trajectory(gen)
    rdir   = getcwd()
    for i,image in enumerate(images):
        wdir = 'stru_'+str(i)
        system('mkdir '+wdir)
        system('cp *.psf %s/' %wdir)
        chdir(wdir)

        if exists('siesta.MDE') or exists('siesta.MD_CAR'):
           system('rm siesta.MDE siesta.MD_CAR')
        print('\n-  running siesta md for structure %d ...' %i)
        siesta_md(image,ncpu=ncpu,T=T,dt=dt,tstep=tstep,us=us)
        chdir(rdir)


def opt(ncpu=20,T=2500,us='F',tstep=2001,dt=1.0,gen='poscar.gen'):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=-1)
    A = press_mol(A)
    print('\n-  running siesta opt ...')
    siesta_opt(A,ncpu=ncpu,us=us)


def traj():
    cwd = getcwd()
    d = MDtoData(structure='siesta',dft='siesta',direc=cwd,batch=10000)
    d.get_traj()
    d.close()


def nvt(T=350,time_step=0.1,tot_step=5000,gen='siesta.traj',index=-1,mode='w'):
    ''' a gulp MD run '''
    A = read(gen,index=index)
    write_gulp_in(A,runword='md qiterative conv',
                  T=T,
                  time_step=time_step,
                  tot_step=tot_step,
                  lib='reax')
    print('\n-  running gulp nvt ...')
    system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode=mode)


def x(mode='w'):
    xyztotraj('his.xyz',mode=mode)


def pm(gen='gulp.traj',index=-1):
    ''' pressMol '''
    A = read(gen,index=index)
    cell = A.get_cell()
    print(cell)
    A = press_mol(A)
    A.write('poscar.gen')
    del A 


def pd():
    ''' collect number (frame=5) in energy directory, split the frame to traj file 
        evergy (split_batch=1000)
    '''
    direcs={'cn31':'/media/feng/NETAC/siesta/cn3',    
            'cn3':'/media/feng/NETAC/siesta/cn31' 
           }

    trajs = prep_data(label='cn3',direcs=direcs,split_batch=1000,frame=100)
    print('-  trajs:\n',trajs)


if __name__ == '__main__':
   ''' use commond like ./cp.py scale-md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [md,mmd,cmd,opt,traj,nvt,pm,x,pd])
   argh.dispatch(parser)



