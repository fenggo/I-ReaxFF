#!/usr/bin/env python
from irff.dft.siesta import siesta_md,siesta_opt
from irff.md.gulp import write_gulp_in,xyztotraj
from irff.molecule import compress,press_mol
from irff.data.mdtodata import MDtoData
from ase.io import read,write
from ase import Atoms
from ase.io.trajectory import Trajectory
import argh
import argparse
from os import environ,system,getcwd,chdir
from os.path import exists


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


def md(ncpu=20,T=300,us='F',tstep=50,dt=1.0,gen='poscar.gen',index=-1):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=index)
    # A = press_mol(A)
    print('\n-  running siesta md ...')
    siesta_md(A,ncpu=ncpu,T=T,dt=dt,tstep=tstep,us=us)


def npt(ncpu=20,P=10.0,T=300,us='F',tstep=50,dt=1.0,gen='poscar.gen',index=-1):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=index)
    # A = press_mol(A)
    print('\n-  running siesta npt ...')
    siesta_md(A,ncpu=ncpu,P=P,T=T,dt=dt,tstep=tstep,us=us,opt='NoseParrinelloRahman')


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


def opt(ncpu=8,T=2500,us='F',tstep=2001,dt=1.0,
        gen='poscar.gen',VariableCell='false'):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=-1)
    A = press_mol(A)
    print('\n-  running siesta opt ...')
    siesta_opt(A,ncpu=ncpu,us=us,VariableCell=VariableCell)


def traj():
    cwd = getcwd()
    d = MDtoData(structure='siesta',dft='siesta',direc=cwd,batch=10000)
    d.get_traj()
    d.close()

       
def pm(gen='siesta.traj',index=-1):
    ''' pressMol '''
    A = read(gen,index=index)
    cell = A.get_cell()
    print(cell)
    A = press_mol(A)
    A.write('poscar.gen')
    del A 


def mde(equil=250):
    t    = []
    e    = []
    p    = []

    with open('siesta.MDE','r') as f:
        for i,line in enumerate(f.readlines()):
            if i>equil:
               l = line.split()
               if len(l)>0:
                  e.append(float(l[2]))
                  t.append(float(l[1]))
                  p.append(float(l[5]))

    tmean = np.mean(t)
    pmean = np.mean(p)*0.1

    print(' * Mean Temperature: %12.6f K' %tmean)
    print(' * Mean Pressure: %12.6f GPa' %pmean)
    # return e,t,p,tmean,pmean


if __name__ == '__main__':
   ''' use commond like ./cp.py scale-md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [md,mmd,cmd,opt,traj,npt,pm,mde])
   argh.dispatch(parser)



