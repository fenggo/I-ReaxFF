#!/usr/bin/env python
from os import system,getcwd,chdir
from os.path import exists
import argh
import argparse
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.data import chemical_symbols
from irff.dft.siesta import siesta_md,siesta_opt,write_siesta_in
from irff.molecule import compress,press_mol
from irff.data.mdtodata import MDtoData


def md(ncpu=20,T=300,us='F',tstep=50,dt=1.0,gen='poscar.gen',index=-1):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=index)
    # A = press_mol(A)
    print('\n-  running siesta md ...')
    siesta_md(A,ncpu=ncpu,T=T,dt=dt,tstep=tstep,us=us,
              xcf='GGA',xca='PBE',basistype='split')

def npt(ncpu=20,P=10.0,T=300,us='F',tstep=50,dt=1.0,gen='poscar.gen',index=-1):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=index)
    # A = press_mol(A)
    print('\n-  running siesta npt ...')
    siesta_md(A,ncpu=ncpu,P=P,T=T,dt=dt,tstep=tstep,us=us,opt='NoseParrinelloRahman')

def opt(ncpu=8,T=2500,us='F',gen='poscar.gen',l=0,i=-1):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=i)
    # A = press_mol(A)
    print('\n-  running siesta opt ...')
    vc = 'true' if l else 'false'
    siesta_opt(A,ncpu=ncpu,us=us,VariableCell=vc,
               xcf='GGA',xca='PBE',basistype='split')

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

def x(f='siesta.XV'):
    ''' XV to gen '''
    cell = []
    atoms = []
    element={}
    with open(f,'r') as fv:
         for i,line in enumerate(fv.readlines()):
             if i<=2:
                cell.append(line)
             elif i==3:
                natom = int(line)
             else:
                l = line.split()
                atoms.append(line)
                element[int(l[0])] = int(l[1])

    lk = list(element.keys())
    lk.sort()

    with open('geo.gen','w') as fg:
         print(natom,'S',file=fg)
         for k in lk:
             print(chemical_symbols[element[k]],end=' ',file=fg)
         print(' ',file=fg)
         for i,atom in enumerate(atoms):
             a = atom.split()
             print(i+1,a[0],a[2],a[3],a[4],file=fg)
         print('0.0 0.0 0.0',file=fg)
         for c_ in cell:
             c = c_.split()
             print(c[0],c[1],c[2],file=fg)

def wi(gen='poscar.gen'):
    A = read(gen,index=-1)
    print('\n-  writing siesta input ...')
    write_siesta_in(A,coord='cart', md=False, opt='CG',
                    VariableCell='true', xcf='VDW', xca='DRSLL',
                    basistype='DZP')


if __name__ == '__main__':
   ''' use commond like ./smd.py opt --l=1 --g=*.gen --n=8 to run it
       md : molecular dynamics simulations
       opt: structure optimization
       --l: 1,lattice constant optimize 0,fix the lattice constant
       --g: the structure file
       --n: the number of the CPU cores will be used
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [md,opt,traj,npt,pm,mde,x,wi])
   argh.dispatch(parser)


