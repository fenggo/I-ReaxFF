#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json as js
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.io import read
from irff.tools.vdw import vdw


help_  = 'run commond with: ./plevdw.py --b=H-H --f=ffield.json --rmin=2.5 --rmax=3.5'

parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--bd',default='C-C',type=str, help='which bond')
parser.add_argument('--f',default='ffield.json',type=str, help='force field file name')
parser.add_argument('--rmin',default=2.0,type=float, help='min rvdw')
parser.add_argument('--rmax',default=3.5,type=float, help='max rvdw')
args = parser.parse_args(sys.argv[1:])


with open(args.f,'r') as lf:
     j = js.load(lf)
     p = j['p']



bonds = [args.bd]#,'H-H','C-C','N-N','H-N']
for bd in bonds:
    print('vdw parameters for bond: ',bd)
    b  = bd.split('-') 
    atomi,atomj = b
    gammaw      = np.sqrt(p['gammaw_'+atomi]*p['gammaw_'+atomj])
    gamma       = np.sqrt(p['gamma_'+atomi]*p['gamma_'+atomj])
    alfa        = p['alfa_'+bd]
    #alfa       = np.sqrt(p['alfa_'+atomi]*p['alfa_'+atomj])
    vdw1        = p['vdw1']
    rvdw        = p['rvdw_'+bd]
    # rvdw      = np.sqrt(p['rvdw_'+atomi]*p['rvdw_'+atomj])
    Devdw       = p['Devdw_'+bd]
    # Devdw     = np.sqrt(p['Devdw_'+atomi]*p['Devdw_'+atomj])

    print ('Devdw: {:6.4f} '.format(Devdw))
    print ('Gamma: {:6.4f} '.format(gamma))
    print ('Gammaw: {:6.4f} '.format(gammaw))
    print ('alfa: {:6.4f} '.format(alfa))
    print ('vdw1: {:6.4f} '.format(vdw1))
    print ('rvdw: {:6.4f} '.format(rvdw))

    #rint(Devdw*vdw1)
    # print(rvdw*vdw1)
    r   = np.linspace(args.rmin,args.rmax,100)
    ev  = vdw(r,Devdw=Devdw,gamma=gamma,gammaw=gammaw,vdw1=vdw1,rvdw=rvdw)
     
    search = False
    for i,r_ in enumerate(r):
        if i>0:
           de = ev[i]-el
           # print(r[i],de)
           if de>0.0 and not search:
              print('the rvdw radius (dvdw\dr=0) is : {:f} '.format(r[i-1]))
              search = True
              break
           #if de<0.0 and de>-0.005:
              #print('\nr : {:f}  de: {:f}\n'.format(r[i],de))

        el = ev[i]
        rl = r[i]
    if not search:
       print('searching the vdw radius failed!')
       
    im = np.argmin(ev)
    print('minimum Evdw(r={:f}) = {:f}'.format(r[im],ev[im]))
    plt.figure()     
    plt.plot(r,ev,alpha=0.8,linewidth=2,linestyle='-',color='r',
             label=r'$E_{vdw}$')
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('vdw_energy_{:s}.svg'.format(bd))
    # plt.show()
    plt.close()
