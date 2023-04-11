#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json as js
import matplotlib.pyplot as plt
import argh
import argparse
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.io import read
from irff.tools.vdw import vdw

with open('ffield.json','r') as lf:
     j = js.load(lf)
     p = j['p']

bd = 'O-Fe'
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

r   = np.linspace(2.0,3.0,50)
ev  = vdw(r,Devdw=Devdw,gamma=gamma,gammaw=gammaw,vdw1=vdw1,rvdw=rvdw)
# print(ev)

plt.figure()     
plt.plot(r,ev,alpha=0.8,linewidth=2,linestyle='-',color='r',
         label=r'$E_{vdw}$')
plt.legend(loc='best',edgecolor='yellowgreen')
#plt.savefig('vdw_energy_{:s}.pdf'.format(bd))
plt.show()
plt.close()
