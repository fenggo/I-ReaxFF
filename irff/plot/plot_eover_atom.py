#!/usr/bin/env python
# coding: utf-8
import numpy as np
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units
from ase.visualize import view
from irff.irff_np import IRFF_NP
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from irff.AtomOP import AtomOP


atoms=read('c2h4.gen')
view(atoms)


# In[105]:


def over(i=0,j=2,ffield='ffield.json',nn='T',gen='poscar.gen'):
    atoms = read(gen)
    ao = AtomOP(atoms)
    images = ao.stretch([[i,j]],nbin=50,wtraj=False)
    # view(images)
    
    nn_=True if nn=='T'  else False
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 rcut=None,
                 nn=nn_)

    ir.calculate_Delta(atoms)
    natom = ir.natom

    r_,eb,bosi,bop_si,bop,bop_pi,bop_pp,bo = [],[],[],[],[],[],[],[]
    eba,eo,dlpi,dlpj,ev,boe = [],[],[],[],[],[]
    esi,epi,epp = [],[],[]
    Di,Dj = [],[]
    Dpi   = []
   
    for atoms in images:
        positions = atoms.positions
        v = positions[j] - positions[i]
        r = np.sqrt(np.sum(np.square(v)))
        
        ir.calculate(atoms)
        r_.append(ir.r[i][j])
        eb.append(ir.ebond[i][j])
        eba.append(ir.ebond[i][j] + ir.eover[i] + ir.Evdw) 
        ev.append(ir.Evdw)
        eo.append(ir.eover[j]) 
        # print(ir.so[j],ir.eover[j])

        dlpi.append(ir.Delta_lpcorr[i])
        dlpj.append(ir.Delta_lpcorr[j])
        Di.append(ir.Delta[i])
        Dj.append(ir.Delta[j])
        Dpi.append(ir.Dpil[j])

    fig, ax = plt.subplots() 
    ax.plot(r_,eo,label=r'$E_{over}$(%s%d)' %(ir.atom_name[j],j), 
             color='r', linewidth=2, linestyle='-')
    
#     fig, ax = plt.subplots(2,1,2) 
#     plt.plot(r_,dlpj,label=r'$\Delta_{lp}$(%s%d)' %(ir.atom_name[j],j), 
#              color='b', linewidth=2, linestyle='-') # Dpil
    plt.legend(loc='best',edgecolor='yellowgreen')

    # plt.savefig('Eover.eps') 
    plt.show()
    plt.close()


# In[106]:


over(i=0,j=2,ffield='ffield.json',nn='T',gen='c2h4.gen')




# In[97]:


def relu(x):
    return np.where(x>0.0,x,0.0)

def sigmoid(x):
    s = 1.0/(1.0+np.exp(-x))
    return s

def eover(Delta,vale=1.0,val=1.0,lp1=27.0,lp2=9.7,
                ovun2=-6.8,ovun3=0.5,ovun4=3.9,
                ovun5=18,ovun6=0.5,ovun7=120,ovun8=4.7):
    NLPOPT  = 0.5*(vale - val)
    Delta_e = 0.5*(Delta - vale)

    DE      = relu(-np.ceil(Delta_e))  # number of lone pair electron
    nlp     = DE + np.exp(-lp1*4.0*np.square(1.0+Delta_e+DE))

    Delta_lp= NLPOPT-nlp
    Dv      = Delta - val
    Dlp     = Dv - Delta_lp
    Dpil    = 1.0 # np.sum(np.expand_dims(self.Dlp,axis=0)*(self.bopi+self.bopp),1)
    explp   = 1.0+np.exp(-75.0*Delta_lp)
    elone   = lp2*Delta_lp/explp
    
    lpcorr= Delta_lp/(1.0+ovun3*np.exp(ovun4*Dpil))
    Delta_lpcorr = Dv - lpcorr

    otrm1 = 1.0/(Delta_lpcorr+val)
    otrm2 = 1.0/(1.0+np.exp(ovun2*Delta_lpcorr))
    eo = otrm1*Delta_lpcorr*otrm2
    
    expeu1 = np.exp(ovun6*Delta_lpcorr)
    eu1    = sigmoid(ovun2*Delta_lpcorr)

    expeu3 = np.exp(ovun8*Dpil)
    eu2    = 1.0/(1.0+ovun7*expeu3)
    eu     = -ovun5*(1.0-expeu1)*eu1*eu2
    return eo,eu,Delta_lpcorr   


# In[102]:


Delta = np.linspace(0.5,2.0,50)
eo,eu,dlc = eover(Delta,vale=1.0,val=1.0,lp1=16.0,lp2=1.0,
                  ovun2=-6.8,ovun3=0.5,ovun4=3.9,
                  ovun5=18,ovun6=0.5,ovun7=120,ovun8=4.7)  

fig, ax = plt.subplots()
# ax = fig.add_subplot(2,1,1)
# plt.plot(Delta,eo,label=r'$\eover$', 
#          color='r', linewidth=2, linestyle='-')
ax.plot(Delta,eo,label=r'$E_{over}$', 
         color='r', linewidth=2, linestyle='-')
plt.show()
plt.close()

fig, ax = plt.subplots()
# ax = fig.add_subplot(2,1,2)
ax.plot(Delta,eu,label=r'$E_{under}$', 
         color='r', linewidth=2, linestyle='-')
plt.legend(loc='best',edgecolor='yellowgreen')
# plt.savefig('eover.pdf') 
plt.show()
plt.close()

