#!/usr/bin/env python
from __future__ import print_function
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
# from .mdtodata import MDtoData
from ase.io import read,write
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
#from irff.irff import IRFF
#from irff.reax import ReaxFF
    
      
def dh(traj='siesta.traj',batch_size=1,nn=True,frame=7):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    atoms  = images[frame]
    his    = TrajectoryWriter('tmp.traj',mode='w')
    his.write(atoms=atoms)
    his.close()

    from irff.irff import IRFF
    ir = IRFF(atoms=atoms,
              libfile=ffield,
              nn=nn,
              bo_layer=[9,2])
    ir.get_potential_energy(atoms)
    
    from irff.reax import ReaxFF
    rn = ReaxFF(libfile=ffield,
                direcs={'tmp':'tmp.traj'},
                dft='siesta',
                opt=[],optword='nocoul',
                batch_size=batch_size,
                atomic=True,
                clip_op=False,
                InitCheck=False,
                nn=nn,
                pkl=False,
                to_train=False) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    mol    = 'tmp'
    hblab  = rn.lk.hblab
    hbs    = []

    for hb in ir.hbs:
        hbs.append(list(hb))

    eh_    = rn.get_value(rn.EHB) 
    # eh     = ir.ehb.numpy()
    rhb_   = rn.get_value(rn.rhb) 
    # rhb    = ir.rhb.numpy()

    frhb_  = rn.get_value(rn.frhb) 
    # frhb   = ir.frhb.numpy()

    sin4_  = rn.get_value(rn.sin4) 
    # sin4   = ir.sin4.numpy()

    # exphb1_= rn.get_value(rn.exphb1) 
    # exphb2_= rn.get_value(rn.exphb2) 

    # exphb1 = ir.exphb1.numpy()
    # exphb2 = ir.exphb2.numpy()

    # hbsum_ = rn.get_value(rn.hbsum) 
    # hbsum  = ir.hbsum.numpy()

    for hb in rn.hbs:
        for a_ in range(rn.nhb[hb]):
            a = hblab[hb][a_][1:]
            i,j,k = a

            if a in hbs:
               ai = hbs.index(a)
               # if eh_[hb][a_][0]<-0.000001:
               print('-  %2d%s-%2d%s-%2d%s:' %(i,ir.atom_name[i],j,ir.atom_name[j],k,ir.atom_name[k]),
                   'ehb: %10.8f' %(eh_[hb][a_][0]), 
                   'rhb: %10.8f' %(rhb_[hb][a_][0]), 
                  'frhb: %10.8f' %(frhb_[hb][a_][0]), 
                  'sin4: %10.8f' %(sin4_[hb][a_][0]) )
            # else:
            #    if eh_[hb][a_][0]<-0.00001:
            #       print('-  %2d%s-%2d%s-%2d%s:' %(i,ir.atom_name[i],j,ir.atom_name[j],k,ir.atom_name[k]),
            #          'ehb: %10.8f' %(eh_[hb][a_][0]),
            #        'rhb: %10.8f' %(rhb_[hb][a_][0]), 
            #       'frhb: %10.8f' %(frhb_[hb][a_][0]), 
            #       'sin4: %10.8f' %(sin4_[hb][a_][0]) )
    
                  # for c,e in enumerate(ehb):
                  #     if e<-0.000001:
                  #       i_,j_ = self.hbij[c]
                  #       j_,k_ = self.hbjk[c]
                  #       print('-  %2d%s-%2d%s-%2d%s:' %(i_,self.atom_name[i_],j_,
                  #                      self.atom_name[j_],k_,self.atom_name[k_]),
                  #          'ehb: %10.8f' %e, 
                  #          'rhb: %10.8f' %(rjk[c]), 
                  #          'frhb: %10.8f' %(frhb[c]), 
                  #          'sin4: %10.8f' %(sin4[c]) )

                


if __name__ == '__main__':
   ''' use commond like ./cp.py scale-md --T=2800 to run it'''
   # parser = argparse.ArgumentParser()
   # argh.add_commands(parser, [e,db,dl,da,dt,dh,dr])
   # argh.dispatch(parser)

   