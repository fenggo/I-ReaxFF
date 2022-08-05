#!/usr/bin/env python
from __future__ import print_function
from irff.reax import ReaxFF
from irff.irff_np import IRFF_NP
import numpy as np
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
import argh
import argparse


def plbd(traj='md.traj',
         batch_size=50,
         nn=False,
         ffield='ffield.json',
         i=0,j=1):
    i = int(i)
    j = int(j)
    images = Trajectory(traj)
    ir = IRFF_NP(atoms=images[0],
                libfile=ffield,
                rcut=None,
                nn=True)

    r,f,bo,bosi,bopsi,sieng,powb,expb = [],[],[],[],[],[],[],[]
    bopi,bopp,pieng,ppeng = [],[],[],[]

    for atoms in images:
        ir.calculate_Delta(atoms)
        r.append(ir.r[i][j])
        # f.append(ir.F[i][j])

        bo.append(ir.bo0[i][j])
        bosi.append(ir.bosi[i][j])
        bopi.append(ir.bopi[i][j])
        bopp.append(ir.bopp[i][j])

        bopsi.append(ir.bop_si[i][j])

        sieng.append(-ir.sieng[i][j])
        powb.append(ir.powb[i][j])
        expb.append(ir.expb[i][j])
        pieng.append(-ir.pieng[i][j])
        ppeng.append(-ir.ppeng[i][j])


    plt.figure()    
    plt.subplot(3,2,1)        
    plt.plot(r,alpha=0.5,color='b',
             linestyle='-',label="radius@%d-%d" %(i,j))
    plt.legend(loc='best',edgecolor='yellowgreen')

    # plt.subplot(3,2,2)
    # plt.plot(f,alpha=0.5,color='r',
    #                linestyle='-',label="F@%d-%d" %(i,j))
    # plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,2)     
    plt.plot(bo,alpha=0.5,color='b',
             linestyle='-',label="BO@%d-%d" %(i,j))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,3)
    plt.plot(pieng,alpha=0.5,color='b',
             linestyle='-',label=r"$ebond_{pi}$@%d-%d" %(i,j))
    plt.plot(ppeng,alpha=0.5,color='k',
             linestyle='-',label=r"$ebond_{pp}$@%d-%d" %(i,j))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,4)     
    plt.plot(bopsi,alpha=0.5,color='b',
             linestyle=':',label=r"$BO^{'}_{si}$@%d-%d" %(i,j))
    plt.plot(bosi,alpha=0.5,color='r',
             linestyle='-',label=r'$BO_{si}$@%d-%d' %(i,j))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,5)
    plt.plot(sieng,alpha=0.5,color='r',
             linestyle='-',label=r"$ebond_{si}$@%d-%d" %(i,j))
    # plt.plot(pieng,alpha=0.5,color='b',
    #          linestyle='-',label=r"$ebond_{pi}$@%d-%d" %(i,j))
    # plt.plot(ppeng,alpha=0.5,color='k',
    #          linestyle='-',label=r"$ebond_{pp}$@%d-%d" %(i,j))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,6)
    plt.plot(bopi,alpha=0.5,color='b',
             linestyle='-',label=r"$bo_{pi}$@%d-%d" %(i,j))
    plt.plot(bopp,alpha=0.5,color='r',
             linestyle='-',label=r"$bo_{pp}$@%d-%d" %(i,j))
    plt.legend(loc='best',edgecolor='yellowgreen')

    # plt.subplot(3,2,6)
    # plt.plot(powb,alpha=0.5,color='b',
    #          linestyle='-',label=r"$pow_{si}$@%d-%d" %(i,j))
    # plt.plot(expb,alpha=0.5,color='r',
    #          linestyle='-',label=r"$exp_{si}$@%d-%d" %(i,j))
    # plt.legend(loc='best',edgecolor='yellowgreen')

    # plt.subplot(3,2,6)
    # # plt.ylabel(r'$exp$ (eV)')
    # plt.xlabel(r"Step")
    # plt.plot(eterm1[i][j],alpha=0.5,color='b',
    #          linestyle='-',label=r"$exp_{si}$@%d-%d" %(i,j))
    # plt.legend(loc='best',edgecolor='yellowgreen')
    traj_= traj.split('.')[0]
    plt.savefig('%s_bondorder_%d-%d.eps' %(traj_,i,j),transparent=True)  
    plt.close() 
    
    
if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser,[plbd])
   argh.dispatch(parser)

