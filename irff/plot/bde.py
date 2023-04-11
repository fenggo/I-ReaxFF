#!/usr/bin/env python
# coding: utf-8
from os import system
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance
from irff.irff import IRFF
from irff.irff_np import IRFF_NP
from irff.md.gulp import write_gulp_in,get_reax_energy
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


def get_gulp_energy(atoms,lib='reax'):
    write_gulp_in(atoms,lib=lib,runword='gradient nosymmetry conv qite verb')
    system('gulp<inp-gulp>out')
    e_,eb_,el_,eo_,eu_,ea_,ep_,etc_,et_,ef_,ev_,ehb_,ecl_,esl_= get_reax_energy(fo='out')
    return e_


def BDE(traj='c2h6-CC.traj',iatom=0,jatom=3):
    images = Trajectory(traj)
    e,ei,ei_,ebud,d_irff,eg,erdx,d_reax= [],[],[],[],[],[],[],[]

    ir_mpnn = IRFF(atoms=images[0],libfile='ffield.json',nn=True)
    # ir_mat  = IRFF(atoms=images[0],libfile='ffield.reax.mattsson',nn=False)
    ir_reax = IRFF(atoms=images[0],libfile='ffield',nn=False)

    r,v = [],[]
    for i,atoms in enumerate(images):
        if i%2 != 0:
           continue
        emp = ir_mpnn.get_potential_energy(atoms)
        r_  = ir_mpnn.r[iatom][jatom].detach().numpy()

        # if r_>1.7:
        #    continue

        e.append(atoms.get_potential_energy())
        ei.append(emp)
        ei_.append(ir_reax.get_potential_energy(atoms))
        eg.append(get_gulp_energy(atoms,lib='reax_mat'))
        ebud.append(get_gulp_energy(atoms,lib='reax_bud'))
        # erdx.append(get_gulp_energy(atoms,lib='reax_rdx'))
        
        d_irff.append(abs(e[-1]-ei[-1]))
        d_reax.append(abs(e[-1]-ei_[-1]))

        v.append(atoms.get_volume())
        r.append(r_)
        print(' * energy: ',e[-1],ei[-1],ei_[-1],d_irff[-1],' radius: ',r[-1])
       
        # stress = atoms.get_stress()
        # print(stress)

    print(' * mean difference: ',np.mean(d_irff),np.mean(d_reax))
    e_min = min(e)
    e_max = max(e)
    e = np.array(e) - e_min

    ei   = np.array(ei)   - min(ei)
    ei_  = np.array(ei_)  - min(ei_)
    ebud = np.array(ebud) - min(ebud)
    eg   = np.array(eg)   - min(eg)
    # erdx = np.array(erdx) - min(erdx)
    plt.figure()   
    plt.ylabel(r'$Energy$ ($eV$)')
    plt.xlabel(r'$Radius$ ($\AA$)')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)


    plt.plot(r,ei,alpha=0.9,
             linestyle='-',marker='o',markerfacecolor='none',markeredgecolor='k',
             markersize=5,
             color='k',label='IRFF(MPNN)')

    plt.plot(r,e,alpha=0.9,
             linestyle='-',marker='s',markerfacecolor='none',markeredgecolor='r',
             markersize=5,
             color='r',label='DFT(SIESTA)')

    plt.plot(r,ei_,alpha=0.9,
             linestyle='-',marker='^',markerfacecolor='none',markeredgecolor='b',
             markersize=5,
             color='b',label='ReaxFF(trained)')

    plt.plot(r,ebud,alpha=0.9,
             linestyle='-',marker='+',markerfacecolor='none',markeredgecolor='g',
             markersize=5,
             color='g',label='ReaxFF(Thompson)')

    plt.plot(r,eg,alpha=0.9,
             linestyle='-',marker='v',markerfacecolor='none',markeredgecolor='mediumslateblue',
             markersize=5,
             color='mediumslateblue',label='ReaxFF(Mattsson)')

    # plt.plot(r,eg,alpha=0.9,
    #          linestyle='-',marker='d',markerfacecolor='none',markeredgecolor='steelblue',
    #          markersize=5,
    #          color='steelblue',label='ReaxFF(RDX)')
    err = np.abs(ei - e)
    err_= np.abs(ei_ - e)

    plt.fill_between(r, e - err, e + err, color='darkorange',
                     alpha=0.2)
    
    plt.fill_between(r, e - err_, e + err_, color='palegreen', # palegreen
                     alpha=0.2)

    plt.text( 0.0, e_max, '%.3f   ' %e_min, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
    plt.savefig('Energy.svg',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   BDE()

