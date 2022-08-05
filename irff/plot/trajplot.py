#!/usr/bin/env python
from __future__ import print_function
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
# from .mdtodata import MDtoData
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.reax_eager import ReaxFF
from irff.irff import IRFF


def e(traj='siesta.traj',batch_size=100,nn=False):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    e,ei      = [],[]

    ir = IRFF(atoms=images[0],
          libfile='ffield',
          nn=False,
          bo_layer=[8,4])

    ir.get_potential_energy(images[0])

    for i,atoms in enumerate(images):
        e.append(atoms.get_potential_energy())
        ei.append(ir.get_potential_energy(atoms))

    e_max = max(e)
    e = np.array(e) - e_max

    rn = ReaxFF(libfile=ffield,
                direcs={'tmp':'siesta.traj'},
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

    mol  = 'tmp'
    er   = rn.get_value(rn.E[mol])
    dfte = rn.get_value(rn.dft_energy[mol]) 
    zpe  = rn.get_value(rn.zpe[mol])

    er   = er - e_max
    ei   = np.array(ei) - e_max

    plt.figure()   
    plt.ylabel('Energy (eV)')
    plt.xlabel('Step')
    plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_position(('data',0))
    # ax.spines['bottom'].set_position(('data', 0))

    plt.plot(e,alpha=0.01,
             linestyle='-',marker='s',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='r',markersize=4,
             color='red',label='DFT (SIESTA)')
    plt.plot(er,alpha=0.01,
             linestyle='-',marker='o',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='b',markersize=4,
             color='blue',label='ReaxFF')
    plt.plot(ei,alpha=0.01,
             linestyle='-',marker='^',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='g',markersize=4,
             color='blue',label='IRFF')

    plt.text( 0.0, 0.5, '%.3f' %e_max, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='upper left',edgecolor='yellowgreen') # lower left upper right
    plt.savefig('Energy.eps',transparent=True) 
    plt.close() 


def pbc(vr,cell):
    cell   = np.array(cell)
    hfcell = 0.5*cell
    lm     = np.where(vr-hfcell>0)
    lp     = np.where(vr+hfcell<0)
    # print(lm[0],lp[0],len(lm[0]),len(lp[0]))
    while(len(lm[0])!=0 or len(lp[0])!=0):
        vr   = np.where(vr-hfcell>0,vr-cell,vr)
        vr   = np.where(vr+hfcell<0,vr+cell,vr)     # apply pbc
        lm   = np.where(vr-hfcell>0)
        lp   = np.where(vr+hfcell<0)
    return vr


def a(atom_id=[23,22,24],traj='siesta.traj'):
    images = Trajectory(traj)
    e      = []
    ang    = []
    for i,atoms in enumerate(images):
        e.append(atoms.get_potential_energy())
        cell   = atoms.get_cell()
        box    = [cell[0][0],cell[1][1],cell[2][2]]
         
        # atoms.positions[atom_id[2]][0] += cell[0][0]
        v1 = atoms.positions[atom_id[0]]-atoms.positions[atom_id[1]]
        v2 = atoms.positions[atom_id[2]]-atoms.positions[atom_id[1]]
        v3 = atoms.positions[atom_id[2]]-atoms.positions[atom_id[0]]
        v1 = pbc(v1,box)
        v2 = pbc(v2,box)
        v3 = pbc(v3,box)

        r1_2= np.sum(np.square(v1))
        r1  = np.sqrt(r1_2)
        r2_2= np.sum(np.square(v2))
        r2  = np.sqrt(r2_2)
        r3_2= np.sum(np.square(v3))

        cos_ = 0.5*(r1_2 + r2_2 - r3_2)/(r1*r2)
        ang.append(180.0*np.arccos(cos_)/3.1415926)

    e_max = max(e)
    e = np.array(e) - e_max

    plt.figure()   
    plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    ax = plt.gca()
    #ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('top')

    ax.xaxis.set_label('Step')
    ax.yaxis.set_label('Angle (degree)')

    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    # ax.set_title('Step versus Angle (degree)', color='0.7')
    
    ax.set_facecolor('none') 

    plt.plot(ang,alpha=0.01,color='red')
    # plt.legend(loc='best')
    plt.savefig('Angle.eps',transparent=True) 
    plt.close() 


def rt(atom_id=[1,18,35],traj='siesta.traj'):
    images = Trajectory(traj)
    e      = []
    r1,r2  = [],[]
    for i,atoms in enumerate(images):
        e.append(atoms.get_potential_energy())
        cell   = atoms.get_cell()
        box    = [cell[0][0],cell[1][1],cell[2][2]]
         
        v1 = atoms.positions[atom_id[0]]-atoms.positions[atom_id[1]]
        v2 = atoms.positions[atom_id[2]]-atoms.positions[atom_id[1]]
        v1 = pbc(v1,box)
        v2 = pbc(v2,box)

        r1_= np.sqrt(np.sum(np.square(v1)))
        r1.append(r1_)
        r2_= np.sqrt(np.sum(np.square(v2)))
        r2.append(r2_)

    plt.figure()   
    plt.xlim(0,i)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('top')

    ax.xaxis.set_label('Step')
    ax.yaxis.set_label(r'Distance ($\AA$)')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

    plt.plot(r1,alpha=0.01,color='red')
    plt.plot(r2,alpha=0.01,color='blue')
    plt.savefig('Radius.eps',transparent=True) 
    plt.close() 


def rdb(atom_id1=[0,1],atom_id2=[0,9],traj='siesta.traj'):
    images = Trajectory(traj)
    e      = []
    r1,r2  = [],[]
    for i,atoms in enumerate(images):
        e.append(atoms.get_potential_energy())
        cell   = atoms.get_cell()
        box    = [cell[0][0],cell[1][1],cell[2][2]]
         
        v1 = atoms.positions[atom_id1[0]]-atoms.positions[atom_id1[1]]
        v2 = atoms.positions[atom_id2[0]]-atoms.positions[atom_id2[1]]
        v1 = pbc(v1,box)
        v2 = pbc(v2,box)

        r1_= np.sqrt(np.sum(np.square(v1)))
        r1.append(r1_)
        r2_= np.sqrt(np.sum(np.square(v2)))
        r2.append(r2_)

    plt.figure()   
    plt.xlim(0,i)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('top')

    ax.xaxis.set_label('Step')
    ax.yaxis.set_label(r'Distance ($\AA$)')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

    plt.plot(r1,alpha=0.01,color='red')
    plt.plot(r2,alpha=0.01,color='blue')
    plt.savefig('Radius.eps',transparent=True) 
    plt.close() 


def r(atom_id=[0,4],traj='siesta.traj'):
    images = Trajectory(traj)
    e      = []
    r      = []
    for i,atoms in enumerate(images):
        e.append(atoms.get_potential_energy())
        cell = atoms.get_cell()

        v1 = atoms.positions[atom_id[0]]-atoms.positions[atom_id[1]]
        r1_2= np.sum(np.square(v1))
        r1  = np.sqrt(r1_2)
        r.append(r1)

    e_max = max(e)
    e = np.array(e) - e_max

    plt.figure()   
    plt.xlim(0,i)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('top')

    ax.xaxis.set_label('Step')
    ax.yaxis.set_label(r'Distance ($\AA$)')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

    plt.plot(r,alpha=0.01,color='red')
    plt.savefig('Radius.eps',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./cp.py scale-md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [e,a,r,rt,rdb])
   argh.dispatch(parser)

   