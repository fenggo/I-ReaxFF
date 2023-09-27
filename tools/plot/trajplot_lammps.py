#!/usr/bin/env python
import sys
import argparse
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff_np import IRFF_NP
from ase.calculators.lammpslib import LAMMPSlib

LAMMPSlib.default_parameters['lammps_header'] =  ['units real',
                                                  'atom_style charge',
                                                  'atom_modify map array sort 0 0']

cmd1 = ["pair_style     reaxff control nn yes checkqeq yes",
        "pair_coeff     * * ffield C H N O",
        "fix   REX all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff"]

cmd2 = ["pair_style     reaxff control.reax nn no checkqeq yes",
        "pair_coeff     * * ffield.reax.rdx C H N O",
        "fix   REX all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff"]

cmd3 = ["pair_style     reaxff control.lg lgvdw yes checkqeq yes",
        "pair_coeff     * * ffield.reax.lg C H N O",
        "fix   REX all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff"]

def trajplot(traj='siesta.traj',nn=True,i=0,j=1):
    ffield = 'ffield.json' if nn else 'ffield'
    images         = Trajectory(traj)
    step,e1,ei,e2  = [],[],[],[]
    e3             = []
    r              = []

    ir = IRFF_NP(atoms=images[0],
              libfile=ffield,
              nn=nn)

    ir.calculate(images[0])

    for i_,atoms in enumerate(images):
        step.append(i_)
        atoms1=atoms.copy()
        atoms2=atoms.copy()
        # e.append(atoms.get_potential_energy())

        ir.calculate(atoms)
        ei.append(ir.E)
        r.append(ir.r[i][j])

        lmp1 = LAMMPSlib(lmpcmds=cmd1, log_file='test1.log')
        lmp2 = LAMMPSlib(lmpcmds=cmd2, log_file='test2.log')
        lmp3 = LAMMPSlib(lmpcmds=cmd3, log_file='test3.log')
        atoms.calc = lmp1
        e1.append(atoms.get_potential_energy())
        atoms1.calc = lmp2
        e2.append(atoms1.get_potential_energy())
        atoms2.calc = lmp3
        e3.append(atoms2.get_potential_energy())
        print("Energy: ", e1[-1],e2[-1],e3[-1],ir.E)

    e_max = min(e1)
    e1 = np.array(e1) - e_max
    e_max = min(ei)
    ei   = np.array(ei) - e_max
    e_max = min(e2)
    e2   = np.array(e2) - e_max
    e_max = min(e3)
    e3   = np.array(e3) - e_max
    plt.figure()   
    plt.ylabel(r'$Energy$ ($eV$)')
    plt.xlabel(r'$Time$ $Step$ ($fs$)')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_position(('data',0))
    # ax.spines['bottom'].set_position(('data', 0))

    plt.plot(r,e2,alpha=0.8,
             linestyle='-',#marker='s',markerfacecolor='none',
             #markeredgewidth=1,markeredgecolor='r',markersize=3,
             color='red',label=r'$ReaxFF$-$RDX$')

    plt.plot(r,e1,alpha=0.8,
             linestyle='-',marker='o',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='b',markersize=10,
             color='k',label=r'$ReaxFF-nn$')

    plt.plot(r,e3,alpha=0.8,
             linestyle='-',#marker='s',markerfacecolor='none',
             #markeredgewidth=1,markeredgecolor='r',markersize=3,
             color='blue',label=r'$ReaxFF$-$lg$')
    # ediff = np.abs(e - ei)
    # plt.fill_between(r,ei - ediff, ei + ediff, color='palegreen',
    #                  alpha=0.2)

    # pdiff = np.abs(pdft - preax)
    # plt.fill_between(v_, pdft - pdiff, pdft + pdiff, color='palegreen',
    #                  alpha=0.2)

    plt.text( 0.0, 0.5, '%.3f' %e_max, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
    plt.savefig('{:s}'.format(traj.replace('traj','svg')),transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''

   parser = argparse.ArgumentParser(description='stretch molecules')
   parser.add_argument('--t', default='gulp.traj',type=str, help='trajectory file')
   parser.add_argument('--i', default=0,type=int, help='atom i')
   parser.add_argument('--j', default=1,type=int, help='atom j')
   args = parser.parse_args(sys.argv[1:])
   trajplot(traj=args.t,i=args.i,j=args.j)


