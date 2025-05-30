#!/usr/bin/env python
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
from irff.irff_np import IRFF_NP
from irff.deb.compare_energies import deb_gulp_energy


def trajplot(traj='siesta.traj',nn=True,i=0,j=1):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    step,e,ei      = [],[],[]
    r              = []
    d              = []
    # v            = []
    id_            = []
    atoms = images[0]
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 nn=nn)

    ir.calculate(atoms)
    masses = np.sum(atoms.get_masses())
	# ###### compare energies with GULP
    (e_,ebond_,eunder_,eover_,elone_,eang_,etcon_,epen_,
        etor_,efcon_,evdw_,ehb_,ecoul_) = deb_gulp_energy(images, ffield='reaxff_nn')
    for i_,atoms in enumerate(images):
        step.append(i_)
        e.append(atoms.get_potential_energy())
        ir.calculate(atoms)
        # ei.append(ir.E)
        ei.append(e_[i_])
        r.append(ir.r[i][j])
        id_.append(i_)
        
        volume = atoms.get_volume()
        density = masses/volume/0.602214129
        d.append(density)
    
    e_max = np.mean(e)
    e = np.array(e) - e_max
    e_max = np.mean(ei)
    ei   = np.array(ei) - e_max

    plt.figure()   
    plt.ylabel(r'$Energy$ ($eV$)',fontdict={"fontsize":16})
    # plt.xlabel(r'$Crystal$ $ID$')
    # plt.xlabel(r'$Distance$ ($\AA$)',fontdict={"fontsize":16})
    if args.x==0:
       lab = '$Density$ ($g/cm^3$)'
    elif args.x==1:
       lab = '$Crystal$ $ID$'
    elif args.x==2:
       lab = '$Distance$ ($\AA$)'
    plt.xlabel(r'{:s}'.format(lab),fontdict={"fontsize":16})
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_position(('data',0))
    # ax.spines['bottom'].set_position(('data', 0))
    if args.x==0:
       x = d
    elif args.x==1:
       x = id_
    else:
       x = r

    plt.plot(x,e,alpha=0.8,   # d: density;  id_: id
             linestyle='-',linewidth=1.5,marker='s',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='r',markersize=10,
             color='red',label=r'$DFT$ ($SIESTA$)')

    plt.plot(x,ei,alpha=0.8,
             linestyle='-',linewidth=1.5,marker='o',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='k',markersize=10,
             color='k',label=r'$ReaxFF-nn$')

    ediff = np.abs(e - ei)
    plt.fill_between(x,ei - ediff, ei + ediff, color='#4E8872',
                     alpha=0.2)

    # pdiff = np.abs(pdft - preax)
    # plt.fill_between(v_, pdft - pdiff, pdft + pdiff, color='palegreen',
    #                  alpha=0.2)
    # plt.text( 0.0, 0.5, '%.3f' %e_max, fontdict={'size':13.0, 'color': 'k'})
    plt.legend(loc='best',edgecolor='yellowgreen',fontsize=16) # loc = lower left upper right best
    plt.savefig('{:s}'.format(traj.replace('traj','svg')),transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''

   parser = argparse.ArgumentParser(description='stretch molecules')
   parser.add_argument('--t', default='gulp.traj',type=str, help='trajectory file')
   parser.add_argument('--x', default=0,type=int, help='x axis: 0 density, 1 id, 2 radius')
   parser.add_argument('--i', default=0,type=int, help='atom i')
   parser.add_argument('--j', default=1,type=int, help='atom j')
   args = parser.parse_args(sys.argv[1:])
   trajplot(traj=args.t,i=args.i,j=args.j)

