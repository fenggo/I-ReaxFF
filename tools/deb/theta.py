#!/usr/bin/env python
import numpy as np
import sys
import argparse
from ase.io.trajectory import Trajectory
from irff.irff_np import IRFF_NP
import matplotlib.pyplot as plt


def theta(images,trace):
    atoms = images[0]
    ir = IRFF_NP(atoms=atoms,
                libfile='ffield.json',
                nn=True)
    ir.calculate(atoms)

    for a,ang in enumerate(ir.angs):
        i_,j_,k_ = ang
        if ir.eang[a]>1000.00000001:
           print('{:3d} {:2d}-{:2d}-{:2d}  {:s}-{:s}-{:s}  {:8.4f}  {:8.4f} '.format(a,i_,j_,k_,
                 ir.atom_name[i_],ir.atom_name[j_],ir.atom_name[k_],ir.eang[a],ir.theta[a]))
        else: # if ir.eang[a]<0.0:
           print('{:3d} {:2d}-{:2d}-{:2d}  {:s}-{:s}-{:s} E: {:8.4f} {:8.4f} f7 {:8.4f} f8 {:8.4f} thet {:8.4f}'.format(a,i_,j_,k_,
                 ir.atom_name[i_],ir.atom_name[j_],ir.atom_name[k_],ir.eang[a],
                 ir.fijk[a],ir.f_7[a],ir.f_8[a],ir.theta[a]))
            
    if trace:
        a_ = int(input('please input the id of the  angle to output(-1 to exit): '))
        i,j,k = ir.angs[a_]
        ang_ =  '{:s}-{:s}-{:s}'.format(ir.atom_name[i],ir.atom_name[j],ir.atom_name[k])

        theta_ = []
        Eang   = []

        if a_>=0:
            for atoms in images:
                ir.calculate(atoms)
                a = np.where(np.logical_and(np.logical_and(ir.angs[:,0]==i,ir.angs[:,1]==j),ir.angs[:,2]==k))
                a = np.squeeze(a)

                if len(ir.angs)==1:
                   print('{:3d}  theta0: {:6.4f} theta {:6.4f} Dang: {:6.4f} rnlp: {:6.4f} '
                            'SBO: {:6.4f} sbo: {:6.4f} pbo: {:6.4f} SBO3: {:6.4f} Expang: {:6.4f} Eang: {:6.4f}'.format(a,
                            ir.thet0[a],ir.theta,ir.dang,ir.rnlp,ir.SBO,ir.sbo,ir.pbo,
                            ir.SBO3,ir.expang[a],ir.eang[a])) # self.thet0-self.theta
                   theta_.append(ir.theta)
                   Eang.append(ir.eang[a])
                else:
                   print('{:3d}  theta0: {:6.4f} theta {:6.4f} Dang: {:6.4f} rnlp: {:6.4f} '
                            'SBO: {:6.4f} sbo: {:6.4f} pbo: {:6.4f} SBO3: {:6.4f} Expang: {:6.4f} Eang: {:6.4f}'.format(a,
                            ir.thet0[a],ir.theta[a],ir.dang[a],ir.rnlp[a],ir.SBO[a],ir.sbo[a],ir.pbo[a],
                            ir.SBO3[a],ir.expang[a],ir.eang[a])) # self.thet0-self.theta
                   theta_.append(ir.theta[a])
                   Eang.append(ir.eang[a])
                   else:
                      print(a)

        # plt.figure()     
        # # plt.plot(theta_,Eang,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$Eangle({:s})$'.format(ang_))
        # plt.scatter(theta_,Eang,marker='o',color='none',edgecolors='r',s=10,label=r'$Eangle({:s})$'.format(ang_))

        # plt.legend(loc='best',edgecolor='yellowgreen')
        # plt.show() # if show else plt.savefig('deb_bo.pdf')
        # plt.close()


help_ = 'run with commond: ./theta.py --t=md.traj '

parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--t',default='md.traj',type=str,help='the atomic gementry file name')
parser.add_argument('--trace',default=0,type=int,help='trace this angel')
args = parser.parse_args(sys.argv[1:])

images = Trajectory(args.t)
theta(images,args.track)

# 3.1415926*120/180.0   = 2.0943950666666664
# 3.1415926*109.8/180.0 = 1.9163714859999998
