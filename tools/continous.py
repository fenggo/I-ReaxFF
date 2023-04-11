#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')


atoms  = read('gulp.traj',index=-1)
ad     = AtomDance(atoms=atoms,rmax=1.26)
# print(ad.InitZmat)
#print(ad.zmat_id[i_],ad.zmat_index[i_])
# print(ad.neighbors)
# for i,jj in enumerate(ad.zmat_index):
#     for j,j_ in enumerate(jj):
#         if j_>=0:
#            iatom = ad.zmat_id[i]
#            jatom = jj[0]
#            katom = jj[1]
#            latom = jj[2]
#            log,v = ad.get_optimal_zv(atoms,(i,j))
#            # print(log)
#            #v = ad.InitZmat
#            if j==0:
#               print('bond {:3d} {:2s} - {:3d} {:2s}:'.format(iatom,
#                            ad.atom_name[iatom],jatom,ad.atom_name[jatom]),v)
#            elif j==1:    
#               print('angle {:3d} {:2s} - {:3d} {:2s} - {:3d} {:2s}:'.format(iatom,
#                            ad.atom_name[iatom],jatom,ad.atom_name[jatom],
#                            katom,ad.atom_name[katom]),v)    
#            elif j==2:    
#               print('torsion {:3d} {:2s} - {:3d} {:2s} - {:3d} {:2s} - {:3d} {:2s}:'.format(iatom,
#                            ad.atom_name[iatom],jatom,ad.atom_name[jatom],
#                            katom,ad.atom_name[katom],jatom,ad.atom_name[jatom]),v) 
for i,jj in enumerate(ad.zmat_index):
    #for j,j_ in enumerate(jj):
    #    if j_>=0:
    if jj[0]>=0:
       iatom = ad.zmat_id[i]
       jatom = jj[0]
       katom = jj[1]
       latom = jj[2]
       log,v = ad.get_optimal_zv(atoms,(i,0))
       # print(log)
       #v = ad.InitZmat
       print('bond {:3d} {:2s} - {:3d} {:2s}:'.format(iatom,
             ad.atom_name[iatom],jatom,ad.atom_name[jatom]),v)
ad.close()

#view(images)

