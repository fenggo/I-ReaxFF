#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance,get_group
# get_ipython().run_line_magic('matplotlib', 'inline')


# atoms = read('aimd_nm/nm-56/nm.traj',index=0)
atoms  = read('nmhb.gen',index=-1)
# atoms   = read('c2h6.gen',index=-1)

i,j     = (4,2)
#i,j    = (0,3)

ad      = AtomDance(atoms=atoms)
#images = ad.bond_momenta_bigest(atoms)
images  = ad.stretch([i,j],nbin=50,rst=1.7,red=3.0,scale=1.26,traj='md.traj')
#images = ad.swing([1,0,4],st=90.0,ed=110.0,nbin=30,traj='md.traj')
ad.close()
# view(images)



