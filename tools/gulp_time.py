#!/usr/bin/env python
from irff.md.gulp import opt,nvt
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory


# atoms = read('md.traj',index=-1)
# his   = TrajectoryWriter('opt.traj', mode='w')
fd = open('time_reaxnn.txt','w')

for i in [8,9,10,12]: # 2,3,4,5,6,7
    atoms    = read('gpp.gen',index=-1)*(i,i,1)
    natom    = len(atoms)
    nvt(atoms=atoms,T=300,time_step=0.1,tot_step=5000,
        keyword='md qiterative conv',movieFrequency=10,
        ncpu=8,lib='reaxff_nn')

    with open('gulp.out','r') as f:
         for line in f.readlines():
             if line.find('Total CPU time')>=0:
                t = float(line.split()[3])
    print('natom: {:d} use time: {:f} '.format(natom,t))
    fd.write('{:d} {:f} \n'.format(natom,t))

fd.close()

