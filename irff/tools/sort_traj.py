#!/usr/bin/env python
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter,Trajectory
from os import getcwd,listdir # system, chdir,
from os.path import isfile,exists,isdir


def sortraj(traj='siesta.traj'):
    newt= traj[:-5] + '_.traj'
    images = Trajectory(traj)

    his = TrajectoryWriter(newt,mode='w')

    images = Trajectory(traj)
    for i in range(50):
        atoms = images[i]
        his.write(atoms=atoms)
    his.close()


def coltraj(traj='siesta.traj'):
    newt= traj[:-5] + '_.traj'
    # images = Trajectory(traj)

    his = TrajectoryWriter(newt,mode='w')

    cdir   = getcwd()
    trajs  = listdir(cdir)

    for traj in trajs:
        if traj.find('.traj')>0 and traj != 'siesta_.traj':
           print('- reading file %s ...' %traj)
           images = Trajectory(traj)
           for i in range(50):
               atoms = images[i]
               his.write(atoms=atoms)
               
    his.close()



if __name__ == '__main__':
   coltraj(traj='siesta.traj')

