#!/usr/bin/env python
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter,Trajectory
from os import getcwd,listdir # system, chdir,
from os.path import isfile,exists,isdir
import argh
import argparse


def merge(t1='O7H20C2.traj',t2='O7H20C2opt.traj'):
    his = TrajectoryWriter('merged.traj',mode='w')
    trajs=[t1,t2]
    for traj in trajs:
        images = Trajectory(traj)
        for atoms in images:
            his.write(atoms=atoms)
    his.close()


def col(traj='siesta.traj',start=0,end=20):
    newt= traj.replace('.traj','_.traj')
    images = Trajectory(traj)

    his = TrajectoryWriter(newt,mode='w')

    images = Trajectory(traj)
    for i in range(start,end):
        atoms = images[i]
        his.write(atoms=atoms)
    his.close()


def collect(traj='siesta.traj',start=0,end=20):
    newt= traj[:-5] + '_.traj'
    # images = Trajectory(traj)

    his = TrajectoryWriter(newt,mode='w')

    cdir   = getcwd()
    trajs  = listdir(cdir)

    for traj in trajs:
        if traj.find('.traj')>0 and traj != 'siesta_.traj':
           print('- reading file %s ...' %traj)
           images = Trajectory(traj)
           for i in range(start,end):
               atoms = images[i]
               his.write(atoms=atoms)
               
    his.close()



if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [merge,collect,col])
   argh.dispatch(parser)

