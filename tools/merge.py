#!/usr/bin/env python
import argh
import argparse
import sys
from os import getcwd,listdir # system, chdir,
from os.path import isfile,exists,isdir
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter,Trajectory


def merge(t1='O7H20C2.traj',t2='O7H20C2opt.traj'):
    his = TrajectoryWriter('merged.traj',mode='w')
    trajs=[t1,t2]
    for traj in trajs:
        images = Trajectory(traj)
        for atoms in images:
            his.write(atoms=atoms)
    his.close()


if __name__ == '__main__':
   '''
       Usage: ./merge.py --t1=md.traj --t2=out.traj
   '''
   parser = argparse.ArgumentParser(description='merg the molecular dynamics trajectory')
   parser.add_argument('--t1',default='md1.traj',type=str, help='trajectory 1 file name')
   parser.add_argument('--t2',default='md2.traj',type=str, help='trajectory 2 file name')
   args = parser.parse_args(sys.argv[1:])
   merge(t1=args.t1,t2=args.t2)

