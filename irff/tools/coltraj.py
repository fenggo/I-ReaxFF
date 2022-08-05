#!/usr/bin/env python
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter,Trajectory
from os import getcwd,listdir,system #, chdir,
from os.path import isfile,exists,isdir
import argh
import argparse
import sys



def collect(traj='md.traj',start=0,end=36):
    traj_  = traj.replace('.traj','_.traj')
    system('cp '+traj +' ' + traj_)
    images = Trajectory(traj_)
    his    = TrajectoryWriter(traj,mode='w')
    for i in range(start,end):
        atoms = images[i]
        his.write(atoms=atoms)
    his.close()



if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='collect the molecular dynamics trajectory')
   parser.add_argument('--start',default=0,type=int, help='the start frame')
   parser.add_argument('--end',default=1,type=int, help='the end frame')
   parser.add_argument('--traj',default='md.traj',type=str, help='trajectory file name')
   args = parser.parse_args(sys.argv[1:])

   collect(traj=args.traj,start=args.start,end=args.end)



