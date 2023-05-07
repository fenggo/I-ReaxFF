#!/usr/bin/env python
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter,Trajectory
from os import getcwd,listdir,system #, chdir,
from os.path import isfile,exists,isdir
import argh
import argparse
import sys



def collect(traj='md.traj',start=0,end=36,o=None,interval=1):
    if o is None:
       newt= traj.replace('.traj','_.traj')
    else:
       newt = o
    # system('cp '+traj +' ' + traj_)
    images = Trajectory(traj)
    his    = TrajectoryWriter(newt,mode='w')
    for i in range(start,end):
        if i%interval==0:
           his.write(atoms=images[i])
    his.close()



if __name__ == '__main__':
   '''
       select frames in all trajectoies
   '''
   help_  = 'Run with commond: ./coltraj.py --t=md.traj --s=0 --e=50 --o=out.traj'
   parser = argparse.ArgumentParser(description=help_)
   parser.add_argument('--start',default=0,type=int, help='the start frame')
   parser.add_argument('--end',default=1,type=int, help='the end frame')
   parser.add_argument('--i',default=1,type=int, help='collect interval')
   parser.add_argument('--traj',default='md.traj',type=str, help='trajectory file name')
   parser.add_argument('--o',default='output.traj',type=str, help='output trajectory file name')
   args = parser.parse_args(sys.argv[1:])

   collect(traj=args.traj,start=args.start,end=args.end,o=args.o,interval=args.i)



