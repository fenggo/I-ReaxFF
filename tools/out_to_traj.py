#!/usr/bin/env python
import sys
# import argh
import argparse
from os import getcwd,listdir
from irff.dft.siesta import parse_out
from ase.io.trajectory import TrajectoryWriter

def siesta_out_to_traj(traj):
    traj =  TrajectoryWriter(traj,mode='w')
    current_dir = getcwd()
    outs = listdir(current_dir)
    
    for o_ in outs:
        if o_.endswith('.out'):
           atoms = parse_out(outfile=o_)
           traj.write(atoms=atoms)

    traj.close()
     

help_ = 'run with commond: ./out_to_traj.py --t=md.traj'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--t',default='md.traj',type=str, help='Trajectory file name')
args    = parser.parse_args(sys.argv[1:])

if __name__=='__main__':
   siesta_out_to_traj(args.t)

