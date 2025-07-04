#!/usr/bin/env python
import sys
# import argh
import argparse
# from os import getcwd,listdir
from irff.dft.siesta import siesta_traj
from ase.io.trajectory import TrajectoryWriter

def md_out_to_traj(traj='md.traj',fdf='in.fdf',label='siesta',out='siesta.out'):
    ''' siesta MD out to traj
    '''
    traj =  TrajectoryWriter(traj,mode='w')
    # current_dir = getcwd()
    images = siesta_traj(label=label,fdf=fdf,out=out)
    for atoms in images:
        traj.write(atoms=atoms)
    traj.close()
     

if __name__=='__main__':
   md_out_to_traj(traj='md.traj',fdf='in.fdf',label='siesta',out='siesta.out')