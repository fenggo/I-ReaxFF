#!/usr/bin/env python
import sys
# import argh
import argparse
# from os import getcwd,listdir
from irff.dft.siesta import siesta_traj
from ase.io.trajectory import TrajectoryWriter

def md_out_to_traj(traj):
    ''' siesta MD out to traj
    '''
    traj =  TrajectoryWriter(traj,mode='w')
    # current_dir = getcwd()
    images = siesta_traj(label='Carbonate',fdf='INPUT.fdf',out='log.out')
    for atoms in images:
        traj.write(atoms=atoms)
    traj.close()
     



if __name__=='__main__':
   md_out_to_traj('md.traj')