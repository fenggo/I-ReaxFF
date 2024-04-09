#!/usr/bin/env python3
import argh
import argparse
from irff.dft.nwchem import write_nwchem_inp


def w(gen='POSCAR'):
    write_nwchem_inp(gen=gen)
   

if __name__ == '__main__':
   ''' run this script as :
        ./md.py --s=100 --g=md.traj 
        s 模拟步长
        g 初始结构'''
   # moleculardynamics()
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [w])
   argh.dispatch(parser)

