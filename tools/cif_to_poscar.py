#!/usr/bin/env python
import sys
import argparse
import pymatgen.core as pmg
# Integrated symmetry analysis tools from spglib
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='Individuals.traj',type=str, help='trajectory file')
args = parser.parse_args(sys.argv[1:])

structure = pmg.Structure.from_file(args.g)

finder = SpacegroupAnalyzer(structure)
sp = finder.get_space_group_symbol()
print(sp)

structure.to_file('POSCAR')

