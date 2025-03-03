#!/usr/bin/env python
import pymatgen.core as pmg
# Integrated symmetry analysis tools from spglib
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


structure = pmg.Structure.from_file("251401.cif")

finder = SpacegroupAnalyzer(structure)
sp = finder.get_space_group_symbol()
print(sp)

structure.to_file('POSCAR')

