#!/usr/bin/env python
import os
import sys
import argh
import argparse
import numpy as np
import json
from ase import units
from ase.spacegroup import crystal
from ase.build import bulk
from flare.gp import GaussianProcess
from flare.utils.parameter_helper import ParameterHelper
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase.units import Ry
from flare.ase.calculator import FLARE_Calculator
from ase import units
from ase.io import read
from ase.io.trajectory import TrajectoryWriter



def diamond(supercell=[1,1,1],write_=False):
    a = 3.52678
    super_cell = bulk('C', 'diamond', a=a, cubic=True)
    super_cell = super_cell*supercell# [2,2,2]
    if write_:
       super_cell.write('diamond.gen')
    return super_cell


if __name__ == '__main__':
   diamond()

