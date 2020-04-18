from __future__ import print_function
from ase import Atoms
from ase.io import read,write
import os


def structure(gen):
    _root = os.path.abspath(os.path.dirname(__file__))
    A = read(_root+'/'+gen+'.gen')
    return A


