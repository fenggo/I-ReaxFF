from __future__ import print_function
from os.path import isfile
from ase import Atoms
from ase.io import read,write
import os


def structure(gen):
    _root = os.path.abspath(os.path.dirname(__file__))
    if isfile(_root+'/'+gen+'.gen'):
       A = read(_root+'/'+gen+'.gen')
    elif isfile(_root+'/'+gen+'.cif'):
       A = read(_root+'/'+gen+'.cif')
    else:
       raise RuntimeError('Structure is not found!')
    return A


