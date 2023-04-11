#!/usr/bin/env python
from os import system, getcwd, chdir,listdir
# from train_reaxff_perfect20181028 import test_reax
from irff.reax import test_reax

# system('rm *.pkl')
system('./r2l<ffield>reax.lib')
dirc={'hb6':'/home/gfeng/siesta/train/hb6'}
test_reax(direcs=dirc,dft='siesta',batch_size=50)

