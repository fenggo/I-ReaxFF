#!/usr/bin/env python
from __future__ import print_function
from os import system,listdir,getcwd



eps = listdir(getcwd())

for fil in eps:
    if fil.find('.eps')>=0:
       system('epstopdf '+fil)



