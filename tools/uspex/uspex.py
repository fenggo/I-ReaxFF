#!/usr/bin/env python
import subprocess

''' A script that run USPEX.m
'''

while True:
    subprocess.call('./run-octave USPEX.m > log',shell=True)
    output = subprocess.check_output('cat log | tail -2',shell=True)
    if output.find('-- stopping myself...'):
       subprocess.call('rm still_running',shell=True)
    else:
       raise SystemExit(1)
 
