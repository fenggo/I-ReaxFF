#!/usr/bin/env python
import subprocess

''' A script that run USPEX.m
    run this scrip with:
      nohup ./uspex.py 2>&1 & 
'''

while True:
    subprocess.call('./run-octave USPEX.m > log',shell=True)
    output = subprocess.check_output('cat log | tail -2',shell=True).decode('utf-8')
    
    if output.find('Relaxation is done')<0 and output.find('USPEX IS DONE')<0::
       subprocess.call('rm still_running',shell=True)
    else:
       raise SystemExit(1)
