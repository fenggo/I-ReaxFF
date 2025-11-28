#!/usr/bin/env python
import subprocess
from os import getcwd,chdir,listdir
from os.path import exists

cwd = getcwd()

if cwd.endswith('Seeds'):
   wkd = cwd[:-5]
elif cwd.endswith('Seeds/'):
   wkd = cwd[:-6]
else:
   wkd = cwd
# print(wkd)

chdir(wkd)
dirs = listdir() 
res  = {}

for d in dirs:
    if d.startswith('results11-'):
       # print(d)
       res[d] = {}
       if not exists(f'{d}/density.log'):
          continue
       with open(f'{d}/density.log','r') as fd:
            for line in fd:
                if not line.startswith('#'):
                   l = line.split()
                   res[d][l[0]] = l[2]

       # print(res[d])
        
chdir(cwd)
with open('Seeds.dat','r') as fd:
     for line in fd:
         st = line.split()[0]
         st_= st[2:].split('_')[0]
         for r_ in res:
             r = res[r_]
             if st_ in r:
                print(st,r[st_])
# output = subprocess.check_output('grep EA POSCARS_1',shell=True)
