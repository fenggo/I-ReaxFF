#!/usr/bin/env python

def control_to_poscar():
    with open('CONTROL','r') as f:
         lines = f.readlines()
    element    = []
    lattvec    = []
    positions  = []
    for line in lines:
        if line.find('elements=\"')>=0:
           l = line.split('\"')
           element.append(l[1])
        elif line.find('lattvec(:')>=0:
           l = line.split('=')[1].split(',')[0]
           lattvec.append(l)
        elif line.find('natoms=')>=0:
           l = line.split('=')[1].split(',')[0]
           natom = l
        elif line.find('positions(:')>=0:
           l = line.split('=')[1].split(',')[0]
           positions.append(l)

    with open('POSCAR','w') as f:
         for e in element:
             print(e,end=' ',file=f)
         print(' ',file=f)
         print(1.0,file=f)
         for v in lattvec:
             print(v,file=f)
         for e in element:
             print(e,end=' ',file=f)
         print(' ',file=f)
         print(natom,file=f)
         print('Direc',file=f)
         for p in positions:
             print(p,file=f)

if __name__=='__main__':
   control_to_poscar()
