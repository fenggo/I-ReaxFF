#!/usr/bin/env python


with open('POSCARS','r') as f:
     lines = f.readlines()

with open('POSCARS','w') as f:
     card = False
     for i,line in enumerate(lines):
         if line.find('direct')>=0 or line.find('Direct')>=0:
            card = True
		 elif line.find('EA')>=0 and line.find('Sym.group')>=0:
		    card = True
         if card and line.find('direct')<0 and line.find('Direct')<0:
		    l = line.split()
            print(l[0],l[1],l[2],file=f)
         else:
            print(line.rstrip(),file=f)