#!/usr/bin/env python
from __future__ import print_function
from os import system,getcwd,chdir 
from os.path import isfile,exists
#
#  Pseudopotential generation
#  pg: simple generation
#
# rc_dic = {'C':1.50,'H':1.20,'N':1.40,'O':1.40,'Fe':2.00}

rc_dic = {'C':1.56,'H':1.33,'N':1.48,'O':1.47,'Fe':1.95,'F':1.41,'Al':1.86}
# rc_dic = {'Fe':2.00}
pa = 'vwr'# pseudo authour r=polarzed

for key in rc_dic:
    if exists(key+'.'+pa):
       system('rm -rf '+key+'.'+pa)
    if exists(key+'.'+pa+'.inp'):
       system('rm '+key+'.'+pa+'.inp') 
    fp = open(key+'.'+pa+'.inp','w')
    print('   pg %s TM2 Pseudopotencial GS ref' %key, file=fp)
    if key =='H':
       print('        tm2     2.00', file=fp)
    elif key =='Fe':
       print('        tm2     3.00', file=fp)  # Flavor and radius 
    else:
       print('        tm2     2.00', file=fp)
    el = len(key)
    if el==1:
       print(' n=%s  c=%s ' %(key,pa), file=fp)
    else:
       print(' n=%s c=%s ' %(key,pa), file=fp) 
    print('         0 ', file=fp)
    rc = rc_dic[key]
    if key=='H':
       print('    0    4', file=fp)
       print('    1    0      1.00      0.00', file=fp)
       print('    2    1      0.00      0.00', file=fp)
       print('    3    2      0.00      0.00', file=fp)
       print('    4    3      0.00      0.00', file=fp)
    elif key == 'O':
       print('    1    4', file=fp)
       print('    2    0      2.00      0.00', file=fp)
       print('    2    1      4.00      0.00', file=fp)
       print('    3    2      0.00      0.00', file=fp)
       print('    4    3      0.00      0.00', file=fp)
    elif key == 'F':
       print('    1    4', file=fp)
       print('    2    0      2.00      0.00', file=fp)
       print('    2    1      5.00      0.00', file=fp)
       print('    3    2      0.00      0.00', file=fp)
       print('    4    3      0.00      0.00', file=fp)
    elif key == 'N':
       print('    1    4', file=fp)
       print('    2    0      2.00      0.00', file=fp)
       print('    2    1      3.00      0.00', file=fp)
       print('    3    2      0.00      0.00', file=fp)
       print('    4    3      0.00      0.00', file=fp)
    elif key == 'C':
       print('    1    4', file=fp)
       print('    2    0      2.00      0.00', file=fp)
       print('    2    1      2.00      0.00', file=fp)
       print('    3    2      0.00      0.00', file=fp)
       print('    4    3      0.00      0.00', file=fp)
    elif key == 'Al':
       print('    3    4', file=fp)
       print('    3    0      2.00      0.00    # 3s2', file=fp)
       print('    3    1      1.00      0.00    # 3p1', file=fp)
       print('    3    2      0.00      0.00    # 3d0', file=fp)
       print('    4    3      0.00      0.00    # 4f0', file=fp)
    elif key == 'Fe':
       print('    5    4', file=fp)
       print('    4    0      2.00      0.00    # 4s2', file=fp)
       print('    4    1      0.00      0.00    # 4p0', file=fp)
       print('    3    2      6.00      0.00    # 3d6', file=fp)
       print('    4    3      0.00      0.00    # 4f0', file=fp)
    if key == 'Fe':
       print('      %4.2f      %4.2f      %4.2f      %4.2f      0.01     -1.00' %(rc,rc,rc+0.05,rc+0.06), file=fp)
    elif key == 'H':
       print('      %4.2f      %4.2f      %4.2f      %4.2f      0.00      0.00' %(rc,rc,0.37,rc), file=fp)
    if key == 'Al':
       print('      %4.2f      %4.2f      %4.2f      %4.2f      0.01     -1.00' %(rc,rc+0.2,rc+0.36,rc+0.36), file=fp)    
    else:
       print('      %4.2f      %4.2f      %4.2f      %4.2f      0.01     -1.00' %(rc,rc,rc,rc), file=fp)

    print(' ', file=fp)
    print('#2345678901234567890123456789012345678901234567890 Ruler', file=fp)
    fp.close()
    print('* generating pseudo for %s, by Author %s ...' %(key,pa))
    system('./pg.sh '+key+'.'+pa+'.inp' )

for e in rc_dic:
    print('cp '+e+'.'+pa+'.psf ../work/'+e+'.psf')
    system('cp '+e+'.'+pa+'.psf ../work/'+e+'.psf')
