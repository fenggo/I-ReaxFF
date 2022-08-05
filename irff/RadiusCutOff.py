from os.path import isfile
import json as js

def setRcut(bonds,rcut_,rcuta_,re_):
    ''' in princeple, the cutoff should 
        cut the second nearest neithbors
    '''
    # if isfile('ffield.json'):
    # with open('ffield.json','r') as fj:
    #      RC = js.load(fj)
    #      if 'rcut' in RC:
    #         rcut_ = RC['rcut']
    #      if 'rcuta' in RC:
    #         rcuta_= RC['rcuta']
    #      if 're' in RC:
    #         re_   = RC['re']
    if rcut_ is None:
       rcut = {'C-C':2.5,'C-H':2.0,'C-N':2.5,'C-O':2.5,'C-F':2.5,'C-Al':3.2,
              'N-N':2.5,'N-O':2.5,'N-H':2.0,'N-F':2.5,'N-Al':3.2,
              'O-O':2.5,'O-H':2.0,'O-F':2.5,'O-Al':3.2,
              'H-H':2.0,'F-H':2.0,'H-Al':2.5,
              'F-F':2.5,'F-Al':3.2,
              'Al-Al':3.8,
              'others':2.5}
    else:
       rcut = rcut_

    # in princeple, the cutoff should 
    # cut the first nearest neithbors
    if rcuta_ is None:
       # rcuta = rcut
       rcuta = {'C-C':1.95,'C-H':1.75,'C-N':1.95,'C-O':1.95,'C-F':1.95,'C-Al':2.9,
               'N-N':1.95,'N-O':1.95,'N-H':1.75,'N-F':1.95,'N-Al':2.8,
               'O-O':1.95,'O-H':1.75,'O-F':1.95,'O-Al':2.8,
               'H-H':1.35,'F-H':1.75,'H-Al':2.5,
               'F-F':1.95,'F-Al':2.8,
               'Al-Al':3.4,
               'others':1.95}
    else:
       rcuta = rcuta_

    # in princeple, the cutoff should 
    # cut the first nearest neithbors
    if re_ is None:
       re = {'C-C':1.4,'C-H':1.10,'C-N':1.5,'C-O':1.28,'C-F':1.32,'C-Al':2.02,
             'N-N':1.5,'N-O':1.25,'N-H':1.07,'N-F':1.28,'N-Al':2.02,
             'O-O':1.25,'O-H':1.07,'O-F':1.24,'O-Al':1.91,
             'H-H':0.7,'H-F':0.88,'H-Al':1.55,
             'F-F':1.16,'F-Al':1.83,
             'Al-Al':2.8,
             'others':1.5}
    else:
       re = re_
       
    # with open('ffield.json','w') as fj:
    #      js.dump(RC,fj,sort_keys=True,indent=2)

    if bonds is not None:
       for bd in bonds:
           b = bd.split('-')
           bdr = b[1]+'-'+b[0]
           if bd in rcut:
              rcut[bdr]  = rcut[bd]
           elif bdr in rcut:
              rcut[bd]   = rcut[bdr]
           else:
              rcut[bd]   = rcut['others']
              rcut[bdr]   = rcut['others']
 
           if bd in rcuta:
              rcuta[bdr] = rcuta[bd] 
           elif bdr in rcuta:
              rcuta[bd]  = rcuta[bdr]
           else:
              rcuta[bd]   = rcuta['others']
              rcuta[bdr]   = rcuta['others']
 
           if bd in re:
              re[bdr] = re[bd] 
           elif bdr in re:
              re[bd]  = re[bdr]
           else:
              re[bd]   = re['others']
              re[bdr]   = re['others']
    return rcut,rcuta,re

