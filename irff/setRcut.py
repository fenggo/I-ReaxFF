

def setRcut(bonds):
    # in princeple, the cutoff should 
    # cut the second nearest neithbors
    rcut = {'C-C':2.8,'C-H':2.5,'C-N':2.8,'C-O':2.8,'C-F':2.8,'C-Al':3.2,
            'N-N':2.8,'N-O':2.7,'N-H':2.5,'N-F':2.5,'N-Al':3.2,
            'O-O':2.7,'O-H':2.5,'O-F':2.5,'O-Al':3.2,
            'H-H':2.3,'F-H':2.5,'H-Al':3.0,
            'F-F':2.8,'F-Al':3.2,
            'Al-Al':3.6,
            'others':2.8}

    # in princeple, the cutoff should 
    # cut the first nearest neithbors
    rcuta = {'C-C':1.95,'C-H':1.75,'C-N':1.95,'C-O':1.95,'C-F':1.95,'C-Al':2.9,
             'N-N':1.95,'N-O':1.95,'N-H':1.75,'N-F':1.95,
             'O-O':1.95,'O-H':1.75,'O-F':1.95,
             'H-H':1.35,'F-H':1.75,
             'F-F':1.95,
             'Al-Al':3.3,
             'others':1.95}

    # in princeple, the cutoff should 
    # cut the first nearest neithbors
    re = {'C-C':1.54,'C-H':1.10,'C-N':1.5,'C-O':1.43,'C-F':1.37,'C-Al':2.02,
          'N-N':1.5,'N-O':1.25,'N-H':1.07,'N-F':1.28,'N-Al':2.02,
          'O-O':1.16,'O-H':1.07,'O-F':1.24,'O-Al':1.91,
          'H-H':0.7,'H-F':0.88,'H-Al':1.55,
          'F-F':1.16,'F-Al':1.83,
          'Al-Al':2.5,
          'others':1.5}

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

