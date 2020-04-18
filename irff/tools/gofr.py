#!/usr/bin/env python
from __future__ import print_function
import numpy as np


def gofr(r,bins=0.05,rcut=8.0,natom1=1.0,natom2=1.0,volume=1.0,s=1.0):
    nbin = int(rcut/bins)
    gr_  = np.zeros([nbin])
    rou  = natom2/volume
    pi   = 3.1415926

    for r_ in r:
        if r_<=rcut and r_>0.001:
           nb = int(r_/bins)
           rsq= ((nb+1)*bins)**2
           gr_[nb] += s/(rou*4*pi*rsq*bins*natom1)

    bin_= np.array([bins*i+0.5*bins for i in range(nbin)])
    # bin_= np.linspace(0.0,rcut,nbin)
    return bin_,gr_



# lst = list(dit)
# lst2 = list(dit.values())
