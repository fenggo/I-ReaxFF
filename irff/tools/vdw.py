import numpy as np


def taper(r,vdwcut=10.0):
    tp = 1.0+np.divide(-35.0,np.power(vdwcut,4.0))*np.power(r,4.0)+ \
         np.divide(84.0,np.power(vdwcut,5.0))*np.power(r,5.0)+ \
         np.divide(-70.0,np.power(vdwcut,6.0))*np.power(r,6.0)+ \
         np.divide(20.0,np.power(vdwcut,7.0))*np.power(r,7.0)
    return tp


def vdw(r,Devdw=0.01,gamma=1.0,gammaw=1.0,vdw1=1.0,rvdw=2.0,alfa=12.0):
    gm3  = np.power(1.0/gamma,3.0)
    r3   = np.power(r,3.0)

    rr   = np.power(r,vdw1) + np.power(1.0/gammaw,vdw1)
    f13  = np.power(rr,1.0/vdw1)
    tpv  = taper(r)

    expvdw1 = np.exp(0.5*alfa*(1.0-np.divide(f13,2.0*rvdw)))
    expvdw2 = np.square(expvdw1)
    evdw    = tpv*Devdw*(expvdw2-2.0*expvdw1)
    return evdw

