#!/usr/bin/env python
import json as js
import csv
import numpy as np
import matplotlib.pyplot as plt
from irff.tools.vdw import vdw as Evdw


def get_parameters(ffield):
    with open(ffield, 'r') as lf:
        j = js.load(lf)
        p = j['p']
        m = j['m']
    return p, m


def Ebond(bd, p, R,ro=1.6, rovdw=3.0, k=0.1):
    b = bd.split('-')
    gamma = np.sqrt(p['gamma_'+b[0]]*p['gamma_'+b[1]])
    gammaw = np.sqrt(p['gammaw_'+b[0]]*p['gammaw_'+b[1]])

    evdw = Evdw(R, Devdw=p['Devdw_'+bd]*4.3364432032e-2, gamma=gamma, gammaw=gammaw,
                vdw1=p['vdw1'], rvdw=p['rvdw_'+bd], alfa=p['alfa_'+bd])

    evdw_ro = Evdw(ro, Devdw=p['Devdw_'+bd]*4.3364432032e-2, gamma=gamma, gammaw=gammaw,
                   vdw1=p['vdw1'], rvdw=p['rvdw_'+bd], alfa=p['alfa_'+bd])
    evdw_rovdw = Evdw(rovdw, Devdw=p['Devdw_'+bd]*4.3364432032e-2, gamma=gamma, gammaw=gammaw,
                      vdw1=p['vdw1'], rvdw=p['rvdw_'+bd], alfa=p['alfa_'+bd])

    E0 = evdw_ro - evdw_rovdw + k*(ro-rovdw)**2
    Eb = []
    for i, r in enumerate(R):
        Eb.append(evdw_ro - evdw[i])

    Eb = np.array(Eb)
    Eb = Eb - E0

    for i, r in enumerate(R):
        Eb[i] += k*(r-ro)**2
    return Eb,evdw


def integrate(bd,ro=1.26,rst=1.15,red=1.5,rovdw=3.0,k=0.1,npoints=10): 
    p, _ = get_parameters('ffield.json')
    R = np.linspace(rst, red, num=npoints)
    
    k_lo = 3.1
    k_up = 3.0

    eb1,evdw = Ebond(bd, p, R,ro=ro,rovdw=rovdw,k=k_lo)
    eb2,evdw = Ebond(bd, p, R,ro=ro, rovdw=rovdw, k=k_up)

    # Esi= Eb/-p['Desi_'+bd]
    # Esi= Esi/4.3364432032e-2
    # print(Esi)

    plt.figure()
    plt.subplot(2, 2, 1)

    plt.plot(R, evdw, alpha=0.8, linewidth=2, linestyle='-',  # evdw+
             marker='o', color='r', ms='8', markerfacecolor='none',
             label=r'$E_{vdw}$')
    plt.legend(loc='best', edgecolor='yellowgreen')

    plt.subplot(2, 2, 2)
    plt.plot(R, eb1, alpha=0.8, linewidth=2, linestyle='-',  # evdw+
             marker='o', color='b', ms='6', markerfacecolor='none',
             label=r'$E_{bond}^{lower}$')
    plt.plot(R, eb2, alpha=0.8, linewidth=2, linestyle='-.',  # evdw+
             marker='s', color='b', ms='6', markerfacecolor='none',
             label=r'$E_{bond}^{upper}$')
    plt.legend(loc='best', edgecolor='yellowgreen')
    # plt.savefig('Bond_Energy_{:s}.pdf'.format(bd))

    plt.subplot(2, 1, 2)
    plt.plot(R, eb1+evdw, alpha=0.8, linewidth=2, linestyle='-',  # evdw+
             marker='o', color='purple', ms='8', markerfacecolor='none',
             label=r'$E_{vdw}$+$E_{bond} lower$')
    plt.plot(R, eb2+evdw, alpha=0.8, linewidth=2, linestyle='-.',  # evdw+
             marker='D', color='purple', ms='8', markerfacecolor='none',
             label=r'$E_{vdw}$+$E_{bond} upper$')
    plt.legend(loc='best', edgecolor='yellowgreen')

    plt.show()
    plt.close()


if __name__ == '__main__':
    ''' according the value of Vdw compute mim-bond energy '''
    integrate('O-Fe',rst=2.0,ro=2.17,red=2.4,rovdw=3.0,npoints=20)  
    #integrate('Fe-Fe',rst=2.2,ro=2.45,red=3.0,rovdw=3.0,npoints=10) 
    #integrate('O-O',ro=1.26,rst=1.15,red=1.5,rovdw=3.0,npoints=10)  
