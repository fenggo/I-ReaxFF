#!/usr/bin/env python
import sys
import argparse
import json as js
import numpy as np
import matplotlib.pyplot as plt
from irff.tools.vdw import vdw as Evdw
from irff.ml.data import get_md_data #,get_data,get_bond_data,get_atoms_data

def fmorse(r,ro,De,alpha):
    r_ = r - ro
    return De*(np.exp(-2.0*alpha*r_)-2.0*np.exp(-alpha*r_))

def get_parameters(ffield):
    with open(ffield,'r') as lf:
        j = js.load(lf)
        p = j['p']
        m = j['m']
    return p,m

def Ebond(bd, p, R,ro=1.6, rovdw=3.0, k=0.1,potential='morse',De=1.0,alpha=1.0):
    b = bd.split('-')
    gamma = np.sqrt(p['gamma_'+b[0]]*p['gamma_'+b[1]])
    gammaw = np.sqrt(p['gammaw_'+b[0]]*p['gammaw_'+b[1]])

    evdw = Evdw(R, Devdw=p['Devdw_'+bd]*4.3364432032e-2, gamma=gamma, gammaw=gammaw,
                vdw1=p['vdw1'], rvdw=p['rvdw_'+bd], alfa=p['alfa_'+bd])

    evdw_ro = Evdw(ro, Devdw=p['Devdw_'+bd]*4.3364432032e-2, gamma=gamma, gammaw=gammaw,
                   vdw1=p['vdw1'], rvdw=p['rvdw_'+bd], alfa=p['alfa_'+bd])
    evdw_rovdw = Evdw(rovdw, Devdw=p['Devdw_'+bd]*4.3364432032e-2, gamma=gamma, gammaw=gammaw,
                      vdw1=p['vdw1'], rvdw=p['rvdw_'+bd], alfa=p['alfa_'+bd])

    if potential == 'harm':
       Eo = evdw_ro - evdw_rovdw + k*(ro-rovdw)**2
    elif potential == 'morse':
       # em_ = 0.0 if rovdw is None else fmorse(rovdw,ro,De,alpha)
       Eo = evdw_ro - evdw_rovdw # + em_
    else:
       raise RuntimeError('Potential not supported!')
    Eb = []
    for i, r in enumerate(R):
        Eb.append(evdw_ro - evdw[i])

    Eb = np.array(Eb)
    # R  = np.array(R)
    Eb = Eb - Eo

    for i, r in enumerate(R):
        if potential == 'harm':
           e_harm  = k*(r-ro)**2
           Eb[i] += e_harm
        elif  potential == 'morse':
           e_morse = fmorse(r,ro,De,alpha)
           Eb[i] += e_morse 
           # print(' r: {:9.6f} ro: {:9.6f} E: {:9.6f} Eb: {:9.6f}'.format(r,ro,e_morse,Eb[i]))
        else:
           raise RuntimeError('Potential not supported!')
    return Eb,evdw

def be(gen='gulp.traj',bonds=None,ro=2.17,rcut=3.0,k=(5.0,4.75),
       potential='morse',
       De=(0.9,1.0),alpha=(1.9,1.0)):
    # D, Bp, B, R, E = get_atoms_data(gen=gen,bonds=bonds)
    D, Bp, B, R, E = get_md_data(images=None, traj=gen,bonds=bonds)
    p,_ = get_parameters('ffield.json') 
    for bd in bonds:
        for i,bp in enumerate(Bp[bd]):
            eb = - p['Desi_'+bd]* E[bd][i] * 4.3364432032e-2 
            print('id: {:3d} R: {:6.4f} '
                  'Di: {:7.4f} B\': {:6.4f} Dj: {:7.4f} B: {:7.5f} '
                  'Ebd: {:7.5f}'.format(i,R[bd][i],D[bd][i][0],D[bd][i][1],D[bd][i][2],
                                        np.sum(B[bd][i]),eb))

    eb1, evdw = Ebond(bd, p, R[bd],rovdw=rcut, ro=ro,k=k[0],De=De[0],alpha=alpha[0],potential=potential)
    eb2, evdw = Ebond(bd, p, R[bd],rovdw=rcut, ro=ro,k=k[1],De=De[1],alpha=alpha[1],potential=potential)
    
    plt.figure()
    
    ebd_= np.array(E[bd])
    ebd = -p['Desi_'+bd]*ebd_*4.3364432032e-2  ## Bond-Energy
    
    R_   = np.array(R[bd])
    ind  = np.argsort(R_)
    eb1_ = eb1[ind]
    eb2_ = eb2[ind]
    Rs   = R_[ind]
    # for i in ind:
    #     print('{:9.6f} {:9.6f}'.format(R_[i],ebd[i]))
    
    plt.subplot(2,2,1) 
    plt.scatter(R[bd],evdw,alpha=0.8,marker='o',color='r',s=10,
                label=r'$E_{vdw}$')
    plt.legend(loc='best',edgecolor='yellowgreen')  

    plt.subplot(2,2,2) 
    plt.scatter(R[bd],ebd,alpha=0.8,marker='o',color='r',s=10,
               label=r'$E_{bond}$')
    #plt.scatter(R[bd],eb1,alpha=0.8,marker='^',color='y',s=2)
    #plt.scatter(R[bd],eb2,alpha=0.8,marker='v',color='c',s=2)
    plt.plot(Rs,eb1_,alpha=0.8,color='y',linestyle='dashdot')
    plt.plot(Rs,eb2_,alpha=0.8,color='c',linestyle='dashdot')
    plt.fill_between(Rs, eb1_, eb2_, color='palegreen',
                     alpha=0.2)
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(2,1,2) 
    plt.scatter(R[bd],evdw+ebd,alpha=0.8,marker='o',color='r',s=10,
                label=r'$E_{vdw}$ + $E_{bond}$')
    
    elo = evdw+eb1
    plt.plot(Rs,elo[ind],alpha=0.8, linestyle='dashdot',
                label=r'$E_{bond+vdw}^l$')
    eup = evdw+eb2
    plt.plot(Rs,eup[ind],alpha=0.8,linestyle='dashdot',
                label=r'$E_{bond+vdw}^u$')
    
    plt.fill_between(Rs, elo[ind], eup[ind], color='bisque',
                     alpha=0.2)
    # plt.savefig('vdw_energy_{:s}.pdf'.format(bd))
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('morse-{:s}.svg'.format(gen.split('.')[0]))
    # plt.show()
    plt.close() 


if __name__ == '__main__':
   ''' constrain the bond-energy according harmonic approximation
       Run with commond: ./be_harm.py  '''
   parser = argparse.ArgumentParser(description='Harmonic bond energy analysis')
   parser.add_argument('--traj',default='md.traj',type=str, help='trajectory name')
   parser.add_argument('--b',default='C-H',type=str, help='bond')
   args = parser.parse_args(sys.argv[1:])
   
   morp = {}
   morp['C-N'] = {'ro':1.45,'De':1.36,'alpha':3.30,'rrange':(1.38,1.8),
                  'Di':(4.8,8.5),'Dj':(4.8,8.5),
                  'fluctuation':0.15} 
   #morp['O-N'] = {'ro':1.285,   'De':1.1, 'alpha':2.83,'rrange':(1.20,1.8),
                   #'fluctuation':0.1} 
   #morp['N-N'] = {'ro':1.46,    'De':0.95,'alpha':3.30,'rrange':(1.38,1.8),
                   #'fluctuation':0.06}
   morp['C-H'] = {'ro':1.075,    'De':5.0,'alpha':1.6,'rrange':(0.87,1.5),
                   'rvdw':2.1,'fluctuation':0.1}
   #morp['H-H'] = {'ro':0.77,    'De':1.95,'alpha':2.53,'rrange':(0.6,1.6),
                   #'fluctuation':0.1}
   bd = args.b # 'C-H'
   # CN:     ro=1.467391, De=3.271696, alpha=2.381341
   # CN(H2): ro=1.420637, De=3.772875, alpha=1.931173
   # HN:     ro=1.066021, De=4.011183, alpha=1.767804
   # p = morp[bd]
   rcut_ = 3.0 if 'rvdw' not in morp[bd] else morp[bd]['rvdw']
   be(gen=args.traj,bonds=[bd],
      ro=morp[bd]['ro'],
      De=(morp[bd]['De']*(1.0+morp[bd]['fluctuation']),
          morp[bd]['De']*(1.0-morp[bd]['fluctuation'])),
      alpha=(morp[bd]['alpha'],morp[bd]['alpha']),
      rcut=rcut_)

   
   #ro         ={'C-N':1.495982,'O-N':1.285,   'C-H':1.0937,  'N-N':1.46,
                #'H-N':1.0937}
   #De         ={'C-N':0.58,    'O-N':1.1,     'C-H':6.3,     'N-N':0.9,
                #'H-N':1.95}
   #alfa       ={'C-N':3.3,     'O-N':2.832588,'C-H':1.533029,'N-N':2.8,
                #'H-N':2.533029}
   #fluctuation={'C-N':0.1,     'O-N':0.15,    'C-H':0.05,    'N-N':0.18,
                #'H-N':0.12}
