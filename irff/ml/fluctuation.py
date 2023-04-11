import pandas as pd
import json as js
import numpy as np
from irff.tools.vdw import vdw as Evdw

def fmorse(r,ro,De,alpha,eo=0.0):
    r_ = r - ro
    return De*(np.exp(-2.0*alpha*r_)-2.0*np.exp(-alpha*r_)) + eo

def make_fluct(fluct=0.1,bond=['C-C','H-H','O-O'],csv='fluct'):
    beup = {}
    belo = {}
    vup = {}
    vlo = {}
    for bd in bond:
        csv_ = csv+'_'+bd+'.csv'
        d    = pd.read_csv(csv_)
        r    = d['r']
        eb   = d['ebond']
        ev   = d['evdw']

        beup[bd] = []
        belo[bd] = []
        vup[bd]  = []
        vlo[bd]  = []

        for r_,eb_,ev_ in zip(r,eb,ev):
            up = 1.0+fluct
            lo = 1.0-fluct
            beup[bd].append((r_,eb_*lo))
            belo[bd].append((r_,eb_*up))

            vup[bd].append((r_,ev_*up))
            vlo[bd].append((r_,ev_*lo))
    return belo,beup,vlo,vup

def bo_fluct(fluct=0.1,bond=['C-C','H-H','O-O'],csv='bo_fluct'):
    boup = {}
    bolo = {}
    for bd in bond:
        csv_ = csv+'_'+bd+'.csv'
        d   = pd.read_csv(csv_)
        r   = d['r']
        bsi = d['bosi1']
        bpi = d['bopi1']
        bpp = d['bopp1']
        boup[bd] = []
        bolo[bd] = []
         
        for r_,bsi_,bpi_,bpp_ in zip(r,bsi,bpi,bpp):
            up = 1.0+fluct
            boup[bd].append((r_,bsi_*up,bpi_*up,bpp_*up))
            lo = 1.0-fluct
            bolo[bd].append((r_,bsi_*lo,bpi_*lo,bpp_*lo))
    return bolo,boup

def get_parameters(ffield):
    with open(ffield,'r') as lf:
        j = js.load(lf)
        p = j['p']
        m = j['m']
    return p,m

def harmonic(bd,ro=2.45,rst=2.1,red=2.8,rovdw=3.0,
             Di=None,Dj=None,
             De=1.0,alpha=4.0,eo=0.0,
             npoints=7,fluctuation=0.1):
    p,_ = get_parameters('ffield.json')
    b  = bd.split('-')
    gamma  = np.sqrt(p['gamma_'+b[0]]*p['gamma_'+b[1]])
    gammaw = np.sqrt(p['gammaw_'+b[0]]*p['gammaw_'+b[1]])
    
    R = np.linspace(rst, red, num=npoints)
    evdw       = Evdw(R,Devdw=p['Devdw_'+bd]*4.3364432032e-2,gamma=gamma,gammaw=gammaw,
                      vdw1=p['vdw1'],rvdw=p['rvdw_'+bd],alfa=p['alfa_'+bd])
    evdw_ro    = Evdw(ro,Devdw=p['Devdw_'+bd]*4.3364432032e-2,gamma=gamma,gammaw=gammaw,
                      vdw1=p['vdw1'],rvdw=p['rvdw_'+bd],alfa=p['alfa_'+bd])
    evdw_rovdw = Evdw(rovdw,Devdw=p['Devdw_'+bd]*4.3364432032e-2,gamma=gamma,gammaw=gammaw,
                      vdw1=p['vdw1'],rvdw=p['rvdw_'+bd],alfa=p['alfa_'+bd])

    # em_ = fmorse(rovdw,ro,De,alpha)
    Eo = evdw_ro - evdw_rovdw  # + em_

    Eb = evdw_ro - evdw 
    Eb = Eb - Eo
   
    e_morse = fmorse(R,ro,De,alpha,eo=eo)
    Eb += e_morse 
    # print('r_: {:9.6f} r: {:9.6f} ro: {:9.6f} E: {:9.6f}'.format(r_,r,ro,e_morse))
        
    belo = []
    beup = []
    for r_,eb_ in zip(R,Eb):
        if Di is None:
           Di = (0,100)
        if Dj is None:
           Dj = (0,100)
        belo.append((r_,Di[0],Di[1],Dj[0],Dj[1],eb_*(1+fluctuation)))
        beup.append((r_,Di[0],Di[1],Dj[0],Dj[1],eb_*(1-fluctuation)))
    return belo,beup 

def morse(morp={}):
    ''' make potential bounds according morse potential '''
    belo,beup    = {},{}
    for bd in morp:
        p = morp[bd]
        fluct = 0.1 if 'fluctuation' not in p else p['fluctuation']
        if 'rrange' in p:
           rst_ = p['rrange'][0]
           red_ = p['rrange'][1]
        else:
           rst_ = p['ro']*0.8
           red_ = p['ro']+max(0.4,p['ro']*0.3)
        
        Di_ = None if 'Di' not in p else p['Di']
        Dj_ = None if 'Dj' not in p else p['Dj']
        rcut = 3.0 if 'rcut' not in p else p['rcut']
        np_ = 10   if 'npoints' not in p else p['npoints']

        belo[bd],beup[bd]  = harmonic(bd,Di=Di_,Dj=Dj_,rovdw=rcut,rst=rst_,red=red_,
                                      ro=p['ro'],De=p['De'],alpha=p['alpha'],eo=p['eo'],
                                      npoints=np_,fluctuation=fluct) 
        # belo[bd]  = harmonic(bd,Di=Di_,Dj=Dj_,rovdw=rcut,
        #                      ro=p['ro'],rst=rst_,red=red_,De=p['De']*(1+fluct),alpha=p['alpha'],npoints=np_) 
    return belo,beup

