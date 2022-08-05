import pandas as pd


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

 

