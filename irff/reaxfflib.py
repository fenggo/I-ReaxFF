from __future__ import print_function
from os import system, getcwd, chdir,listdir
from os.path import isfile,exists,isdir
import numpy as np


p_name = ['boc1','boc2','coa2','trip4','trip3','kc2','ovun6','trip2',
         'ovun7','ovun8','trip1','swa','swb','n.u.','val6','lp1',
         'val9','val10','lp3','pen2','pen3','pen4','n.u.','tor2',
         'tor3','tor4','n.u.','cot2','vdw1','cutoff','coa4','ovun4',
         'ovun3','val8','acut','hbtol','n.u.','n.u.','coa3']
         
line_spec = []
line_spec.append(['rosi','val','mass','rvdw','Devdw','gamma','ropi','vale'])
line_spec.append(['alfa','gammaw','valang','ovun5','n.u.','chi','mu','atomic'])
line_spec.append(['ropp','lp2','n.u.','boc4','boc3','boc5','n.u.','n.u.'])
line_spec.append(['ovun2','val3','n.u.','valboc','val5','n.u.','n.u.','n.u.'])

line_bond = []
line_bond.append(['Desi','Depi','Depp','be1','bo5','corr13','bo6','ovun1'])
line_bond.append(['be2','bo3','bo4','n.u.','bo1','bo2','ovcorr','be0'])

line_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
line_ang  = ['theta0','val1','val2','coa1','val7','pen1','val4']
line_tor  = ['V1','V2','V3','tor1','cot1','n.u.','n.u.']
line_hb   = ['rohb','Dehb','hb1','hb2']


def read_lib(p={},zpe=False,libfile='ffield',
             p_name=p_name,line_spec=line_spec,
             line_bond=line_bond,line_offd=line_offd,
             line_ang=line_ang,line_tor=line_tor,line_hb=line_hb):
    print('-  initial variable read from: %s.' %libfile)
    if isfile(libfile):
       flib = open(libfile,'r')
       lines= flib.readlines()
       flib.close()
       npar = int(lines[1].split()[0])
       
       if npar>len(p_name):
          print('error: npar >39')
          exit()

       zpe_= {}
       if zpe:
          lin = lines[0].split()
          for i in range(int(len(lin)/2-1)):
              k = lin[i*2+1]
              zpe_[k] = float(lin[i*2])

       for i in range(npar):
           pn = p_name[i]
           p[pn] = float(lines[2+i].split()[0]) 

       # ---------   parameters for species   ---------
       nofc   = 1                 #  number of command line
       nsl    = len(line_spec)
       nsc    = nsl
       npar   = npar + 1
       nspec  = int(lines[nofc+npar].split()[0])
       spec   = []   
       for i in range(nspec):
           spec.append(lines[nofc+npar+nsc+i*nsl].split()[0]) # read in species name in first line
           for il,line in enumerate(line_spec):
               ls = 1 if il == 0 else 0
               for ip,pn in enumerate(line):
                   p[pn+'_'+spec[i]] = np.float(lines[nofc+npar+nsc+i*nsl+il].split()[ls+ip])

       # ---------  parameters for bonds   ---------
       bonds = []
       nbl = len(line_bond)
       nbc = nbl
       nbond = int(lines[nofc+npar+nsc+nspec*nsl].split()[0])
       for i in range(nbond):
           b1= int(lines[nofc+npar+nsc+nspec*nsl+nbc+i*nbl].split()[0])
           b2= int(lines[nofc+npar+nsc+nspec*nsl+nbc+i*nbl].split()[1])
           bond = spec[b1-1] + '-' +spec[b2-1]
           bonds.append(bond)
           for il,line in enumerate(line_bond):
               ls = 2 if il == 0 else 0
               for ip,pn in enumerate(line):
                   p[pn+'_'+bond] = np.float(lines[nofc+npar+nsc+nspec*nsl+nbc+i*nbl+il].split()[ls+ip])

       # ---------   parameters for off-diagonal bonds   ---------
       offd  = []
       nol   = 1
       noc   = nol
       noffd = int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl].split()[0])
       
       for i in range(noffd):
           b1=int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+i].split()[0])
           b2=int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+i].split()[1])
           bond = spec[b1-1] + '-' +spec[b2-1]
           offd.append(bond)
           for ip,pn in enumerate(line_offd):
               p[pn+'_'+bond] = np.float(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+i].split()[2+ip])
           
       # ---------   parameters for angles   ---------
       angs = []
       nal  = 1
       nac  = nal
       nang = int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd].split()[0])
       for i in range(nang):
           l = lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+i].split()
           b1,b2,b3  = int(l[0]),int(l[1]),int(l[2])
           # print(l)
           ang = spec[b1-1] + '-' +spec[b2-1] + '-' +spec[b3-1]
           angr= spec[b3-1] + '-' +spec[b2-1] + '-' +spec[b1-1]
           if (not ang in angs) and (not angr in angs):
              angs.append(ang)
              for ip,pn in enumerate(line_ang):
                  p[pn+'_'+ang] = np.float(l[3+ip])

       # ---------   parameters for torsions   --------- 
       ntl  = 1 
       ntc  = ntl
       ntor = int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+nang].split()[0])
       tors = []
       for i in range(ntor):
           l = lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+nang+ntc+i].split()
           b1,b2,b3,b4 = int(l[0]),int(l[1]),int(l[2]),int(l[3])
           e2,e3 = spec[b2-1],spec[b3-1]
           e1 = 'X' if b1==0  else spec[b1-1]
           e4 = 'X' if b4==0  else spec[b4-1]
            
           tor  = e1+'-'+e2 +'-'+e3 +'-'+e4
           tor1 = e4+'-'+e2 +'-'+e3 +'-'+e1
           torr = e4+'-'+e3 +'-'+e2 +'-'+e1
           torr1= e1+'-'+e3 +'-'+e2 +'-'+e4
           if (not tor in tors) and (not torr in tors) and\
              (not tor1 in tors) and (not torr1 in tors):
              tors.append(tor)
              for ip,pn in enumerate(line_tor):
                  p[pn+'_'+tor] = np.float(l[4+ip])

       # ---------   parameters for HBs   ---------
       hbs = []
       nhl = 1
       nhc = nhl
       nhb = int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+nang+ntc+ntor].split()[0])
       for i in range(nhb):
           l = lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+nang+ntc+ntor+nhc+i].split()
           hb1,hb2,hb3  = int(l[0]),int(l[1]),int(l[2])
           hb = spec[hb1-1] + '-' +spec[hb2-1] + '-' +spec[hb3-1]
           hbs.append(hb)
           for ip,pn in enumerate(line_hb):
               p[pn+'_'+hb] = np.float(l[3+ip])
    else:
       print('-  Error: lib file is not found!')
       p = None
    return p,zpe_,spec,bonds,offd,angs,tors,hbs


def write_lib(p,spec,bonds,offd,angs,tors,hbs,zpe=None,libfile='irnn.lib',
              p_name=p_name,line_spec=line_spec,
              line_bond=line_bond,line_offd=line_offd,
              line_ang=line_ang,line_tor=line_tor,line_hb=line_hb,
              loss=None,
              logo='!-ReaxFF-Parameter-From-Machine-Learning-Computational-Materials-Science-172-(2020)-109393'):
    flib = open(libfile,'w')

    if zpe is None:
       if loss is None:
          print(logo, file=flib)
       else:
          print(logo+'--Loss:%s' %str(loss), 
                file=flib)
    else:
       ZPE = ''
       for key in zpe:
           ZPE += str(zpe[key]) +' '+key+' '
       if loss is None:
          print('%s # %s ' %(ZPE,'zpe_energy'), file=flib)
       else:
          print('%s # %s ' %(ZPE,'zpe_energy-with-Loss:%s') %str(loss), file=flib)  
                
    print('%d         ! Number of general parameters ' %len(p_name), file=flib)

    for pn in p_name:
        if pn == 'n.u.':
           print('%10.4f      ! %s ' %(0.0,pn), file=flib)
        else:
           print('%10.4f      ! %s ' %(p[pn],pn), file=flib)
    for i in range(len(line_spec)):
        txt = '%d ! Nr of atoms; atomID;' %len(spec) if i==0 else '            '
        for ls in line_spec[i]:
            txt += ls + '; '
        print(txt,file=flib)

    for sp in spec:  # atomic species 
        for i,line in enumerate(line_spec):
            txt = '%2s' %sp if i==0 else '  '
            for pn in line:
                pn_ = pn+'_'+sp
                if pn=='n.u.':
                   v = 0.0
                else:
                   v = p[pn_]
                txt += '%9.4f' %v
            print(txt,file=flib)

    # ---------   parameters for bonds   --------- 
    for i,line in enumerate(line_bond): 
        txt = '%3d  ! Nr of bonds;' %len(bonds) if i==0  else '                   '
        for ls in line:
            txt += ls +'; '
        print(txt,file=flib)
    for bs in bonds:
        for i,line in enumerate(line_bond):
            txt = '%3d%3d' %(spec.index(bs.split('-')[0])+1,spec.index(bs.split('-')[1])+1) if i==0  else '      '
            for pn in line:
                pn_ = pn+'_'+bs
                if pn=='n.u.':
                   v = 0.0
                else:
                   if pn_ in p:
                      v = p[pn_]
                   else:
                      v = 0.0 # np.random.normal(loc=0.0, scale=0.2, size=0)
                txt += '%9.4f' %v
            print(txt,file=flib)

    # ---------    parameters for off-diagonal bonds   --------- 
    txt = '%3d  ! off-diagonal terms;' %len(offd) 
    for ls in line_offd:
        txt += ls +'; '
    print(txt,file=flib)

    for bs in offd:
        b = bs.split('-')
        if len(b)==1:
           continue
        if b[0]==b[1]:
           continue
        txt = '%3d%3d' %(spec.index(bs.split('-')[0])+1,spec.index(bs.split('-')[1])+1) 
        for pn in line_offd:
            pn_ = pn+'_'+bs
            if pn=='n.u.':
               v = 0.0
            else:
               v = p[pn_]
            txt += '%9.4f' %v
        print(txt,file=flib)

    # ---------   parameters for angles   ---------
    txt = '%3d  ! Nr of angles;' %len(angs) 
    for ls in line_ang:
        txt += ls +'; '
    print(txt,file=flib)

    for a in angs:
        a_ = a.split('-')
        txt = '%3d%3d%3d' %(spec.index(a_[0])+1,spec.index(a_[1])+1,spec.index(a_[2])+1)
        for pn in line_ang:
            pn_ = pn+'_'+a
            if pn=='n.u.':
               v = 0.0
            else:
               v = p[pn_]
            txt += '%9.4f' %v
        print(txt,file=flib)

    # ---------  parameters for tors   ---------
    txt = '%3d  ! Nr of torsions;' %len(tors) 
    for ls in line_tor:
        txt += ls +'; '
    print(txt,file=flib)
    
    for a in tors:
        aa = a.split('-')
        e1 = spec.index(aa[0])+1 if aa[0] != 'X' else 0
        e2 = spec.index(aa[1])+1
        e3 = spec.index(aa[2])+1
        e4 = spec.index(aa[3])+1 if aa[3] != 'X' else 0
        txt = '%3d%3d%3d%3d' %(e1,e2,e3,e4)
        for pn in line_tor:
            pn_ = pn+'_'+a
            if pn=='n.u.':
               v = 0.0
            else:
               v = p[pn_]
            txt += '%9.4f' %v
        print(txt,file=flib)

    # ---------  parameters for HBs   ---------
    txt = '%3d  ! Nr of Hbonds;' %len(hbs) 
    for ls in line_hb:
        txt += ls +'; '
    print(txt,file=flib)

    for a in hbs:
        txt = '%3d%3d%3d' %(spec.index(a.split('-')[0])+1,spec.index(a.split('-')[1])+1,spec.index(a.split('-')[2])+1)
        for pn in line_hb:
            pn_ = pn+'_'+a
            if pn=='n.u.':
               v = 0.0
            else:
               v = p[pn_]
            txt += '%9.4f' %v
        print(txt,file=flib)
    flib.close()

