from __future__ import print_function
from os import system, getcwd, chdir,listdir
from os.path import isfile,exists,isdir
import numpy as np

# contains modules that write the parameters to ffield or in GULP format .lib file.
# write_ffield, write_lib


p_name = ['boc1','boc2','coa2','trip4','trip3','kc2','ovun6','trip2',
         'ovun7','ovun8','trip1','swa','swb','n.u.','val6','lp1',
         'val9','val10','n.u.','pen2','pen3','pen4','n.u.','tor2',
         'tor3','tor4','n.u.','cot2','vdw1','cutoff','coa4','ovun4',
         'ovun3','val8','acut','hbtol','n.u.','n.u.','coa3']
         
line_spec = []
line_spec.append(['rosi','val','mass','rvdw','Devdw','gamma','ropi','vale'])
line_spec.append(['alfa','gammaw','valang','ovun5','n.u.','chi','mu','atomic'])
line_spec.append(['ropp','lp2','n.u.','boc4','boc3','boc5','n.u.','n.u.'])
line_spec.append(['ovun2','val3','n.u.','valboc','val5','n.u.','n.u.','n.u.'])

line_bond = []
line_bond.append(['Desi','Depi','Depp','be1','bo5','corr13','bo6','ovun1'])
line_bond.append(['be2','bo3','bo4','n.u.','bo1','bo2','ovcorr','n.u.'])

line_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
line_ang  = ['theta0','val1','val2','coa1','val7','pen1','val4']
line_tor  = ['V1','V2','V3','tor1','cot1','n.u.','n.u.']
line_hb   = ['rohb','Dehb','hb1','hb2']


def read_ffield(p={},zpe=False,libfile='ffield',
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
              try:
                 zpe_[k] = float(lin[i*2])
              except ValueError:
                 zpe_[k] = 0.0

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


def write_ffield(p,spec,bonds,offd,angs,tors,hbs,zpe=None,libfile='ffield',
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

def write_lib(p,spec,bonds,offd,angs,tors,hbs,
              m=None,mf_layer=(6,0),be_layer=(6,0),vdw_layer=(6,0),
              libfile='reax.lib',vdwcutoff=10.0,coulcutoff=10.0):
    ''' write the force field parameters and weight and bias of neural networks to GULP library file '''
    nn = False if m is None else True

    glib = open(libfile,'w')
    print('#',file=glib)
    if nn:
       print('# ReaxFF-nn Machine Learning Potential: ',file=glib)
    else:
       print('# ReaxFF Parameter From Machine Learning:',file=glib)
    print('#',file=glib)
    print('# Computational Materials Science 172 (2020) 109393',file=glib)
    print('#',file=glib)
    print('#',file=glib)
    print('# Cutoffs for VDW & Coulomb terms',file=glib)
    print('reaxFFvdwcutoff      {:12.8f}'.format(vdwcutoff),file=glib)
    print('reaxFFqcutoff        {:12.8f}'.format(coulcutoff),file=glib)
    print('#',file=glib)
    print('# Bond order threshold - check anglemin as this is cutof2 given in control file',file=glib)
    print('reaxFFtol        {:12.8f}  {:8.8f} {:8.8f}'.format(p['cutoff']*0.01,p['acut'],p['hbtol']),file=glib)
    print('#',file=glib)
    print('# Species independent parameters ',file=glib)
    print('#',file=glib)
    print('reaxff0_bond     {:12.8f}  {:12.8f}'.format(p['boc1'],p['boc2']),file=glib)
    print('reaxff0_over     {:12.8f}  {:12.8f} {:12.8f}  {:12.8f} {:12.8f}'.format(p['ovun3'],p['ovun4'],p['ovun6'],p['ovun7'],p['ovun8']),file=glib)
    print('reaxff0_valence  {:12.8f}  {:12.8f} {:12.8f}  {:12.8f}'.format(p['val6'],p['val8'],p['val9'],p['val10']),file=glib)
    print('reaxff0_penalty  {:12.8f}  {:12.8f} {:12.8f}'.format(p['pen2'],p['pen3'],p['pen4']),file=glib)
    print('reaxff0_torsion  {:12.8f}  {:12.8f} {:12.8f}  {:12.8f}'.format(p['tor2'],p['tor3'],p['tor4'],p['cot2']),file=glib)
    print('reaxff0_vdw      {:12.8f}'.format(p['vdw1']),file=glib)
    print('reaxff0_lonepair {:12.8f}'.format(p['lp1']),file=glib)
    print('#',file=glib)
    print('# Species parameters ',file=glib)
    print('#',file=glib)
    print('reaxff1_radii',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f} {:12.8f} {:12.8f}'.format(sp,p['rosi_'+sp],p['ropi_'+sp],p['ropp_'+sp]),file=glib)
    print('reaxff1_valence',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(sp,p['val_'+sp],p['valboc_'+sp],p['vale_'+sp],p['valang_'+sp]),file=glib)
    print('reaxff1_over',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(sp,p['boc3_'+sp],p['boc4_'+sp],p['boc5_'+sp],p['ovun2_'+sp]),file=glib)
    print('reaxff1_under kcal',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f}'.format(sp,p['ovun5_'+sp]),file=glib)
    print('reaxff1_lonepair kcal',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f} {:12.8f}'.format(sp,0.5*(p['vale_'+sp]-p['val_'+sp]),p['lp2_'+sp]),file=glib)
    print('reaxff1_angle',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f} {:12.8f}'.format(sp,p['val3_'+sp],p['val5_'+sp]),file=glib)     
    print('reaxff1_morse kcal',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(sp,p['alfa_'+sp],p['Devdw_'+sp],p['rvdw_'+sp],p['gammaw_'+sp]),file=glib) 
    print('#',file=glib)
    print('# Element parameters',file=glib)
    print('#',file=glib)
    print('reaxff_chi',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f}'.format(sp,p['chi_'+sp]),file=glib)   
    print('reaxff_mu',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f}'.format(sp,p['mu_'+sp]),file=glib) 
    print('reaxff_gamma',file=glib)
    for sp in spec:
        print('{:s} core {:12.8f}'.format(sp,p['gamma_'+sp]),file=glib)  
    print('#',file=glib)
    print('# Bond parameters',file=glib)
    print('#',file=glib)

    if not nn:
       first = True
       for bd in bonds:
           b = bd.split('-')
           if p['corr13_'+bd]>0.001 and p['ovcorr_'+bd]>0.001:
              if first: print('reaxff2_bo over bo13',file=glib)
           elif p['corr13_'+bd]>0.001 and p['ovcorr_'+bd]<0.001:
              if first: print('reaxff2_bo bo13',file=glib)
           elif p['corr13_'+bd]<0.001 and p['ovcorr_'+bd]>0.001:
              if first: print('reaxff2_bo over',file=glib)
           elif p['corr13_'+bd]<0.001 and p['ovcorr_'+bd]<0.001:
              if first: print('reaxff2_bo',file=glib)
           print('{:s} core {:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(b[0],b[1],
                  p['bo1_'+bd],p['bo2_'+bd],p['bo3_'+bd],p['bo4_'+bd],p['bo5_'+bd],p['bo6_'+bd]),
                  file=glib) 
           first = False
    else:
       first = True
       for bd in bonds:
           b = bd.split('-')  
           if first:
              #if p['ovcorr_'+bd]>=0.0001:
              #   print('reaxff2_bo over bo13',file=glib) 
              #else: 
              print('reaxff2_bo bo13',file=glib)  
           print('{:s} core {:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(b[0],b[1],
                  p['bo1_'+bd],p['bo2_'+bd],p['bo3_'+bd],p['bo4_'+bd],p['bo5_'+bd],p['bo6_'+bd]),
                  file=glib) 
           first = False

    print('reaxff2_bond kcal',file=glib)
    for bd in bonds:
        b = bd.split('-')
        print('{:s} core {:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(b[0],b[1],
              p['Desi_'+bd],p['Depi_'+bd],p['Depp_'+bd],p['be1_'+bd],p['be2_'+bd]),file=glib) 

    print('reaxff2_over',file=glib)
    for bd in bonds:
        b = bd.split('-')
        print('{:s} core {:s} core {:12.8f} '.format(b[0],b[1],p['ovun1_'+bd]),file=glib) 

    print('reaxff2_morse kcal',file=glib)
    for bd in offd:
        b = bd.split('-')
        print('{:s} core {:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(b[0],b[1],
              p['Devdw_'+bd],p['alfa_'+bd],p['rvdw_'+bd],
              p['rosi_'+bd],p['ropi_'+bd],p['ropp_'+bd]),file=glib) 

    print('#',file=glib)
    print('# Angle parameters',file=glib)
    print('#',file=glib)

    print('reaxff3_angle kcal',file=glib)
    for ag in angs:
        a = ag.split('-')
        print('{:s} core {:s} core {:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(a[1],a[0],
              a[2],p['theta0_'+ag],p['val1_'+ag],p['val2_'+ag],
              p['val4_'+ag],p['val7_'+ag]),file=glib) 

    print('reaxff3_penalty kcal ',file=glib)
    for ag in angs:
        a = ag.split('-')
        # if p['pen1_'+ag]>=0.0001:
        print('{:s} core {:s} core {:s} core {:12.8f}'.format(a[1],a[0],a[2],p['pen1_'+ag]),file=glib) 

    print('reaxff3_conjugation kcal',file=glib)
    for ag in angs:
        a = ag.split('-')
        if abs(p['coa1_'+ag])>=0.0001:
           print('{:s} core {:s} core {:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(a[1],a[0],
                  a[2],p['coa1_'+ag],p['coa2'],p['coa3'],
                  p['coa4']),file=glib) 

    print('#',file=glib)
    print('# Hydrogen bond parameters ',file=glib)
    print('#',file=glib)

    print('reaxff3_hbond kcal',file=glib)
    for hb in hbs:
        h = hb.split('-')
        print('{:s} core {:s} core {:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(h[1],h[0],
                  h[2],p['rohb_'+hb],p['Dehb_'+hb],p['hb1_'+hb],
                  p['hb2_'+hb]),file=glib) 

    print('#',file=glib)
    print('# Torsion parameters ',file=glib)
    print('#',file=glib)
    print('reaxff4_torsion kcal ',file=glib)
    for ag in tors:
        a = ag.split('-')
        print('{:s} core {:s} core {:s} core {:s} core {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}'.format(a[0],
               a[1],a[2],a[3],p['V1_'+ag],p['V2_'+ag],p['V3_'+ag],
               p['tor1_'+ag],p['cot1_'+ag]),file=glib) 

    if m is not None:
       print('#',file=glib)
       print('# Nerual network weight and bias for Bond-Order',file=glib)
       print('#',file=glib)
       print('mflayer {:d} {:d}'.format(mf_layer[0],mf_layer[1]),file=glib)
       print(' ',file=glib)
       print('reaxff1_fnn wi',file=glib)
       shap = len(m['fmwo_'+spec[0]][0])
       nin  = 3
       nout = shap
       for sp in spec:
           print('{:2s} core'.format(sp),end=' ',file=glib) 
           for i in range(nin):
               if i!=0:
                  print('       ',end=' ',file=glib)
               for j in range(mf_layer[0]):
                   print('{:20.16f}'.format(m['fmwi_'+sp][i][j]),end=' ',file=glib)
               print(' ',file=glib)

       print('reaxff1_fnn bi',file=glib)
       for sp in spec:
           print('{:2s} core'.format(sp),end=' ',file=glib) 
           for j in range(mf_layer[0]):
               print('{:20.16f}'.format(m['fmbi_'+sp][j]),end=' ',file=glib)
           print(' ',file=glib)

       print('reaxff1_fnn wo',file=glib)
       for sp in spec:
           print('{:2s} core'.format(sp),end=' ',file=glib) 
           for i in range(mf_layer[0]):
               if i!=0 and nout>1:
                  print('       ',end=' ',file=glib)
               for j in range(nout):
                   print('{:20.16f}'.format(m['fmwo_'+sp][i][j]),end=' ',file=glib)
               if nout>1:
                  print(' ',file=glib)
           if nout==1:
              print(' ',file=glib) 

       print('reaxff1_fnn bo',file=glib)
       for sp in spec:
           print('{:2s} core'.format(sp),end=' ',file=glib) 
           for j in range(nout):
               print('{:20.16f}'.format(m['fmbo_'+sp][j]),end=' ',file=glib)
           print(' ',file=glib)

       if mf_layer[1]>0:
          print('reaxff1_fnn wh',file=glib)
          for sp in spec:
              print('{:2s} core'.format(sp),end=' ',file=glib) 
              for l in range(mf_layer[1]):
                  for i in range(mf_layer[0]):
                      if i!=0:
                         print('       ',end=' ',file=glib)
                      for j in range(mf_layer[0]):
                          print('{:20.16f}'.format(m['fmw_'+sp][l][i][j]),end=' ',file=glib)
                      print(' ',file=glib)

       if mf_layer[1]>0:
          print('reaxff1_fnn bh',file=glib)
          for sp in spec:
              print('{:2s} core'.format(sp),end=' ',file=glib) 
              for l in range(mf_layer[1]):
                  if l!=0:
                     print('       ',end=' ',file=glib)
                  for j in range(mf_layer[0]):
                      print('{:20.16f}'.format(m['fmb_'+sp][l][j]),end=' ',file=glib)
                  print(' ',file=glib)

       print('#',file=glib)
       print('# Nerual network weight and bias for Bond-Energy',file=glib)
       print('#',file=glib)
       print('belayer {:d} {:d}'.format(be_layer[0],be_layer[1]),file=glib)
       print(' ',file=glib)
       print('reaxff2_enn wi',file=glib)
       nin  = 3
       nout = 1
       for bd in bonds:
           b = bd.split('-') 
           print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
           for i in range(nin):
               if i!=0:
                  print('               ',end=' ',file=glib)
               for j in range(be_layer[0]):
                   print('{:20.16f}'.format(m['fewi_'+bd][i][j]),end=' ',file=glib)
               print(' ',file=glib)

       print('reaxff2_enn bi',file=glib)
       for bd in bonds:
           b = bd.split('-') 
           print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
           for j in range(be_layer[0]):
               print('{:20.16f}'.format(m['febi_'+bd][j]),end=' ',file=glib)
           print(' ',file=glib)

       print('reaxff2_enn wo',file=glib)
       for bd in bonds:
           b = bd.split('-') 
           print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
           for i in range(be_layer[0]):
               if i!=0 and nout>1:
                  print('               ',end=' ',file=glib)
               for j in range(nout):
                   print('{:20.16f}'.format(m['fewo_'+bd][i][j]),end=' ',file=glib)
               if nout>1:
                  print(' ',file=glib)
           if nout==1:
              print(' ',file=glib) 

       print('reaxff2_enn bo',file=glib)
       for bd in bonds:
           b = bd.split('-') 
           print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
           for j in range(nout):
               print('{:20.16f}'.format(m['febo_'+bd][j]),end=' ',file=glib)
           print(' ',file=glib)

       if be_layer[1]>0:
          print('reaxff2_enn wh',file=glib)
          for bd in bonds:
              b = bd.split('-') 
              print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
              for l in range(be_layer[1]):
                  for i in range(be_layer[0]):
                      if i!=0:
                         print('               ',end=' ',file=glib)
                      for j in range(be_layer[0]):
                          print('{:20.16f}'.format(m['few_'+bd][l][i][j]),end=' ',file=glib)
                      print(' ',file=glib)

       if be_layer[1]>0:
          print('reaxff2_enn bh',file=glib)
          for bd in bonds:
              b = bd.split('-') 
              print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
              for l in range(be_layer[1]):
                  if l!=0:
                     print('                ',end=' ',file=glib)
                  for j in range(be_layer[0]):
                      print('{:20.16f}'.format(m['feb_'+bd][l][j]),end=' ',file=glib)
                  print(' ',file=glib)

       if vdw_layer:
          print('#',file=glib)
          print('# Nerual network weight and bias for Vdw Tapre Function',file=glib)
          print('#',file=glib)
          print('vwlayer {:d} {:d}'.format(vdw_layer[0],vdw_layer[1]),file=glib)
          print(' ',file=glib)
          print('reaxff2_vnn wi',file=glib)
          nin  = 1
          nout = 1
          for bd in bonds:
              b = bd.split('-') 
              print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
              for i in range(nin):
                  if i!=0:
                      print('               ',end=' ',file=glib)
                  for j in range(vdw_layer[0]):
                      print('{:20.16f}'.format(m['fvwi_'+bd][i][j]),end=' ',file=glib)
                  print(' ',file=glib)

          print('reaxff2_vnn bi',file=glib)
          for bd in bonds:
              b = bd.split('-') 
              print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
              for j in range(vdw_layer[0]):
                  print('{:20.16f}'.format(m['fvbi_'+bd][j]),end=' ',file=glib)
              print(' ',file=glib)

          print('reaxff2_vnn wo',file=glib)
          for bd in bonds:
              b = bd.split('-') 
              print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
              for i in range(vdw_layer[0]):
                  if i!=0 and nout>1:
                      print('               ',end=' ',file=glib)
                  for j in range(nout):
                      print('{:20.16f}'.format(m['fvwo_'+bd][i][j]),end=' ',file=glib)
                  if nout>1:
                      print(' ',file=glib)
              if nout==1:
                  print(' ',file=glib) 

          print('reaxff2_vnn bo',file=glib)
          for bd in bonds:
              b = bd.split('-') 
              print('{:2s} core {:2s} core'.format(b[0],b[1]),end=' ',file=glib) 
              for j in range(nout):
                  print('{:20.16f}'.format(m['fvbo_'+bd][j]),end=' ',file=glib)
              print(' ',file=glib)

    glib.close()
    
