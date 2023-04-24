from os import system
import numpy as np
import csv
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units
from ase.visualize import view
from ..irff_np import IRFF_NP
from ..AtomDance import AtomDance
from ..md.gulp import write_gulp_in,get_reax_energy
import matplotlib.pyplot as plt


def plot(e,Eb,Eu,Eo,El,Ea,Et,Ep,Etor,Ef,Ev,Ehb,Ec,figsize=(15,12),r=None,show=False):
    total = 4
    sEu = np.sum(np.square(Eu))
    sEo = np.sum(np.square(Eo))
    sEl = np.sum(np.square(El))
    sEa = np.sum(np.square(Ea))
    sEt = np.sum(np.square(Et))
    sEp = np.sum(np.square(Ep))
    sEtor = np.sum(np.square(Etor))
    sEf = np.sum(np.square(Ef))
    sEhb = np.sum(np.square(Ehb))
    sEc = np.sum(np.square(Ec))

    for e_ in [sEu,sEo,sEl,sEa,sEt,sEp,sEtor,sEf,sEhb,sEc]:
        if e_>0.001:
           total += 1
    
    if total<=6:
       nrow = 3
       ncol = 2
    elif total<=9:
       nrow = 3
       ncol = 3
    elif total<=12:
       nrow = 4
       ncol = 3  
    else:
       nrow = 4
       ncol = 4  
    Evb = np.array(Ev) + np.array(Eb)
    Energies = [Eb,Ev,Evb,Eu,Eo,El,Ea,Et,Ep,Etor,Ef,Ehb,Ec,e]
    labels   = ['Ebond','Evdw','Ebond & Evdw','Eunder','Eover',
                'Elone','Eangle','Ethree','Epenialty','Etorsion',
                'Efour','Ehbond','Ecoulomb','Etotal']
    Es       = [1,1,1,sEu,sEo,sEl,sEa,sEt,sEp,sEtor,sEf,sEhb,sEc,1]

    plt.figure(figsize=figsize) 
    nf = 1
    for es,e_,label in zip(Es,Energies,labels):
        if es>0.001:
           plt.subplot(nrow,ncol,nf)   
           if r is None:
              plt.plot(e_,alpha=0.8,linestyle='-',color='b',label=label)
           else:
              plt.plot(r,e_,alpha=0.8,linestyle='-',color='b',label=label)
           plt.legend(loc='best',edgecolor='yellowgreen')
           nf += 1
    plt.show() if show else plt.savefig('deb_energies.pdf')
    plt.close()

def deb_vdw(images,i=0,j=1,show=False):
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nn=True)
    ir.calculate_Delta(images[0])

    Eb,Ea,e = [],[],[]
    Ehb,Eo,Ev,Eu,El = [],[],[],[],[]
    Etor,Ef,Ep,Et = [],[],[],[]

    for i_,atoms in enumerate(images):       
        ir.calculate(images[i_])
        # print('%d Energies: ' %i_,'%12.4f ' %ir.E, 'Ebd: %8.4f' %ir.ebond[0][1],'Ebd: %8.4f' %ir.ebond[2][3] )
        Eb.append(ir.Ebond)
        Ea.append(ir.Eang)
        Eo.append(ir.Eover)
        Ev.append(ir.Evdw)
        Eu.append(ir.Eunder)
        El.append(ir.Elone)
        Ep.append(ir.Epen)
        Et.append(ir.Etcon)
        Ef.append(ir.Efcon)
        Etor.append(ir.Etor)
        Ehb.append(ir.Ehb)
        e.append(ir.E)
        
    emin_ = np.min(Eb)
    eb    =  np.array(Eb) - emin_# )/emin_
    vmin_ =  np.min(Ev)
    ev    =  np.array(EV) - vmin_# )/emin_

    plt.figure()     
    # plt.plot(bopsi,alpha=0.8,linewidth=2,linestyle=':',color='k',label=r'$BO_p^{\sigma}$')
    # plt.plot(boppi,alpha=0.8,linewidth=2,linestyle='-.',color='k',label=r'$BO_p^{\pi}$')
    # plt.plot(boppp,alpha=0.8,linewidth=2,linestyle='--',color='k',label=r'$BO_p^{\pi\pi}$')
    # plt.plot(bo0,alpha=0.8,linewidth=2,linestyle='-',color='g',label=r'$BO^{t=0}$')
    
    plt.plot(ev,alpha=0.8,linewidth=2,linestyle='-',color='y',label=r'$E_{vdw}$')
    plt.plot(eb,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$E_{bond}$')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.show() if show else plt.savefig('deb_bo.pdf')
       
    plt.close()
    return eb,ev


def deb_energy(images,atomi=0,atomj=1,r_is_x=False,savecsv=False,show=False,nn=True,libfile='ffield.json',figsize=None):
    ir = IRFF_NP(atoms=images[0],
                 libfile=libfile,
                 nn=nn)
    ir.calculate_Delta(images[0])

    Eb,Ea,Ec,e = [],[],[],[]
    Ehb,Eo,Ev,Eu,El = [],[],[],[],[]
    Etor,Ef,Ep,Et = [],[],[],[]
    R=[]
    
    if savecsv:
       fcsv = open('energies.csv','w')
       csv_write = csv.writer(fcsv)
       csv_write.writerow(['r','etotal','ebond','elone','eover','eunder','eangle',
                            'econj','epen','etor',
                            'efcon','evdw','ecoul','ehb'])

    for i_,atoms in enumerate(images):       
        ir.calculate(images[i_])
        # print('%d Energies: ' %i_,'%12.4f ' %ir.E, 'Ebd: %8.4f' %ir.ebond[0][1],'Ebd: %8.4f' %ir.ebond[2][3] )
        r  = ir.r[atomi][atomj]
        R.append(r)
        Eb.append(ir.Ebond)
        Ea.append(ir.Eang)
        Eo.append(ir.Eover)
        Ev.append(ir.Evdw)
        Eu.append(ir.Eunder)
        El.append(ir.Elone)
        Ep.append(ir.Epen)
        Et.append(ir.Etcon)
        Ef.append(ir.Efcon)
        Etor.append(ir.Etor)
        Ehb.append(ir.Ehb)
        Ecoul = ir.Ecoul if ir.Ecoul>0.00000001 else 0.0
        Ec.append(Ecoul)
        e.append(ir.E)
        if savecsv: csv_write.writerow([r,ir.E,ir.Ebond,ir.Elone,ir.Eover,ir.Eunder,ir.Eang, ir.Etcon,ir.Epen,ir.Etor,
                            ir.Efcon,ir.Evdw,ir.Ecoul,ir.Ehb])
    if savecsv: fcsv.close()
    if r_is_x:
       plot(e,Eb,Eu,Eo,El,Ea,Et,Ep,Etor,Ef,Ev,Ehb,Ec,r=R,show=show,figsize=figsize)
    else:
       plot(e,Eb,Eu,Eo,El,Ea,Et,Ep,Etor,Ef,Ev,Ehb,Ec,r=None,show=show,figsize=figsize)
    return e

def deb_bp(atoms,figsize=(8,6),libfile='ffield.json',nn=True):    
    ir = IRFF_NP(atoms=atoms,
                 libfile=libfile,
                 nn=nn)
    ir.calculate_Delta(atoms)
    # print(ir.rcbo)
    for i in range(ir.natom-1):
        for j in range(i+1,ir.natom): 
            if ir.bo0[i][j]<0.1:
               continue 
            ir.calculate(atoms)
            r     = ir.r[i][j]
            bsi   = ir.bop_si[i][j]
            bpi   = ir.bop_pi[i][j]
            bpp   = ir.bop_pp[i][j]
            Bp    = ir.bop[i][j]      
            B     = ir.bo0[i][j]  
            Di    = ir.Deltap[i]-ir.bop[i][j]
            Dj    = ir.Deltap[j]-ir.bop[i][j]
 
            print('{:3d}{:2s}-{:3d}{:2s} r: {:5.3f} Bsi\': {:6.4f} Bpi\': {:6.4f} Bpp\': {:6.4f} '
                  'Di: {:5.3f} B\': {:6.4f} Dj: {:5.6f} '
                  'B: {:6.4f} '.format(i,ir.atom_name[i],j,ir.atom_name[j],r,
                  bsi,bpi,bpp,Di,Bp,Dj,B))   

def deb_bo(images,i=0,j=1,figsize=(8,6),print_=False,show=False,more=False,
           fluctuation=False,delta=False,u=1.5,l=1.35,libfile='ffield.json',
           nn=True,debug=False,bo_p=0,x_distance=False):
    r,bopsi,boppi,boppp,bo0,bo1,eb,esi = [],[],[],[],[],[],[],[]
    Di,Dj = [],[]
   #  fcsv_ = open('bo.csv','w')
   #  csv_write = csv.writer(fcsv_)
   #  csv_write.writerow(['r','bosi0','bopi0','bopp0','bosi1','bopi1','bopp1',
   #                      'bosi2','bopi2','bopp2',
   #                      'bo0','bo1','esi','eb'])
    
    ir = IRFF_NP(atoms=images[0],
                 libfile=libfile,
                 nn=nn)
    ir.calculate_Delta(images[0])
    print('-  rcut bo:',ir.rcbo[0][1])

    for i_,atoms in enumerate(images):       
        ir.calculate_Delta(atoms)
        if bo_p==0:
           bopsi.append(ir.eterm1[i][j])
           boppi.append(ir.eterm2[i][j])
           boppp.append(ir.eterm3[i][j])
        elif bo_p==1:
           bopsi.append(ir.bop_si[i][j])
           boppi.append(ir.bop_pi[i][j])
           boppp.append(ir.bop_pp[i][j])
        else:
           bopsi.append(ir.bosi[i][j])
           boppi.append(ir.bopi[i][j])
           boppp.append(ir.bopp[i][j])

        bo0.append(ir.bop[i][j])      
        bo1.append(ir.bo0[i][j])  
        eb.append(ir.ebond[i][j])   
        esi.append(ir.esi[i][j])
        Di.append(ir.Deltap[i]-ir.bop[i][j])
        Dj.append(ir.Deltap[j]-ir.bop[i][j])
        # N.append(ir.N[i][j])
      #   csv_write.writerow([ir.r[i][j],ir.eterm1[i][j],ir.eterm2[i][j],ir.eterm3[i][j],
      #                       ir.bop_si[i][j],ir.bop_pi[i][j],ir.bop_pp[i][j],
      #                       ir.bosi[i][j],ir.bopi[i][j],ir.bopp[i][j],
      #                       ir.bop[i][j],ir.bo0[i][j],ir.esi[i][j],ir.ebond[i][j]])

        if x_distance:
           r.append(ir.r[i][j]) 
        else:
           r.append(i_)
        if print_:
           if debug:
              print('\n r: ',r[-1],'\n')
              print(' BO_uncorrected: \n')
              for bo in ir.bop:
                  for b in bo :
                      print('{:6.4f} '.format(b),end='')
                  print('')
                  
              print('\n Ni: \n')
              for ni in ir.Ni:
                  for n in ni :
                      print('{:6.4f} '.format(n),end='')
                  print('')
              print('\n Nj: \n')
              for ni in ir.Nj:
                  for n in ni :
                      print('{:6.4f} '.format(n),end='')
                  print('')
              print('\n S: \n')
              for ni in ir.S:
                  for n in ni :
                      print('{:6.4f} '.format(n),end='')
                  print('')
           elif not nn:
              print('r: {:6.4f} bosi: {:6.4f} bopi: {:6.4f} bopp: {:6.4f} '
                'bo0: {:6.4f} F:{:6.4f} F11:{:6.4f} F45:{:6.4f} '
                'bo1: {:6.4f} '
                'e: {:6.4f}'.format(r[-1],
                bopsi[-1],boppi[-1],boppp[-1],bo0[-1],
                ir.F[i][j],ir.F11[i][j],ir.F45[i][j],
                bo1[-1],
                ir.ebond[i][j])) 
           elif fluctuation:
              # S: {:2.1f} ir.S[i][j],
              print('r: {:6.4f} bosi: {:6.4f} bopi: {:6.4f} bopp: {:6.4f} '
                'esi: {:6.4f} l: {:6.4f} u: {:6.4f} esi\': {:6.4f} e: {:6.4f}'.format(r[-1],
                bopsi[-1],boppi[-1],boppp[-1],
                ir.esi[i][j],bopsi[-1]*l,bopsi[-1]*u,
                ir.esi[i][j]*bopsi[0]/esi[0],ir.ebond[i][j])) 
           elif delta:
              print('{:3d}  r: {:6.4f} Di: {:6.4f} Dj: {:6.4f} '
                    'bo0: {:6.4f} bo1: {:6.4f} '
                    'e: {:6.6f}'.format(i_,
                    r[-1],Di[-1],Dj[-1],bo0[-1],bo1[-1],ir.ebond[i][j])) 
           else:  
              print('{:3d}  r: {:6.4f} bosi: {:6.4f} bopi: {:6.4f} bopp: {:6.4f} '
                   'bo0: {:6.4f} bo1: {:6.4f} '
                   'bosi: {:6.4f} bopi: {:6.4f} bopp: {:6.4f} e: {:6.6f}'.format(i_,
               r[-1],bopsi[-1],boppi[-1],boppp[-1],bo0[-1],bo1[-1],
               ir.bosi[i][j],ir.bopi[i][j],ir.bopp[i][j],ir.ebond[i][j]))  
 
    emin_ = np.min(eb)
    eb = (emin_ - np.array(eb) )/emin_

    ems = np.min(esi)
    emx = np.max(esi)
    esi = (np.array(esi)-ems)/emx

    plt.figure(figsize=figsize)     
    plt.plot(r,bopsi,alpha=0.8,linewidth=2,linestyle=':',color='k',label=r'$BO_p^{\sigma}$')
    plt.plot(r,boppi,alpha=0.8,linewidth=2,linestyle='-.',color='k',label=r'$BO_p^{\pi}$')
    plt.plot(r,boppp,alpha=0.8,linewidth=2,linestyle='--',color='k',label=r'$BO_p^{\pi\pi}$')
    plt.plot(r,bo0,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$BO^{t=0}$')
    plt.plot(r,bo1,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$BO^{t=1}$')
    if more:
       plt.plot(r,eb,alpha=0.8,linewidth=2,linestyle='-',color='indigo',label=r'$E_{bond}$ ($-E_{bond}/%4.2f$)' %-emin_)
       # plt.plot(r,esi,alpha=0.8,linewidth=2,linestyle='-',color='indigo',label=r'$E_{esi}$ ($E_{si}/%4.2f$)' %emx)
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.show() if show else plt.savefig('deb_bo.pdf')
    plt.close()
    # return r,eb
    #fcsv_.close()

def deb_f(images,i=0,j=1,figsize=(8,6),print_=False,show=False,
           nn=True,x_distance=False):
    r,bopsi,boppi,boppp,bo0,bo1,eb,esi = [],[],[],[],[],[],[],[]
    
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nn=nn)
    ir.calculate_Delta(images[0])
    # print(ir.rcbo)
    for i_,atoms in enumerate(images):       
        ir.calculate(atoms)

        bo0.append(ir.bop[i][j])      
        bo1.append(ir.bo0[i][j])  
        # N.append(ir.N[i][j])

        if x_distance:
           r.append(ir.r[i][j]) 
        else:
           r.append(i_)
        if print_:
           print('r: {:6.4f} bo0: {:6.4f} Di: {:6.4f} Dj: {:6.4f} '
                 'F:{:6.4f} F11:{:6.4f}  F2:{:6.4f} F3:{:6.4f} F45:{:6.4f} '
                 'bo1: {:6.4f} '.format(r[-1],bo0[-1],ir.Deltap[i],ir.Deltap[j],
                 ir.F[i][j],ir.F11[i][j],ir.f_2[i][j],ir.f_3[i][j],ir.F45[i][j],
                 bo1[-1])) 

    plt.figure(figsize=figsize)     
    plt.plot(r,bopsi,alpha=0.8,linewidth=2,linestyle=':',color='k',label=r'$BO_p^{\sigma}$')
    plt.plot(r,boppi,alpha=0.8,linewidth=2,linestyle='-.',color='k',label=r'$BO_p^{\pi}$')
    plt.plot(r,boppp,alpha=0.8,linewidth=2,linestyle='--',color='k',label=r'$BO_p^{\pi\pi}$')
    plt.plot(r,bo0,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$BO^{t=0}$')
    plt.plot(r,bo1,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$BO^{t=1}$')
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('deb_bo.pdf')
    if show: plt.show()
    plt.close()
    # return r,eb

def deb_delta(atoms,libfile='ffield.json',nn=True):    
    ir = IRFF_NP(atoms=atoms,
                 libfile=libfile,
                 nn=nn)
    #ir.calculate_Delta(atoms)
    ir.calculate(atoms)

    for i in range(ir.natom):
      print('ID: {:4d} Delta\': {:6.4f}'.format(i,ir.Deltap[i]))   

def deb_eang(images,ang=[0,1,2],figsize=(8,6),show=False,print_=False,frame=[0]):
    i,j,k = ang
    a     = 0
    
    eang,ecoa,epen = [],[],[]
    
    ir = IRFF_NP(atoms=images[frame],
                 libfile='ffield.json',
                 nn=True)
    for f_ in frame:
        ir.calculate(images[f_])
      
        if print_:
           print('\n All angle: \n')
           for a,angle in enumerate(ir.angs): 
               print('{:3d} {:6.4f}  {:6.4f} i: {:3d} j: {:3d} k: {:3d} eang: {:6.4f}'.format(a,
                        ir.thet0[a],ir.theta[a],ir.angi[a],ir.angj[a],ir.angk[a],ir.eang[a]))
    if print_: print('\n Angle specified: \n')

    for i_,atoms in enumerate(images):  
        found = False 
        for na,angle in enumerate(ir.angs):  
            i_,j_,k_ = angle
            if (i_==i and j_==j and k_==k) or (i_==k and j_==j and k_==i):
               a = na
               found = True
        if not found:
           # print('Warning: no angle found for {:s} in this trajector frame!'.format(ang))   
           continue 
        ir.calculate(atoms)

        eang.append(ir.Eang)    
        ecoa.append(ir.Etcon)
        epen.append(ir.Epen)
        if print_:
           print('{:3d}  {:6.4f}  {:6.4f} Dpi: {:6.4f} pbo: {:6.4f} N: {:6.4f} SBO3: {:6.4f} eang: {:6.4f}'.format(i_,
                     ir.thet0[a],ir.theta[a],ir.sbo[a],ir.pbo[a],
                     ir.nlp[j],ir.SBO3[a],ir.eang[a])) # self.thet0-self.theta
         
    plt.figure(figsize=figsize)     
    plt.plot(eang,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$E_{ang}$')# ($-E_{ang}/{:4.2f}$)'.format(ang_m))
    # plt.plot(ecoa,alpha=0.8,linewidth=2,linestyle='-',color='indigo',label=r'$E_{coa}$') # ($E_{coa}/%4.2f$)' %emx)
    # plt.plot(epen,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$E_{pen}$') # ($E_{pen}/%4.2f$)' %eox)
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('deb_ang.pdf')
    if show: plt.show()
    plt.close()


def get_theta(atoms,figsize=(8,6)):
    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 nn=True)
    ir.calculate(atoms)
     
    for a,angle in enumerate(ir.angs):  
        i,j,k = angle
        print('{:3d} {:3d} {:3d} {:3d}  {:6.4f}  {:6.4f} Dpi: {:6.4f} SBO: {:6.4f} pbo: {:6.4f} SBO3: {:6.4f}'.format(a,
                     i,j,k,
                     ir.thet0[a],ir.theta[a],ir.sbo[a],ir.SBO[a],ir.pbo[a],
                     ir.SBO3[a])) # self.thet0-self.theta


def deb_eover(images,i=0,j=1,figsize=(16,10),show=False,print_=True):
    bopsi,boppi,boppp,bo0,bo1,eb = [],[],[],[],[],[]
    eo,eu,el,esi,r = [],[],[],[],[]
    eo_,eu_,eb_ = [],[],[]
    
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nn=True)
    ir.calculate_Delta(images[0])
    
    for i_,atoms in enumerate(images):       
        ir.calculate(atoms)      
        bo0.append(ir.bop[i][j])      
        bo1.append(ir.bo0[i][j])  
        eb.append(ir.ebond[i][j])    
        eo.append(ir.Eover)      
        eu.append(ir.Eunder)
        r.append(ir.r[i][j])
        
        if print_:
           print('r: {:6.4f} bo: {:6.4f} eu: {:6.4f} ev: {:6.4f} eb: {:6.4f}'.format(ir.r[i][j],
                 ir.bo0[i][j],ir.Eunder,ir.Eover,ir.ebond[i][j]))
    
    ebx  = np.max(np.abs(eb))
    eb   = np.array(eb)/ebx+1.0
    
    eox  = np.max(np.abs(eo))
    if eox<0.0001:
       eox = 1.0
    eo   = np.array(eo)/eox + 1.0
    
    eux  = np.max(np.abs(eu))
    eu   = np.array(eu)/eux + 1.0
    
    plt.figure(figsize=figsize)     
    # plt.plot(r,bo0,alpha=0.8,linewidth=2,linestyle='-',color='g',label=r'$BO^{t=0}$')
    # plt.plot(r,bo1,alpha=0.8,linewidth=2,linestyle='-',color='y',label=r'$BO^{t=1}$')
    plt.plot(r,eb,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$E_{bond}$')
    plt.plot(r,eo,alpha=0.8,linewidth=2,linestyle='-',color='indigo',label=r'$E_{over}$ ($E_{over}/%4.2f$)' %eox)
    plt.plot(r,eu,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$E_{under}$ ($E_{under}/%4.2f$)' %eux)
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('deb_bo.pdf')
    if show: plt.show()
    plt.close()


def deb_gulp_energy(images,atomi=0,atomj=1,libfile='reax',show=False):
    ''' test reax with GULP, and run validation-set'''
    fcsv = open('gulp.csv','w')
    csv_write = csv.writer(fcsv)
    csv_write.writerow(['r','etotal','ebond','elone','eover','eunder','eangle',
                        'econj','epen','etor',
                        'efcon','evdw','ecoul','ehb'])

    GE={}
    GE['ebond'],GE['elonepair'],GE['eover'],GE['eunder'],GE['eangle'], \
    GE['econjugation'],GE['evdw'],GE['Total-Energy'] = \
             [],[],[],[],[],[],[],[]
    GE['epenalty'],GE['etorsion'],GE['fconj'],GE['ecoul'],GE['eself'],GE['ehb']\
           = [],[],[],[],[],[] 

    for atoms in images: 
        # A = Atoms(symbols=molecules[mol].atom_name,
        #           positions=molecules[mol].x[nf],
        #           cell=cell,
        #           pbc=(1, 1, 1))
        # A = atoms irff_jax(A)
        write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',
                      lib=libfile)
        system('gulp<inp-gulp>out')

        e_,eb_,el_,eo_,eu_,ea_,ep_,etc_,et_,ef_,ev_,ehb_,ecl_,esl_= \
            get_reax_energy(fo='out')
        positions = atoms.get_positions()
        vr = positions[atomi] - positions[atomj]
        r  = np.sqrt(np.sum(vr*vr,axis=0))

        csv_write.writerow([r,e_,eb_,el_,eo_,eu_,ea_, etc_,ep_,et_,
                            ef_,ev_,ecl_,ehb_])

        GE['ebond'].append(eb_)
        GE['elonepair'].append(el_)
        GE['eover'].append(eo_)
        GE['eunder'].append(eu_)
        GE['eangle'].append(ea_)
        GE['econjugation'].append(etc_)
        GE['epenalty'].append(ep_)
        GE['etorsion'].append(et_)
        GE['fconj'].append(ef_)
        GE['evdw'].append(ev_)
        GE['ecoul'].append(ecl_)
        GE['ehb'].append(ehb_)
        GE['eself'].append(esl_)
        GE['Total-Energy'].append(e_)

    GE['Total-Energy'] = np.array(GE['Total-Energy']) 

    plot(GE['Total-Energy'],GE['ebond'],GE['eunder'],GE['eunder'],GE['elonepair'],
         GE['eangle'],GE['econjugation'],GE['epenalty'],GE['etorsion'],GE['fconj'],
         GE['evdw'],GE['ehb'],GE['ecoul'],show=show)
    fcsv.close()
    return GE['Total-Energy']
 

def compare_dft_energy(images,atomi=None,atomj=None,show=False,nn=True):
    ir = IRFF_NP(atoms=images[0],libfile='ffield.json',nn=nn)
    R,e_dft,e = [],[],[]
    if atomi is None or atomj is None:
       r_is_x=False
    else:
       r_is_x=True

    for i_,atoms in enumerate(images):  
        e_dft.append(atoms.get_potential_energy())
        ir.calculate(images[i_])
        e.append(ir.E)
        # print(e_dft[-1],e[-1])
        if r_is_x:
           r  = ir.r[atomi][atomj]
           R.append(r)

    e_min = np.min(e_dft)
    e_dft = np.array(e_dft) - e_min
    e_min = np.min(e)
    e     = np.array(e) - e_min

    plt.figure()     
    
    if r_is_x:
       plt.plot(R,e,alpha=0.8,linewidth=2,linestyle='-.',color='b',label=r'$E_{ReaxFF-nn}$')
       plt.plot(R,e_dft,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$E_{DFT}$')
    else:
       plt.plot(e,alpha=0.8,linewidth=2,linestyle='-.',color='b',label=r'$E_{ReaxFF-nn}$')
       plt.plot(e_dft,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$E_{DFT}$')

    plt.legend(loc='best',edgecolor='yellowgreen')
    # plt.savefig('compare_dft.pdf')
    if show: plt.show()
    plt.close()
    return e

def get_theta(atoms,figsize=(8,6),show=False,print_=False):
    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 nn=True)
    ir.calculate(atoms)
     
    for a,angle in enumerate(ir.angs):  
        i,j,k = angle
        print('{:3d} {:3d} {:3d} {:3d}  {:6.4f}  {:6.4f} Dpi: {:6.4f} SBO: {:6.4f} pbo: {:6.4f} SBO3: {:6.4f}'.format(a,
                     i,j,k,
                     ir.thet0[a],ir.theta[a],ir.sbo[a],ir.SBO[a],ir.pbo[a],
                     ir.SBO3[a])) # self.thet0-self.theta
                     
def deb_thet(images,ang=[0,1,2],figsize=(8,6),show=False,print_=False,dFrame=[]):
    i,j,k = ang
    ang_  = [k,j,i]
    a     = 0
    found = False
    eang,ecoa,epen = [],[],[]
    e,d,s = [],[],[]
    
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nn=True)
    ir.calculate_Delta(images[0])
     
    for i_,atoms in enumerate(images):       
        ir.calculate(atoms)
        
        for na,angle in enumerate(ir.angs):  
            ii,j_,k_ = angle
            # print(angle)
            if (ii==i and j_==j and k_==k) or (ii==k and j_==j and k_==i):
               a = na
               found = True
               # print('find: ',a)
        if not found:
           print('Error: no angle found for',ang,angle)
        eang.append(ir.Eang)    
        ecoa.append(ir.Etcon)
        epen.append(ir.Epen)
        if i_ in dFrame:
           e.append(ir.eang)
           d.append(ir.dang)
           s.append(ir.sbo)
           # print(ir.Eang)
        # if print_:
           # for a,angle in enumerate(ir.angs): 
           # if i_ in dFrame:
        print('{:3d}  {:6.4f}  {:6.4f} Dpi: {:6.4f} SBO: {:6.4f} SBO3: {:6.4f} f7: {:6.4f} f8: {:6.4f} thet: {:6.4f} Eang: {:6.4f}'.format(i_,
                     ir.thet0[a],ir.theta[a],ir.Dpi[j],ir.SBO[a],ir.SBO3[a],
                     # ir.nlp[j],ir.SBO3[a],
                     ir.f_7[a],ir.f_8[a], 
                     ir.thet[a], 
                     ir.eang[a])) # self.thet0-self.theta

    if print_:
       for a_,angle in enumerate(ir.angs):
           print('id: {:3d}  {:3d}  {:3d}  {:3d}'.format(a_,
                  angle[0],angle[1],angle[2]),end=' ')
           print('Eang:',end=' ')
           for e_ in e:
               print(' {:6.4f} '.format(e_[a_]),end=' ')
           print('Delta:',end=' ')
           for e_ in d:
               print('{:6.4f} '.format(e_[a_]),end=' ')
           print('sbo:',end=' ')
           for e_ in s:
               print('{:6.4f} '.format(e_[a_]),end=' ')
           print(' ')

    plt.figure(figsize=figsize)     
    plt.plot(eang,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$E_{ang}$')# ($-E_{ang}/{:4.2f}$)'.format(ang_m))
    # plt.plot(ecoa,alpha=0.8,linewidth=2,linestyle='-',color='indigo',label=r'$E_{coa}$') # ($E_{coa}/%4.2f$)' %emx)
    # plt.plot(epen,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$E_{pen}$') # ($E_{pen}/%4.2f$)' %eox)
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('deb_ang.pdf')
    if show: plt.show()
    plt.close()

def get_changed_fourboday(traj='md.traj'):
    images_md = Trajectory('md.traj')
    Eb,Ea,Etor,E = [],[],[],[]
    ir = IRFF_NP(atoms=images_md[0],
                 libfile='ffield.json',
                 nn='True')
    ir.calculate(images_md[0])

    tor_min = {}
    tor_max = {}
    for i,tor in enumerate(ir.tors):
        t = tuple(tor)
        tor_min[t] = ir.etor[i]
        tor_max[t] = ir.etor[i]
    # print(tor_min)

    for i_,atoms in enumerate(images_md):       
        ir.calculate(images_md[i_])
        # print('%d Energies: ' %i_,'%12.4f ' %ir.E )

        Eb.append(ir.Ebond)
        Ea.append(ir.Eang)
        Etor.append(ir.Etor)
        E.append(ir.E)

        for i,tor in enumerate(ir.tors):
            t  = tuple(tor)
            t_ = (tor[3],tor[2],tor[1],tor[0])
            if t in tor_min:
               if tor_min[t]>ir.etor[i]:
                  tor_min[t] = ir.etor[i]
               if tor_max[t]<ir.etor[i]:
                  tor_max[t] = ir.etor[i]
            elif t_ in tor_min:
               if tor_min[t_]>ir.etor[i]:
                  tor_min[t_] = ir.etor[i]
               if tor_max[t_]<ir.etor[i]:
                  tor_max[t_] = ir.etor[i]
            else:
              tor_min[t] = ir.etor[i]
              tor_max[t] = ir.etor[i]
    tor_change = []
    tor_name   = []
    m_ = 0.0
    mi = 0
    for i,t in enumerate(tor_min):
        m = tor_max[t] - tor_min[t]
        if m>m_:
           mi = i
        tor_change.append(m)
        tor_name.append(t)
    tor  = tor_name[mi]
    tor_ = '{:s}-{:s}-{:s}-{:s}'.format(ir.atom_name[tor[0]],ir.atom_name[tor[1]],
                                        ir.atom_name[tor[2]],ir.atom_name[tor[3]])
    print(tor_,tor,tor_change[mi])
    
## compare the total energy with DFT energy

# images = Trajectory('md.traj')
# E = []
# for atoms in images:
#     E.append(atoms.get_potential_energy())

# e = deb_energy(images)

# plt.figure()
# e_ = np.array(e) - np.min(e)
# E_ = np.array(E) - np.min(E)
# plt.plot(e_,alpha=0.8,linestyle='-',color='b',label='Total Energy')
# plt.plot(E_,alpha=0.8,linestyle='-',color='r',label='DFT Energy')
# plt.legend(loc='best',edgecolor='yellowgreen')
# plt.show()


