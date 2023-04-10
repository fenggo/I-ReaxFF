#!/usr/bin/env python
from __future__ import print_function
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
from ase.io import read,write
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff import IRFF
from irff.irff_np import IRFF_NP
from irff.md.gulp import get_gulp_energy
# from irff.reax import ReaxFF
from irff.mpnn import MPNN


def e(traj='gulp.traj',efunc=1,fm=3,nn=True):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    batch_size= len(images)
    e = []
    eb,el,eo,eu = [],[],[],[]
    ea,ep,etc = [],[],[]
    et,ef    = [],[]
    ev,eh,ec = [],[],[]

    ir = IRFF(atoms=images[0],
              libfile=ffield,
              nn=nn,
              autograd=False)

    for i,atoms in enumerate(images):
        ir.get_potential_energy(atoms)
        e.append(ir.E)
        eb.append(ir.Ebond.numpy())
        el.append(ir.Elone.numpy())
        eo.append(ir.Eover.numpy())
        eu.append(ir.Eunder.numpy())
        # print(ir.Eang)
        try:
           ea.append(ir.Eang.numpy())
        except:
           ea.append(ir.Eang)
        try:
           ep.append(ir.Epen.numpy())
        except:
           ep.append(ir.Epen)
        try:
           et.append(ir.Etor.numpy())
        except:
           et.append(ir.Etor)
        try:
           ef.append(ir.Efcon.numpy())
        except:
           ef.append(ir.Efcon)
        try:
           etc.append(ir.Etcon.numpy())
        except:
           etc.append(ir.Etcon)
        ev.append(ir.Evdw.numpy())
        try:
           eh.append(ir.Ehb.numpy())
        except:
           eh.append(ir.Ehb)
        ec.append(ir.Ecoul.numpy())

    rn = MPNN(libfile=ffield,
              dataset={'tmp':traj},
              opt=[],optword='nocoul',
              batch_size=batch_size,
              clip_op=False,
              InitCheck=False,
              nn=nn,
              mf_layer=[9,1],
              be_layer=[9,1],
              EnergyFunction=efunc,
              MessageFunction=fm) 
    rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    mol  = 'tmp'
    e_   = rn.get_value(rn.E[mol])
    eb_  = rn.get_value(rn.ebond[mol])
    el_  = rn.get_value(rn.elone[mol])
    eu_  = rn.get_value(rn.eunder[mol])
    eo_  = rn.get_value(rn.eover[mol])
    ea_  = rn.get_value(rn.eang[mol])
    ep_  = rn.get_value(rn.epen[mol])
    etc_ = rn.get_value(rn.tconj[mol])
    et_  = rn.get_value(rn.etor[mol])
    ef_  = rn.get_value(rn.efcon[mol])
    ev_  = rn.get_value(rn.evdw[mol])
    ec_  = rn.get_value(rn.ecoul[mol])
    eh_  = rn.get_value(rn.ehb[mol])

    e_irff = {'Energy':e,'Ebond':eb,'Eunger':eu,'Eover':eo,'Eang':ea,'Epen':ep,
              'Elone':el,
              'Etcon':etc,'Etor':et,'Efcon':ef,'Evdw':ev,'Ecoul':ec,'Ehbond':eh}
    e_reax = {'Energy':e_,'Ebond':eb_,'Eunger':eu_,'Eover':eo_,'Eang':ea_,'Epen':ep_,
              'Elone':el_,
              'Etcon':etc_,'Etor':et_,'Efcon':ef_,'Evdw':ev_,'Ecoul':ec_,'Ehbond':eh_}

    for key in e_reax:
        plt.figure()   
        plt.ylabel('%s (eV)' %key)
        plt.xlabel('Step')

        plt.plot(e_reax[key],alpha=0.6,
                 linestyle='-.',marker='o',markerfacecolor='none',
                 markeredgewidth=1,markeredgecolor='b',markersize=4,
                 color='blue',label='ReaxFF')
        plt.plot(e_irff[key],alpha=0.6,
                 linestyle=':',marker='^',markerfacecolor='none',
                 markeredgewidth=1,markeredgecolor='red',markersize=4,
                 color='red',label='IRFF')
        plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
        plt.savefig('%s.pdf' %key) 
        plt.close() 

def ep(traj='opt.traj',efun=3,batch_size=50,nn=True):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    e = []
    eb,el,eo,eu = [],[],[],[]
    ea,ep,etc = [],[],[]
    et,ef    = [],[]
    ev,eh,ec = [],[],[]

    ir = IRFF_NP(atoms=images[0],
              libfile=ffield,
              nn=nn)

    for i,atoms in enumerate(images):
        ir.calculate(atoms)
        e.append(ir.E)
        eb.append(ir.Ebond)
        el.append(ir.Elone)
        eo.append(ir.Eover)
        eu.append(ir.Eunder)
        ea.append(ir.Eang)
        ep.append(ir.Epen)
        et.append(ir.Etor)
        ef.append(ir.Efcon)
        etc.append(ir.Etcon)
        ev.append(ir.Evdw)
        eh.append(ir.Ehb)
        ec.append(ir.Ecoul)
    
    rn = MPNN(libfile=ffield,
                direcs={'tmp':traj},
                dft='siesta',
                opt=[],optword='nocoul',
                batch_size=batch_size,
                atomic=True,
                clip_op=False,
                InitCheck=False,
                nn=nn,
                bo_layer=[4,1],
                mf_layer=[9,3],
                be_layer=[6,1],
                EnergyFunction=efun,
                pkl=False) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    mol  = 'tmp'
    e_   = rn.get_value(rn.E[mol])
    eb_  = rn.get_value(rn.ebond[mol])
    el_  = rn.get_value(rn.elone[mol])
    eu_  = rn.get_value(rn.eunder[mol])
    eo_  = rn.get_value(rn.eover[mol])
    ea_  = rn.get_value(rn.eang[mol])
    ep_  = rn.get_value(rn.epen[mol])
    etc_ = rn.get_value(rn.tconj[mol])
    et_  = rn.get_value(rn.etor[mol])
    ef_  = rn.get_value(rn.efcon[mol])
    ev_  = rn.get_value(rn.evdw[mol])
    ec_  = rn.get_value(rn.ecoul[mol])
    eh_  = rn.get_value(rn.ehb[mol])

    e_reax = {'Energy':e,'Ebond':eb,'Eunger':eu,'Eover':eo,'Eang':ea,'Epen':ep,
              'Elone':el,
              'Etcon':etc,'Etor':et,'Efcon':ef,'Evdw':ev,'Ecoul':ec,'Ehbond':eh}
    e_irff = {'Energy':e_,'Ebond':eb_,'Eunger':eu_,'Eover':eo_,'Eang':ea_,'Epen':ep_,
              'Elone':el_,
              'Etcon':etc_,'Etor':et_,'Efcon':ef_,'Evdw':ev_,'Ecoul':ec_,'Ehbond':eh_}

    for key in e_reax:
        plt.figure()   
        plt.ylabel('%s (eV)' %key)
        plt.xlabel('Step')

        plt.plot(e_reax[key],alpha=0.01,
                 linestyle='-.',marker='o',markerfacecolor='none',
                 markeredgewidth=1,markeredgecolor='b',markersize=4,
                 color='blue',label='ReaxFF')
        plt.plot(e_irff[key],alpha=0.01,
                 linestyle=':',marker='^',markerfacecolor='none',
                 markeredgewidth=1,markeredgecolor='red',markersize=4,
                 color='red',label='IRFF')
        plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
        plt.savefig('%s.eps' %key) 
        plt.close() 

def db(traj='gulp.traj',batch_size=50,ef=1,fm=3,nn=True,frame=0):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    atoms  = images[frame]

    e,ei      = [],[]
    ir = IRFF(atoms=atoms,
              libfile=ffield,
              nn=nn,
              autograd=False)
    ir.calculate(atoms)

    ei.append(ir.Ebond)
    ebond = ir.ebond # .numpy()

    rn = MPNN(libfile=ffield,
              dataset={'tmp':traj},
              dft='siesta',
              opt=[],optword='nocoul',
              batch_size=batch_size,
              clip_op=False,
              InitCheck=False,
              nn=nn,
              mf_layer=[9,1],
              be_layer=[9,1],
              EnergyFunction=ef,
              MessageFunction=fm) 

    rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    mol  = 'tmp'
    er   = rn.get_value(rn.ebond[mol]) 
    blab = rn.lk.bdlab

    bosi   = ir.bosi.numpy()
    bopsi  = ir.bop_si.numpy()
    boppi  = ir.bop_pi.numpy()
    boppp  = ir.bop_pp.numpy()

    bopow3 = ir.bopow3.numpy()
    eterm3 = ir.eterm3.numpy()

    bop    = ir.bop.numpy()
    bo     = ir.bo0.numpy()

    eterm1 = ir.eterm1.numpy()
    bopow1 = ir.bopow1.numpy()
    bodiv1 = ir.bodiv1.numpy()

    r      = ir.r.numpy()
    D_     = rn.get_value(rn.Deltap)
    D      = ir.Deltap.numpy()

    ebd_ = np.zeros([ir.natom,ir.natom])

    for bd in rn.bonds:
        for nb in range(rn.nbd[bd]):
            ebd      = rn.get_value(rn.EBD[bd])
            mol_,i,j = blab[bd][nb]
            bosi_    = rn.get_value(rn.bosi[bd])
            bopsi_   = rn.get_value(rn.bop_si[bd]) 
            boppi_   = rn.get_value(rn.bop_pi[bd]) 
            boppp_   = rn.get_value(rn.bop_pp[bd]) 

            bopow3_  = rn.get_value(rn.bopow3[bd])
            eterm3_  = rn.get_value(rn.eterm3[bd])

            bop_     = rn.get_value(rn.bop[bd]) 
            bo_      = rn.get_value(rn.bo0[bd]) 

            eterm1_  = rn.get_value(rn.eterm1[bd]) 
            bopow1_  = rn.get_value(rn.bopow1[bd]) 
            bodiv1_  = rn.get_value(rn.bodiv1[bd]) 

            rbd      = rn.get_value(rn.rbd[bd])

            ebd_[i][j] = ebd[nb][frame]
            ebd_[j][i] = ebd[nb][frame]
            #　if abs(bo_[nb][0]-bo[i][j])>0.00001:
            print('-  %s %2d %2d:' %(bd,i,j),
                      'rbd %10.7f %10.7f' %(rbd[nb][frame],r[i][j]),
                      'bop %10.7f %10.7f' %(bop_[nb][frame],bop[i][j]),
                      'Di %10.7f %10.7f' %(D_[i][frame],D[i]),
                      'Dj %10.7f %10.7f' %(D_[j][frame],D[j]),
                      'bo %10.7f %10.7f' %(bo_[nb][frame],bo[i][j]),
                      'ebond %10.7f %10.7f' %(ebd[nb][frame],ebond[i][j])  )

    rcbo = rn.get_value(rn.rc_bo)
    eb   = rn.get_value(rn.ebond[mol])
    print('\n-  bond energy:',ir.Ebond.numpy(),eb[frame],end='\n')
  
def dl(traj='siesta.traj',batch_size=1,nn=False,frame=0):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    atoms  = images[frame]
    his    = TrajectoryWriter('tmp.traj',mode='w')
    his.write(atoms=atoms)
    his.close()

    ir = IRFF(atoms=atoms,
          libfile=ffield,
          nn=False,
          bo_layer=[8,4])


    ir.get_potential_energy(atoms)
    el    = ir.elone.numpy()
    Dle   = ir.Delta_e.numpy()
    Dlp   = ir.Delta_lp.numpy()
    de    = ir.DE.numpy()
    nlp   = ir.nlp.numpy()
    elp   = ir.explp.numpy()
    eu    = ir.eunder.numpy()
    eu1   = ir.eu1.numpy()
    eu2   = ir.eu2.numpy()
    eu3   = ir.expeu3.numpy()

    rn = ReaxFF(libfile=ffield,
                direcs={'tmp':'tmp.traj'},
                dft='siesta',
                opt=[],optword='nocoul',
                batch_size=batch_size,
                atomic=True,
                clip_op=False,
                InitCheck=False,
                nn=nn,
                pkl=False,
                to_train=False) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    mol  = 'tmp'
    el_  = rn.get_value(rn.EL) 
    Dle_ = rn.get_value(rn.Delta_e) 
    Dlp_ = rn.get_value(rn.Delta_lp) 
    de_  = rn.get_value(rn.DE) 
    nlp_ = rn.get_value(rn.nlp) 
    elp_ = rn.get_value(rn.explp) 

    eu_  = rn.get_value(rn.EUN) 
    eu1_ = rn.get_value(rn.eu1) 
    eu2_ = rn.get_value(rn.eu2) 
    eu3_ = rn.get_value(rn.expeu3) 

    alab = rn.lk.atlab
    bosi = ir.bosi.numpy()

    for i in range(ir.natom):
        a  = ir.atom_name[i] 
        al = [mol,i]
        na = alab[a].index(al)
        print('-  %d %s:' %(i,ir.atom_name[i]),
                 'elone: %10.8f  %10.8f' %(el_[a][na],el[i]),
              #  'Delta_e: %10.8f  %10.8f' %(Dle_[a][na],Dle[i]),
              # 'Delta_lp: %10.8f  %10.8f' %(Dlp_[a][na],Dlp[i]),
              #      'nlp: %10.8f  %10.8f' %(nlp_[a][na],nlp[i]),
                'eunder: %10.8f  %10.8f' %(eu_[a][na],eu[i]),
              #    'eu1: %10.8f  %10.8f' %(eu1_[a][na],eu1[i]),
                   'eu2: %10.8f  %10.8f' %(eu2_[a][na],eu2[i]),
                   'eu3: %10.8f  %10.8f' %(eu3_[a][na],eu3[i])   )
    print('\n-  under energy:',ir.Eunder.numpy(),rn.eunder[mol].numpy()[0],end='\n')


def da(traj='opt.traj',batch_size=50,ef=3,nn=True,frame=44):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    atoms  = images[frame]
    his    = TrajectoryWriter('tmp.traj',mode='w')
    his.write(atoms=atoms)
    # his.write(atoms=images[frame+1])
    his.close()

    ir = IRFF(atoms=atoms,
             libfile=ffield,
             nn=nn,
             autograd=False)
    ir.calculate(atoms)

    rn = MPNN(libfile=ffield,
              direcs={'tmp':'tmp.traj'},
              dft='siesta',
              opt=[],optword='nocoul',
              batch_size=batch_size,
              atomic=True,
              clip_op=False,
              InitCheck=False,
              nn=nn,
              bo_layer=[9,2],
              EnergyFunction=ef,
              pkl=False) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    mol    = 'tmp'
    anglab = rn.lk.anglab
    angs   = []
    for ang in ir.angs:
        angs.append(list(ang))

    ea_     = rn.get_value(rn.EANG) 
    ea      = ir.eang.numpy()
    f7_     = rn.get_value(rn.f_7) 
    f7      = ir.f_7.numpy()
    f8_     = rn.get_value(rn.f_8) 
    f8      = ir.f_8.numpy()
    expang_ = rn.get_value(rn.expang) 
    expang  = ir.expang.numpy()

    theta_   = rn.get_value(rn.theta) 
    theta    = ir.theta.numpy()
    theta0_  = rn.get_value(rn.theta0) 
    theta0   = ir.thet0.numpy()

    sbo3_  = rn.get_value(rn.SBO3) 
    sbo3   = ir.SBO3.numpy()
    
    fa = open('ang.txt','w')
    for ang in rn.angs:
        for a_ in range(rn.nang[ang]):
            a = anglab[ang][a_][1:]

            if a not in angs:
               a.reverse()
            i,j,k = a
            if a in angs:
               ai = angs.index(a)
               print(' * %2d%s-%2d%s-%2d%s:' %(i,ir.atom_name[i],j,ir.atom_name[j],k,ir.atom_name[k]),
                      #'eang: %10.8f  %10.8f' %(ea_[ang][a_][0],ea[ai]),
                        'f7: %10.8f  %10.8f' %(f7_[ang][a_][0],f7[ai]),
                        'f8: %10.8f  %10.8f' %(f8_[ang][a_][0],f8[ai]),
                    'expang: %10.8f  %10.8f' %(expang_[ang][a_][0],expang[ai]),
                    'theta: %10.8f  %10.8f' %(theta_[ang][a_][0],theta[ai]), 
                     # 'sbo3: %10.8f  %10.8f' %(sbo3_[ang][a_][0],sbo3[ai]),
                    'theta0: %10.8f  %10.8f' %(theta0_[ang][a_][0],theta0[ai]), file=fa)
            else:
               print(' * %2d%s-%2d%s-%2d%s:' %(i,ir.atom_name[i],j,ir.atom_name[j],k,ir.atom_name[k]),
                      #'eang: %10.8f' %(ea_[ang][a_][0]),
                        'f7: %10.8f' %(f7_[ang][a_][0]),
                        'f8: %10.8f' %(f8_[ang][a_][0]),
                    'expang: %10.8f' %(expang_[ang][a_][0]),
                    'expang: %10.8f' %(expang_[ang][a_][0]),
                     'theta: %10.8f' %(theta_[ang][a_][0]), 
                      # 'sbo3: %10.8f' %(sbo3_[ang][a_][0]),
                    'theta0: %10.8f' %(theta0_[ang][a_][0]))
    fa.close()
    epen = rn.get_value(rn.epen) 
    eang = rn.get_value(rn.eang) 
    print(' * penalty energy:',ir.Epen.numpy(),epen[mol][0])
    print(' * angel energy:',ir.Eang.numpy(),eang[mol][0])


def dt(traj='siesta.traj',batch_size=1,nn=True,frame=0):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    atoms  = images[frame]
    his    = TrajectoryWriter('tmp.traj',mode='w')
    his.write(atoms=atoms)
    his.close()

    from irff.irff import IRFF
    ir = IRFF(atoms=atoms,
              libfile=ffield,
              nn=nn)
    ir.get_potential_energy(atoms)
    eb     = ir.Ebond.numpy()

    et     = ir.etor.numpy()
    ef     = ir.efcon.numpy()
    f10    = ir.f_10.numpy()
    f11    = ir.f_11.numpy()

    sijk   = ir.s_ijk.numpy()
    sjkl   = ir.s_jkl.numpy()

    f      = ir.fijkl.numpy()
    v1     = ir.v1.numpy()
    v2     = ir.v2.numpy()
    v3     = ir.v3.numpy()

    expv2  = ir.expv2.numpy()

    boij   = ir.botij.numpy()
    bojk   = ir.botjk.numpy()
    bokl   = ir.botkl.numpy()

    cosw   = ir.cos_w.numpy()
    cos2w  = ir.cos2w.numpy()
    w      = ir.w.numpy()

    Etor   = ir.Etor.numpy()
    del IRFF


    from irff.reax import ReaxFF
    rn = ReaxFF(libfile=ffield,
                direcs={'tmp':'tmp.traj'},
                dft='siesta',
                opt=[],optword='nocoul',
                batch_size=batch_size,
                atomic=True,
                clip_op=False,
                InitCheck=False,
                nn=nn,
                bo_layer=[9,2],
                pkl=False,
                to_train=False) 
    molecules = rn.initialize()
    # rn.calculate(rn.p,rn.m) 
    rn.session(learning_rate=3.0-4,method='AdamOptimizer')

    mol    = 'tmp'
    torlab = rn.lk.torlab
    tors   = []
    for tor in ir.tors:
        tors.append(list(tor))
        print(tor)

    eb_    = rn.get_value(rn.ebond[mol])
    et_    = rn.get_value(rn.ETOR) 
    ef_    = rn.get_value(rn.Efcon) 
    f10_   = rn.get_value(rn.f_10) 
    f11_   = rn.get_value(rn.f_11) 

    sijk_  = rn.get_value(rn.s_ijk) 
    sjkl_  = rn.get_value(rn.s_jkl) 

    f_     = rn.get_value(rn.fijkl) 
    
    boij_  = rn.get_value(rn.BOtij) 
    bojk_  = rn.get_value(rn.BOtjk)
    bokl_  = rn.get_value(rn.BOtkl)

    v1_    = rn.get_value(rn.v1) 
    v2_    = rn.get_value(rn.v2) 
    v3_    = rn.get_value(rn.v3) 

    expv2_ = rn.get_value(rn.expv2) 

    cosw_  = rn.get_value(rn.cos_w) 
    cos2w_ = rn.get_value(rn.cos2w) 
    w_     = rn.get_value(rn.w) 

    for tor in rn.tors:
        for a_ in range(rn.ntor[tor]):
            a = torlab[tor][a_][1:]

            if a not in tors:
               a.reverse()
            i,j,k,l = a
            if a in tors:
               ai = tors.index(a)
               # if abs(et_[tor][a_][0]-et[ai])>0.0001:
               print('-  %2d%s-%2d%s-%2d%s-%2d%s:' %(i,ir.atom_name[i],j,ir.atom_name[j],
                                                     k,ir.atom_name[k],l,ir.atom_name[l]),
                      'etor: %10.8f  %10.8f' %(et_[tor][a_][0],et[ai]), 
                     'sijk: %10.8f  %10.8f' %(sijk_[tor][a_][0],sijk[ai]), 
                     'sjkl: %10.8f  %10.8f' %(sjkl_[tor][a_][0],sjkl[ai]), 
                    'boij: %10.8f  %10.8f' %(boij_[tor][a_][0],boij[ai]), 
                    'bojk: %10.8f  %10.8f' %(bojk_[tor][a_][0],bojk[ai]), 
                    'bokl: %10.8f  %10.8f' %(bokl_[tor][a_][0],bokl[ai]), 
                    'fijkl: %10.8f  %10.8f' %(f_[tor][a_][0],f[ai]), 
                        'v1: %10.8f  %10.8f' %(v1_[tor][a_][0],v1[ai]), 
                        'v2: %10.8f  %10.8f' %(v2_[tor][a_][0],v2[ai]), 
                        'v3: %10.8f  %10.8f' %(v3_[tor][a_][0],v3[ai]), 
                     'expv2: %10.8f  %10.8f' %(expv2_[tor][a_][0],expv2[ai]), 
                     'ptor1: %10.8f  %10.8f' %(rn.p_['tor1_'+tor],ir.P['tor1'][ai]), 
                    # 'cosw: %10.8f  %10.8f' %(cosw_[tor][a_][0],cosw[ai]), 
                    # 'cos2w: %10.8f  %10.8f' %(cos2w_[tor][a_][0],cos2w[ai]), 
                    #  'v1: %10.8f  %10.8f' %(0.5*rn.p_['V1_'+tor]*(1.0+cosw_[tor][a_][0]),
                    #                         0.5*ir.P['V1'][ai]*(1.0+cosw[ai])), 
                    #  'w: %10.8f  %10.8f' %(w_[tor][a_][0],w[ai]), 
                    # 'efcon: %10.8f  %10.8f' %(ef_[tor][a_][0],ef[ai]),
                    #  'f_10: %10.8f  %10.8f' %(f10_[tor][a_][0],f10[ai]),
                    #  'f_11: %10.8f  %10.8f' %(f11_[tor][a_][0],f11[ai]),
                     )
    Etor_ = rn.get_value(rn.etor)
    print('\n-  torsion energy:',Etor,Etor_[mol][0],end='\n')
    print('\n-  Bond energy:',eb,eb_,end='\n')
    del ReaxFF
    
      
def dh(traj='siesta.traj',batch_size=1,nn=True,frame=7):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    atoms  = images[frame]
    his    = TrajectoryWriter('tmp.traj',mode='w')
    his.write(atoms=atoms)
    his.close()

    from irff.irff import IRFF
    ir = IRFF(atoms=atoms,
              libfile=ffield,
              nn=nn,
              bo_layer=[9,2])
    ir.get_potential_energy(atoms)
    
    from irff.reax import ReaxFF
    rn = ReaxFF(libfile=ffield,
                direcs={'tmp':'tmp.traj'},
                dft='siesta',
                opt=[],optword='nocoul',
                batch_size=batch_size,
                atomic=True,
                clip_op=False,
                InitCheck=False,
                nn=nn,
                pkl=False,
                to_train=False) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    mol    = 'tmp'
    hblab  = rn.lk.hblab
    hbs    = []

    for hb in ir.hbs:
        hbs.append(list(hb))

    eh_    = rn.get_value(rn.EHB) 
    # eh     = ir.ehb.numpy()
    rhb_   = rn.get_value(rn.rhb) 
    # rhb    = ir.rhb.numpy()

    frhb_  = rn.get_value(rn.frhb) 
    # frhb   = ir.frhb.numpy()

    sin4_  = rn.get_value(rn.sin4) 
    # sin4   = ir.sin4.numpy()

    # exphb1_= rn.get_value(rn.exphb1) 
    # exphb2_= rn.get_value(rn.exphb2) 

    # exphb1 = ir.exphb1.numpy()
    # exphb2 = ir.exphb2.numpy()

    # hbsum_ = rn.get_value(rn.hbsum) 
    # hbsum  = ir.hbsum.numpy()

    for hb in rn.hbs:
        for a_ in range(rn.nhb[hb]):
            a = hblab[hb][a_][1:]
            i,j,k = a

            if a in hbs:
               ai = hbs.index(a)
               # if eh_[hb][a_][0]<-0.000001:
               print('-  %2d%s-%2d%s-%2d%s:' %(i,ir.atom_name[i],j,ir.atom_name[j],k,ir.atom_name[k]),
                   'ehb: %10.8f' %(eh_[hb][a_][0]), 
                   'rhb: %10.8f' %(rhb_[hb][a_][0]), 
                  'frhb: %10.8f' %(frhb_[hb][a_][0]), 
                  'sin4: %10.8f' %(sin4_[hb][a_][0]) )
            # else:
            #    if eh_[hb][a_][0]<-0.00001:
            #       print('-  %2d%s-%2d%s-%2d%s:' %(i,ir.atom_name[i],j,ir.atom_name[j],k,ir.atom_name[k]),
            #          'ehb: %10.8f' %(eh_[hb][a_][0]),
            #        'rhb: %10.8f' %(rhb_[hb][a_][0]), 
            #       'frhb: %10.8f' %(frhb_[hb][a_][0]), 
            #       'sin4: %10.8f' %(sin4_[hb][a_][0]) )
    
                  # for c,e in enumerate(ehb):
                  #     if e<-0.000001:
                  #       i_,j_ = self.hbij[c]
                  #       j_,k_ = self.hbjk[c]
                  #       print('-  %2d%s-%2d%s-%2d%s:' %(i_,self.atom_name[i_],j_,
                  #                      self.atom_name[j_],k_,self.atom_name[k_]),
                  #          'ehb: %10.8f' %e, 
                  #          'rhb: %10.8f' %(rjk[c]), 
                  #          'frhb: %10.8f' %(frhb[c]), 
                  #          'sin4: %10.8f' %(sin4[c]) )

                  
def dr(traj='poscar.gen',batch_size=1,nn=False):
    ffield = 'ffield.json' if nn else 'ffield'
    atoms  = read(traj)

    e,ei      = [],[]
    ir = IRFF(atoms=atoms,
          libfile=ffield,
          nn=False,
          bo_layer=[8,4])

    ir.get_potential_energy(atoms)

def eg(traj='md.traj',nn=True):
    ffield = 'ffield.json'  
    images = Trajectory(traj)
    batch_size= len(images)
    e = []
    eb,el,eo,eu = [],[],[],[]
    ea,ep,etc = [],[],[]
    et,ef    = [],[]
    ev,eh,ec = [],[],[]

    for i,atoms in enumerate(images):
        ge = get_gulp_energy(atoms,ffield='reaxff_nn')
        e.append(ge['E'])
        eb.append(ge['ebond'])
        el.append(ge['elonepair'])
        eo.append(ge['eover'])
        eu.append(ge['eunder'])
        ea.append(ge['eangle'])
        ep.append(ge['epenalty'])
        et.append(ge['etorsion'])
        ef.append(ge['fconj'])
        etc.append(ge['econjugation'])
        ev.append(ge['evdw'])
        eh.append(ge['ehb'])
        ec.append(ge['ecoul'])
      
    rn = MPNN(libfile=ffield,
              dataset={'tmp':traj},
              dft='siesta',
              opt=[],optword='nocoul',
              batch_size=batch_size,
              clip_op=False,
              InitCheck=False,
              nn=nn,
              #bo_layer=[4,1],
              mf_layer=[9,1],
              be_layer=[9,0],
              EnergyFunction=1,
              MessageFunction=3) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    mol  = 'tmp'
    e_   = rn.get_value(rn.E[mol])
    eb_  = rn.get_value(rn.ebond[mol])
    el_  = rn.get_value(rn.elone[mol])
    eu_  = rn.get_value(rn.eunder[mol])
    eo_  = rn.get_value(rn.eover[mol])
    ea_  = rn.get_value(rn.eang[mol])
    ep_  = rn.get_value(rn.epen[mol])
    etc_ = rn.get_value(rn.tconj[mol])
    et_  = rn.get_value(rn.etor[mol])
    ef_  = rn.get_value(rn.efcon[mol])
    ev_  = rn.get_value(rn.evdw[mol])
    ec_  = rn.get_value(rn.ecoul[mol])
    eh_  = rn.get_value(rn.ehb[mol])

    Dbi  = rn.get_value(rn.Dbi)
    Dbj  = rn.get_value(rn.Dbj)
    B    = rn.get_value(rn.H[0])

    for bd in Dbi:
        print('\n', bd,'\n')
        f = 0
        for i,_ in enumerate(B[bd]):
            print('{:2d} {:8.5f} {:8.5f} {:8.5f}'.format(i,
                     Dbi[bd][f][i],B[bd][f][i],Dbj[bd][f][i]))
    
    e_gulp = {'Energy':e,'Ebond':eb,'Eunger':eu,'Eover':eo,'Eang':ea,'Epen':ep,
              'Elone':el,
              'Etcon':etc,'Etor':et,'Efcon':ef,'Evdw':ev,'Ecoul':ec,'Ehbond':eh}
    e_reax = {'Energy':e_,'Ebond':eb_,'Eunger':eu_,'Eover':eo_,'Eang':ea_,'Epen':ep_,
              'Elone':el_,
              'Etcon':etc_,'Etor':et_,'Efcon':ef_,'Evdw':ev_,'Ecoul':ec_,'Ehbond':eh_}

    for key in e_reax:
        plt.figure()   
        plt.ylabel('%s (eV)' %key)
        plt.xlabel('Step')

        plt.plot(e_reax[key],alpha=0.6,
                 linestyle='-.',marker='o',markerfacecolor='none',
                 markeredgewidth=1,markeredgecolor='b',markersize=4,
                 color='blue',label='ReaxFF')
        plt.plot(e_gulp[key],alpha=0.6,
                 linestyle=':',marker='^',markerfacecolor='none',
                 markeredgewidth=1,markeredgecolor='red',markersize=4,
                 color='red',label='GULP')
        plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
        plt.savefig('%s.pdf' %key) 
        plt.close() 


def d(traj='md.traj',nn=True):
    ffield = 'ffield.json'  
    images = Trajectory(traj)
    batch_size= len(images)
      
    rn = MPNN(libfile=ffield,
              dataset={'tmp':traj},
              dft='siesta',
              opt=[],optword='nocoul',
              batch_size=batch_size,
              clip_op=False,
              InitCheck=False,
              nn=nn,
              #bo_layer=[4,1],
              mf_layer=[9,1],
              be_layer=[9,1],
              EnergyFunction=1,
              MessageFunction=3) 
    rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    Dbi  = rn.get_value(rn.Dbi)
    Dbj  = rn.get_value(rn.Dbj)
    B    = rn.get_value(rn.H[0])
    Eb   = rn.get_value(rn.EBD) 
    R    = rn.get_value(rn.rbd) 

    for bd in Dbi:
        for f in range(batch_size):
            print('\n', bd,'\n')
            # f = 0
            for i,_ in enumerate(B[bd]):
                print('{:2d} R: {:8.5f} Di: {:8.5f} B: {:8.5f} Dj: {:8.5f} Ebd: {:8.5f}'.format(i,
                        R[bd][i][f],Dbi[bd][i][f],B[bd][i][f],Dbj[bd][i][f],Eb[bd][i][f]))

if __name__ == '__main__':
   ''' use commond like ./irffdeb.py d --t=data/*.traj to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [e,eg,ep,db,dl,da,dt,dh,dr,d])
   argh.dispatch(parser)  

   
