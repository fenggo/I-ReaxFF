from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import system, getcwd, chdir,listdir,environ
from irff.reax import ReaxFF
# from train_reaxff import cnn
import numpy as np
from ase import Atoms
from irff.gulp import write_gulp_in,get_reax_energy

#environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def debug_v(direcs={'ch4':'/home/feng/siesta/train/ch4'},
            gulp_cmd='/home/feng/gulp/Src/gulp<inp-gulp >gulp.out'):
    for key in direcs:
        mol = key
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                optword='nocoul',
                batch_size=1,
                sort=False,
                pkl=True,
                interactive=True)
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    vlab     = rn.lk.vlab
    rv       = rn.get_value(rn.rv)
    expvdw1  = rn.get_value(rn.expvdw1)
    expvdw2  = rn.get_value(rn.expvdw2)
    evdw     = rn.get_value(rn.EVDW)
    f13      = rn.get_value(rn.f_13)

    cell  = rn.cell[mol]
    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))
    write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
    system(gulp_cmd)
    atom_name = molecules[mol].atom_name

    fg = open('gulp.out','r')
    for line in fg.readlines():
        if line.find('- evdw:')>=0:
           l = line.split()
           i = int(l[2])-1
           j = int(l[3])-1

           vb = atom_name[i]+'-'+atom_name[j] 
           vbk= [mol,i,j]
           vbkr= [mol,j,i]
           if not vb in rn.bonds:
              vb = atom_name[j]+'-'+atom_name[i]

           find = False
           if vbk in vlab[vb]:
              # nb = hblab[hb].index(hbk)
              # print('------------------------------------')
              nbs = []
              for nb,bb in enumerate(vlab[vb]):
                  if bb==vbk:
                     nbs.append(nb)
              find = True
           elif vbkr in vlab[vb]:
              # nb = hblab[hb].index(hbk)
              # print('------------------------------------')
              nbs = []
              for nb,bb in enumerate(vlab[vb]):
                  if bb==vbkr:
                     nbs.append(nb)
              find = True

           if find:
              ib = 0
              for nb in nbs:
                  if abs(rv[vb][nb][0]-float(l[4]))<0.00001:
                     # if abs(evdw[vb][nb][0]-float(l[5]))>0.0001:
                     print('- ReaxFF %d %s:' %(ib,vb), 
                              'rv: %10.6f' %rv[vb][nb][0],
                            'evdw: %10.6f' %evdw[vb][nb][0])

                     print('-   GULP %d %s:' %(ib,vb),
                              'rv: %10.6f' %float(l[4]),
                            'evdw: %10.6f' %float(l[5]) )
                     ib += 1
           else:
              print('- N.F GULP %s:' %vb,
                      'rv: %10.6f' %float(l[4]),
                    'evdw: %10.6f' %float(l[5]) )
    fg.close()


def debug_h(direcs={'ch4':'/home/feng/siesta/train/ch4'},
            gulp_cmd='/home/feng/gulp/Src/gulp<inp-gulp >gulp.out'):
    for key in direcs:
        mol = key
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                rc_scale='none',
                optword='all',
                batch_size=1,
                sort=False,
                pkl=True,
                interactive=True)
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    hblab = rn.lk.hblab
    bdlab = rn.lk.bdlab
    rbd   = rn.get_value(rn.rbd)
    rhb   = rn.get_value(rn.rhb)
    rik   = rn.lk.rik
    rij   = rn.lk.rij
    fhb   = rn.get_value(rn.fhb)
    frhb  = rn.get_value(rn.frhb)
    bohb  = rn.get_value(rn.BOhb)
    exphb1= rn.get_value(rn.exphb1)
    exphb2= rn.get_value(rn.exphb2)
    sin4  = rn.get_value(rn.sin4)
    hbthe = rn.get_value(rn.hbthe)
    ehb   = rn.get_value(rn.EHB)

    cell  = rn.cell[mol]
    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))
    write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
    # system('/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out')
    system(gulp_cmd)
    # system('/home/gfeng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out')
    atom_name = molecules[mol].atom_name

    fg = open('gulp.out','r')
    for line in fg.readlines():
        if line.find('- ehb:')>=0:
           l = line.split()
           i = int(l[2])-1
           j = int(l[3])-1
           k = int(l[4])-1

           hb = atom_name[i]+'-'+atom_name[j]+'-'+atom_name[k]
           hbk= [mol,i,j,k]

           bd = atom_name[i]+'-'+atom_name[j]
           if not bd in rn.bonds:
              bd = atom_name[j]+'-'+atom_name[i]
           bdk= [mol,i,j]
           bdkr= [mol,j,i]
           if bdk in bdlab[bd]:
              nbd = bdlab[bd].index(bdk)
           elif bdkr in bdlab[bd]:
              nbd = bdlab[bd].index(bdkr)

           find = False
           if hbk in hblab[hb]:
              # nb = hblab[hb].index(hbk)
              # print('------------------------------------')
              nbs = []
              for nb,bb in enumerate(hblab[hb]):
                  if bb==hbk:
                     nbs.append(nb)
              find = True
           

           if find:
              ib = 0
              for nb in nbs:
                  if abs(rhb[hb][nb][0]-float(l[6]))<0.00001:
                     print('- ReaxFF %d %s:' %(ib,hb), 
                            'rbd: %10.6f' %rbd[bd][nbd][0],
                            'rhb: %10.6f' %rhb[hb][nb][0],
                         'exphb1: %10.6f' %exphb1[hb][nb][0],
                           'bohb: %10.6f' %bohb[hb][nb][0],
                         'exphb2: %10.6f' %exphb2[hb][nb][0],
                           'sin4: %10.6f' %sin4[hb][nb][0],
                          'hbthe: %10.6f' %hbthe[hb][nb][0],
                            'ehb: %10.6f' %ehb[hb][nb][0],
                            'rik: %10.6f' %rik[hb][nb][0])

                     print('-   GULP %d %s:' %(ib,hb),
                            'rbd: %10.6f' %float(l[5]),
                            'rhb: %10.6f' %float(l[6]),
                         'exphb1: %10.6f' %float(l[9]),
                           'bohb: %10.6f' %float(l[13]),
                         'exphb2: %10.6f' %float(l[10]),
                           'sin4: %10.6f' %float(l[11]),
                          'hbthe: %10.6f' %float(l[15]),
                            'ehb: %10.6f' %float(l[12]),
                            'rik: %10.6f' %float(l[14]) )
                     ib += 1
           else:
              print('N.F-GULP %s:' %hb,
                    'rbd: %10.6f' %float(l[5]),
                    'rhb: %10.6f' %float(l[6]),
                    'fhb: %10.6f' %float(l[7]),
                 'exphb1: %10.6f' %float(l[9]),
                   'bohb: %10.6f' %float(l[13]),
                 'exphb2: %10.6f' %float(l[10]),
                   'sin4: %10.6f' %float(l[11]),
                    'ehb: %10.6f' %float(l[12]) )
    fg.close()


def debug_bo(direcs={'ch4':'/home/feng/siesta/train/ch4'},
             gulp_cmd='/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out'):
    for key in direcs:
        mol = key
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                optword='nocoul',
                batch_size=1,
                sort=False,
                pkl=True,
                interactive=True)
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    bdlab = rn.lk.bdlab

    bosi  = rn.get_value(rn.bosi)

    bo    = rn.get_value(rn.bo)
    bo0   = rn.get_value(rn.bo0)

    bop   = rn.get_value(rn.bop)
    rbd   = rn.get_value(rn.rbd)
    bop_si= rn.get_value(rn.bop_si)
    bop_pi= rn.get_value(rn.bop_pi)
    bop_pp= rn.get_value(rn.bop_pp)

    f     = rn.get_value(rn.F)
    f11   = rn.get_value(rn.F_11)
    f12   = rn.get_value(rn.F_12)
    f45   = rn.get_value(rn.F_45)
    f4    = rn.get_value(rn.f_4)
    f5    = rn.get_value(rn.f_5)

    ebond = rn.get_value(rn.ebond)

    cell  = rn.cell[mol]

    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))
    write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
    system(gulp_cmd)
    atom_name = molecules[mol].atom_name

    fg = open('gulp.out','r')
    for line in fg.readlines():
        if line.find('- bosi:')>=0:
           l = line.split()
           i = int(l[2])-1
           j = int(l[3])-1
           bn = atom_name[i]+'-'+atom_name[j]
           if not bn in rn.bonds:
              bn = atom_name[j]+'-'+atom_name[i]
           bnk= [mol,i,j]
           bnkr= [mol,j,i]

           find = False
           if bnk in bdlab[bn]:
              nb = bdlab[bn].index(bnk)
              find = True
           elif bnkr in bdlab[bn]:
              nb = bdlab[bn].index(bnkr)
              find = True

           if find:
              # if abs(rbd[bn][nb][0]-float(l[4]))>0.0001:
              print('- ReaxFF %s:' %bn, 
                          'rbd:',rbd[bn][nb][0],
                       'bop_si:',bop_si[bn][nb][0],
                       'bop_pi:',bop_pi[bn][nb][0],
                       'bop_pp:',bop_pp[bn][nb][0])
              print('-   GULP %s:' %bn,
                           'rbd:',l[4],
                        'bop_si:',l[5],
                        'bop_pi:',l[6],
                        'bop_pp:',l[7])
           else:
              print('-   GULP %s:' %bn,
                           'rbd:',l[4],
                        'bop_si:',l[5],
                        'bop_pi:',l[6],
                        'bop_pp:',l[7])
    fg.close()

    e_,eb_,el_,eo_,eu_,ea_,ep_,etc_,et_,ef_,ev_,ehb_,ecl_,esl_= \
            get_reax_energy(fo='gulp.out')

    print('-  ebond - IRFF %f GULP %f.' %(ebond[mol],eb_))
    

def debug_eo(direcs={'ch4':'/home/feng/siesta/train/ch4'},
             gulp_cmd='/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out'):
    for key in direcs:
        mol = key
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                rc_scale='none',
                clip_op=False,
                optword='all',
                batch_size=1,
                sort=False,
                pkl=True,
                interactive=True)
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    atlab = rn.lk.atlab
    rbd   = rn.get_value(rn.rbd)
    eover = rn.get_value(rn.EOV)
    D_lp  = rn.get_value(rn.Delta_lpcorr)
    otrm1 = rn.get_value(rn.otrm1)
    otrm2 = rn.get_value(rn.otrm2)
    p     = rn.get_value(rn.p)
    so    = rn.get_value(rn.so)
    cell  = rn.cell[mol]
    atom_name = molecules[mol].atom_name

    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))
    write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
    system(gulp_cmd)
    

    fg = open('gulp.out','r')
    for line in fg.readlines():
        if line.find('- eover:')>=0:
           l = line.split()
           i = int(l[2])-1

           an = atom_name[i] 
           ank= [mol,i]
           find = False
           if ank in atlab[an]:
              na = atlab[an].index(ank)
              find = True

           if find:
              # if abs(ebond[bn][nb][0]-float(l[4]))>0.001:
              print('- ReaxFF %d %s:' %(na,an),
                      'eover: %10.6f' %eover[an][na][0],
                       'D_lp: %10.6f' %D_lp[an][na][0],
                      'otrm1: %10.6f' %otrm1[an][na][0],
                      'otrm2: %10.6f' %otrm2[an][na][0],
                        'Val: %10.6f' %p['val_'+an],
                         'SO: %10.6f' %so[an][na][0],
                      'ovun2: %10.6f' %p['ovun2_'+an])
              print('-   GULP %d %s:' %(na,an),
                      'eover: %10.6f' %float(l[3]),
                       'D_lp: %10.6f' %float(l[6]),
                      'otrm1: %10.6f' %float(l[5]),
                      'otrm2: %10.6f' %float(l[7]),
                        'Val: %10.6f' %float(l[8]),
                         'SO: %10.6f' %float(l[4]),
                      'ovun2: %10.6f' %float(l[9]))
           else:
              print('-   GULP %d %s:' %(na,an),
                      'eover: %10.6f' %float(l[3]),
                       'D_lp: %10.6f' %float(l[6]),
                      'otrm1: %10.6f' %float(l[5]),
                      'otrm2: %10.6f' %float(l[7]),
                        'Val: %10.6f' %float(l[8]),
                         'SO: %10.6f' %float(l[4]),'not found ... ... ... ')
    fg.close()


def debug_eu(direcs={'ch4':'/home/feng/siesta/train/ch4'},
             gulp_cmd='/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out'):
    for key in direcs:
        mol = key
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                rc_scale='none',
                clip_op=False,
                optword='all',
                batch_size=1,
                sort=False,
                pkl=True,
                interactive=True)
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    atlab = rn.lk.atlab
    rbd   = rn.get_value(rn.rbd)
    eunder= rn.get_value(rn.EUN)
    D_lp  = rn.get_value(rn.Delta_lpcorr)
    Delta_lp  = rn.get_value(rn.Delta_lp)
    DPI  = rn.get_value(rn.Dpi)

    eu1   = rn.get_value(rn.eu1)
    eu2   = rn.get_value(rn.eu2)
    expeu2= rn.get_value(rn.expeu2)
    p     = rn.get_value(rn.p)
   
    cell  = rn.cell[mol]
    atom_name = molecules[mol].atom_name

    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))
    write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
    system(gulp_cmd)
    

    fg = open('gulp.out','r')
    for line in fg.readlines():
        if line.find('- eunder:')>=0:
           l = line.split()
           i = int(l[2])-1

           an = atom_name[i] 
           ank= [mol,i]
           find = False
           if ank in atlab[an]:
              na = atlab[an].index(ank)
              find = True

           if find:
              # if abs(ebond[bn][nb][0]-float(l[4]))>0.001:
              print('- ReaxFF %d %s:' %(na,an),
                      'eunder: %10.6f' %eunder[an][na][0],
                      'eu1: %10.6f' %eu1[an][na][0],
                      'expeu2: %10.6f' %expeu2[an][na][0],
                      'ovun2: %10.6f' %p['ovun2_'+an],
                      'Delta_lp: %10.6f' %Delta_lp[an][na][0],
                      'DPI: %10.6f' %DPI[an][na][0],
                      'Delta_lpcorr: %10.6f' %D_lp[an][na][0])
              print('-   GULP %d %s:' %(na,an),
                      'eunder: %10.6f' %float(l[3]),
                      'eu1: %10.6f' %float(l[5]),
                      'expeu2: %10.6f' %float(l[7]),
                      'ovun2: %10.6f' %float(l[8]),
                      'Delta_lp: %10.6f' %float(l[10]),
                      'DPI: %10.6f' %float(l[11]),
                      'Delta_lpcorr: %10.6f' %float(l[9]))
    fg.close()
    # 'ovun2: %10.6f' %p['ovun2_'+an]
    # 'eu2: %10.6f' %eu2[an][na][0]
    # 'eu2: %10.6f' %float(l[6])


def debug_be(direcs={'ch4':'/home/feng/siesta/train/ch4'}):
    for key in direcs:
        mol = key
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                rc_scale='none',
                clip_op=False,
                optword='all',
                batch_size=1,
                sort=False,
                pkl=True,
                interactive=True)
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    bdlab = rn.lk.bdlab
    rbd   = rn.get_value(rn.rbd)
    ebond = rn.get_value(rn.EBD)
    sieng = rn.get_value(rn.sieng)
    pieng = rn.get_value(rn.pieng)
    ppeng = rn.get_value(rn.ppeng)

    bosi  = rn.get_value(rn.bosi)
    bopi  = rn.get_value(rn.bopi)
    bopp  = rn.get_value(rn.bopp)

    cell  = rn.cell[mol]

    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))
    write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
    system('/home/feng/gulp/Src/gulp<inp-gulp >gulp.out')
    atom_name = molecules[mol].atom_name

    fg = open('gulp.out','r')
    for line in fg.readlines():
        if line.find('- ebond:')>=0:
           l = line.split()
           i = int(l[2])-1
           j = int(l[3])-1
           bn = atom_name[i]+'-'+atom_name[j]
           if not bn in rn.bonds:
              bn = atom_name[j]+'-'+atom_name[i]
           bnk= [mol,i,j]
           bnkr= [mol,j,i]

           find = False
           if bnk in bdlab[bn]:
              nb = bdlab[bn].index(bnk)
              find = True
           elif bnkr in bdlab[bn]:
              nb = bdlab[bn].index(bnkr)
              find = True

           if find:
              if abs(ebond[bn][nb][0]-float(l[4]))>0.001:
                 print('- ReaxFF %s:' %bn, 
                          'rbd:',rbd[bn][nb][0],
                        'ebond:',ebond[bn][nb][0],
                        'sieng:',sieng[bn][nb][0],
                        'pieng:',pieng[bn][nb][0],
                        'ppeng:',ppeng[bn][nb][0],
                         'bosi:',bosi[bn][nb][0],
                         'bopi:',bopi[bn][nb][0],
                         'bopp:',bopp[bn][nb][0])
                 print('-   GULP %s:' %bn,
                          'rbd:',l[4],
                        'ebond:',l[5],
                        'sieng:',l[6],
                        'pieng:',l[7],
                        'ppeng:',l[8],
                         'bosi:',l[9],
                         'bopi:',l[10],
                         'bopp:',l[11])
    fg.close()
    

def debug_pen(mol='ch4',
          direcs={'ch4':'/home/feng/siesta/train/ch4'}):
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                rc_scale='none',
                optword='all',
                batch_size=1,
                sort=False,
                pkl=True,
                interactive=True)
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
  
    # ebond = rn.get_value(rn.ebond[mol])
    bo  = rn.get_value(rn.bo)
    bo0 = rn.get_value(rn.bo0)
    bop = rn.get_value(rn.bop)
    rbd = rn.get_value(rn.rbd)
    bodiv1 = rn.get_value(rn.bodiv1)
    bop_pi = rn.get_value(rn.bop_pi)

    f      = rn.get_value(rn.F)
    f11    = rn.get_value(rn.F_11)
    f12    = rn.get_value(rn.F_12)
    f45    = rn.get_value(rn.F_45)
    f4     = rn.get_value(rn.f_4)
    f5     = rn.get_value(rn.f_5)


    dboci= rn.get_value(rn.Di_boc)
    dbocj= rn.get_value(rn.Dj_boc)
    Dp   = rn.get_value(rn.Dp)
    BOP  = rn.get_value(rn.BOP)
    # print('-  shape of BOP: ',BOP.shape)

    nbd = rn.nbd 
    bdlab = rn.lk.bdlab
    bonds = rn.bonds
    atom_name = molecules[mol].atom_name
    p   = rn.p_

    nang = rn.nang
    anglab = rn.lk.anglab
    angs = rn.angs
    cell = rn.cell[mol]

    fbo = open('bo.txt','w')
    bd = 'C-C'
    # for bd in bonds:
    if nbd[bd]>0:
       # print('-  shape of new style: ',bo[bd].shape)
       print('\n-  bd: %s \n' %bd,file=fbo)
       for nb in range(nbd[bd]):
           print('- ',bd,
                   'r:',rbd[bd][nb][0],
                 'BOp:',bop[bd][nb][0],
                   'F:',f[bd][nb][0],
                'F_11:',f11[bd][nb][0],
                'F_12:',f12[bd][nb][0],
                'F_45:',f45[bd][nb][0],
                'F_4:',f4[bd][nb][0],
                'F_5:',f5[bd][nb][0],
               'Dboci:',dboci[bd][nb][0],
               'Dbocj:',dbocj[bd][nb][0],
                  'BO:',bo0[bd][nb][0],file=fbo)
    fbo.close()

    ff  = open('f.txt','w')
    f1   = rn.get_value(rn.f_1)
    f2   = rn.get_value(rn.f_2)
    f3   = rn.get_value(rn.f_3)
    dexpf3 = rn.get_value(rn.dexpf3)
    dexpf3t= rn.get_value(rn.dexpf3t)
    f3log  = rn.get_value(rn.f3log)

    # for bd in bonds:
    if nbd[bd]>0:
       print('\n-  bd: %s \n' %bd,file=ff)
       for nb in range(nbd[bd]):
           print('- ',bd,
                'r:',rbd[bd][nb][0],
              'f_1:',f1[bd][nb][0],
              'f_2:',f2[bd][nb][0],
              'f_3:',f3[bd][nb][0],
           'dexpf3:',dexpf3[bd][nb][0],
          'dexpf3t:',dexpf3[bd][nb][0],
            'f3log:',f3log[bd][nb][0],
                 file=ff)
    ff.close()

    fa = open('ang.txt','w')
    eang = rn.get_value(rn.EANG)
    expang= rn.get_value(rn.expang)
    f7  = rn.get_value(rn.f_7)
    f8  = rn.get_value(rn.f_8)
    thet= rn.get_value(rn.thet)
    for a in angs:
        if nang[a]>0:
           print('\n-  a: %s \n' %bd,file=fa)
           for na in range(nang[a]):
               iat = anglab[a][na][1]
               jat = anglab[a][na][2]
               kat = anglab[a][na][3]
               print('- ',a,
                        'thet:',thet[a][na][0],
                        'f7: ',f7[a][na][0],
                        'f8: ',f8[a][na][0],
                        'expang: ',expang[a][na][0],
                        'eang: ',eang[a][na][0],
                        file=fa)
    fa.close()

    f9 = rn.get_value(rn.f_9)
    epen = rn.get_value(rn.EPEN)
    fijk = rn.get_value(rn.fijk)
    bo   = rn.get_value(rn.bo) 
    
    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))
    write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
    system('/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out')

    fg = open('gulp.out','r')
    for line in fg.readlines():
        if line.find('- epen:')>=0:
           l = line.split()
           i = int(l[2])-1
           j = int(l[3])-1
           k = int(l[4])-1
           atn = atom_name[i]+'-'+atom_name[j]+'-'+atom_name[k]
           atnk= [mol,i,j,k]
           if not atn in angs:
              atn = atom_name[k]+'-'+atom_name[j]+'-'+atom_name[i]
           atnkr= [mol,k,j,i]

           find = False
           if atnk in anglab[atn]:
              na = anglab[atn].index(atnk)
              find = True
           elif atnkr in anglab[atn]:
              na = anglab[atn].index(atnkr)
              find = True

           if find:
              print('- ReaxFF %s:' %atn, 
                    'f9:',f9[atn][na][0],
                    'epen:',epen[atn][na][0],fijk[atn][na][0])
              print('-   GULP %s:' %atn,
                    'f9:',l[9],
                    'epen:',l[5],l[6])
           else:
              bd1 = atom_name[i]+'-'+atom_name[j]
              bd1k = [mol,i,j]
              if not bd1 in bonds:
                 bd1 = atom_name[j]+'-'+atom_name[i]
                 bd1k = [mol,j,i]

              bd2 = atom_name[i]+'-'+atom_name[k]
              bd2k = [mol,i,k]
              if not bd2 in bonds:
                 bd2 = atom_name[k]+'-'+atom_name[i]
                 bd2k = [mol,k,i]

              if bd1k in bdlab[bd1] and bd2k in bdlab[bd2]:
                 nb1 = bdlab[bd1].index(bd1k)
                 nb2 = bdlab[bd2].index(bd2k)

                 boij = bo[bd1][nb1][0]
                 boik = bo[bd2][nb2][0]
                 print('-   GULP %s:' %atn,
                        l[9],l[5],l[6],
                        'bo:',bd1,boij,l[10],bd2,boik,l[11])
              else:
                 print('-   GULP %s:' %atn,
                        l[9],l[5],l[6],
                        'bo:',bd1,'none',l[10],bd2,'none',l[11])

    fg.close()


def debug_hb(direcs={'ho1':'/home/feng/siesta/train/ho1'}):
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                rc_scale='none',
                optword='all',
                batch_size=300,
                sort=False,
                pkl=True,
                interactive=True)
    for key in direcs:
        mol = key
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
    
    cell = rn.cell[mol]

    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))

    fh   = open('hb.txt','w')         #### hbond
    fhb  = rn.get_value(rn.fhb)
    frhb = rn.get_value(rn.frhb)
    rhb  = rn.get_value(rn.rhb)
    ehb  = rn.get_value(rn.EHB)
    nhb  = rn.nhb
    
    for hb in rn.hbs:
        if nhb[hb]>0:
           print('\n-  hbond: %s \n' %hb,file=fh)
           for nb in range(nhb[hb]):
               print('- ',hb,
                    'rhb:',rhb[hb][nb][0],
                    'fhb:',fhb[hb][nb][0],
                   'frhb:',frhb[hb][nb][0],
                    'ehb:',ehb[hb][nb][0],
                       file=fh)
    fh.close()                    ###### hbond


def debug_ang(direcs={'cho-4':'/home/feng/siesta/cho4'},
              gulp_cmd='/home/feng/gulp/gulp-5.0/Src/gulp<inp-gulp >gulp.out'):
    for key in direcs:
        mol = key
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                optword='nocoul',
                batch_size=1,
                sort=False,
                pkl=True,
                interactive=True)
    molecules = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
  
    f7    = rn.get_value(rn.f_7)
    f8    = rn.get_value(rn.f_8)
    expang= rn.get_value(rn.expang)
    eang  = rn.get_value(rn.EANG)
    theta = rn.get_value(rn.theta)
    theta0 = rn.get_value(rn.theta0)
    SBO   = rn.get_value(rn.SBO)
    sbo   = rn.get_value(rn.sbo)
    pbo   = rn.get_value(rn.pbo)
    rnlp  = rn.get_value(rn.rnlp)
    D_ang = rn.get_value(rn.D_ang)

    Delta = rn.get_value(rn.Delta)

    atom_name = molecules[mol].atom_name
    p   = rn.p_

    nang   = rn.nang
    anglab = rn.lk.anglab
    angs   = rn.angs
    atom_lab   = rn.lk.atom_lab
    cell   = rn.cell[mol]
    # print(eang)
    # print(nang)

    A = Atoms(symbols=molecules[mol].atom_name,
              positions=molecules[mol].x[0],
              cell=cell,
              pbc=(1, 1, 1))
    write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
    system(gulp_cmd)

    fg = open('gulp.out','r')

    angfind = {}
    for atn in rn.angs:
        angfind[atn] = []

    for line in fg.readlines():
        if line.find('- eval:')>=0:
           l = line.split()
           i = int(l[2])-1
           j = int(l[3])-1
           k = int(l[4])-1
           atn = atom_name[i]+'-'+atom_name[j]+'-'+atom_name[k]
           atnk= [mol,i,j,k]
           if not atn in angs:
              atn = atom_name[k]+'-'+atom_name[j]+'-'+atom_name[i]
           atnkr= [mol,k,j,i]

           find = False
           if atnk in anglab[atn]:
              na = anglab[atn].index(atnk)
              find = True
           elif atnkr in anglab[atn]:
              na = anglab[atn].index(atnkr)
              find = True

           aj = atom_lab.index([mol,j])

           if find:
              angfind[atn].append(na)
              print('- ReaxFF %s:' %atn, na,atnk,
                    'sbo:',sbo[atn][na][0],
                    'theta:',theta[atn][na][0],
                    'theta0:',theta0[atn][na][0],
                    'expang:',expang[atn][na][0],
                    'eang:',eang[atn][na][0])
              print('-   GULP %s:' %atn,na,atnk,
                    'sbo:',l[11],
                    'theta:',l[9],
                    'theta0:',l[10],
                    'expang:',l[6],
                    'eang:',l[5])
           else:
              print('-NF GULP %s:' %atn,na,
                    'sbo:',l[11],
                    'theta:',l[9],
                    'theta0:',l[10],
                    'expang:',l[6],
                    'eang:',l[5])

    fg.close()

    print('\n-  angles not find\n')

    for atn in rn.angs:
        for na in range(rn.nang[atn]):
            if not na in angfind[atn]:
               al = anglab[atn][na]
               print('- ReaxFF %s:' %atn, na,al,
                      'sbo:',sbo[atn][na][0],
                      'theta0:',theta0[atn][na][0],
                      'expang:',expang[atn][na][0],
                      'eang:',eang[atn][na][0])


