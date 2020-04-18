from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import system, getcwd, chdir,listdir,environ
from irff.irnn import IRNN
# from train_IRNN import cnn
import numpy as np
import tensorflow as tf
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def allgrad(direcs=None,batch=100):
    rn = IRNN(libfile='ffield',direcs=direcs,
                dft='cpmd',
                batch_size=batch,
                pkl=True)
    rn.initialize()
    rn.session(learning_rate=1.0e-4,method='AdamOptimizer') 
    grads,v,vn = rn.get_all_gradient()

    fg = open('grad.txt','w')
    for g,t,tt in zip(grads,vn,v):
        print('-  gradients of %20s is: %12.5f' %(t,g),' value is:',tt,file=fg)
    fg.close()
    rn.sess.close()


def gradb(direcs,v='bo5',bd='C-O',dft='siesta',batch=100):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
    '''
    v  = v+'_'+bd  

    print('-  grading ... ...')
    for m in direcs:
        mol = m

    rn = IRNN(libfile='ffield',direcs=direcs,
                dft=dft,
                batch_size=batch,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=3.0e-4,method='AdamOptimizer') 

    fg = open('gradient.txt','w')

    bdlit = ['bop_si','bop_pi','bop_pp',
             'bosi','F',
             'bopi',
             'bopp',
             'bo','bso',
             'powb','expb','EBD']
    # bdlit = ['bo','EBD']

    gl = rn.get_gradient(rn.Loss,rn.p[v]) 
    print('-  the gradient of Loss/%s is:' %v,gl,file=fg)
    bonds = rn.bonds
    # bonds = [bd]
    for b in bonds:
        if rn.nbd[b]>0:
           for l in bdlit:
               grad = rn.get_gradient(rn.__dict__[l][b],rn.p[v]) 
               print('-  the gradient of %s/%s is: ' %(l+'_'+b,v),
                     grad,file=fg) 
    fg.close()
    
    ml = ['ebond','elone','eover','eunder','eang',
          'epen','tconj','etor','efcon','evdw','ehb']

    # for mol in direcs:
    for l in ml:
        grad = rn.get_gradient(rn.__dict__[l][mol],rn.p[v]) 
        fg = open('gradient.txt','a')
        print('-  the gradient of %s_%s/%s is: ' %(l,mol,v),grad,file=fg)
        fg.close()

    alit  = ['Delta_lp','Delta_lpcorr','Dpi','Delta_e','nlp','slp','EL',
             'EOV','so','otrm1','otrm2',
             'EUN']
    atoms = rn.spec
    
    for l in alit:
        # atoms = bd.split('-')
        for a in atoms:
            grad = rn.get_gradient(rn.__dict__[l][a],rn.p[v]) 
            fg = open('gradient.txt','a')
            print('-  the gradient of %s_%s/%s is: ' %(l,a,v),grad,file=fg)
            fg.close()
    rn.sess.close()


def grada(direcs,v='bo1_H-H',dft='siesta',batch=100):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
    '''
    print('-  grading ... ... ')
    for m in direcs:
        mol = m

    rn = IRNN(libfile='ffield',direcs=direcs,
                dft=dft,
                batch_size=batch,
                pkl=True)
    rn.initialize()
    rn.session(learning_rate=1.0e-4,method='AdamOptimizer') 

    fg = open('gradient.txt','w')
    l = 'eang'
    grad = rn.get_gradient(rn.__dict__[l][mol],rn.p[v]) 
    print('-  the gradient of %s_%s/%s is: ' %(l,mol,v),grad,file=fg)
    fg.close()

    # anglit = ['D_ang','thet','theta0','expang','f_7','f_8','EANG','EPEN','ETC']
    anglit = ['sbo','f_8','EANG','EPEN','ETC']
    angs = rn.angs
    # angs = ['H-O-H']
    for ang in angs:
        if rn.nang[ang]>0:
           for l in anglit:
               grad = rn.get_gradient(rn.__dict__[l][ang],rn.p[v]) 
               fg = open('gradient.txt','a')
               print('- the gradient of %s/%s is: ' %(l+'_'+ang,v),
                      grad,file=fg) 
               fg.close()
    rn.sess.close()


def gradt(v='bo2_C-N'):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
    '''
    print('-  grading ... ... ')
    direcs={'nm':'/home/feng/cpmd_data/packmol/nm',
           'nme':'/home/feng/cpmd_data/packmol/nme'}

    rn = IRNN(libfile='ffield',direcs=direcs,
                dft='cpmd',
                batch_size=200,
                pkl=True)
    rn.initialize()
    rn.session(learning_rate=1.0e-4,method='AdamOptimizer') 

    fg = open('gradient.txt','w')
    grad = rn.get_gradient(rn.Loss,rn.p[v]) 
    print('-  the gradient of Loss/%s is: ' %v,grad,file=fg) 

    molit = ['etor','efcon']
    for mol in rn.direcs:
        for l in molit:
            grad = rn.get_gradient(rn.__dict__[l][mol],rn.p[v]) 
            print('-  the gradient of %s/%s is: ' %(l+'_'+mol,v),
                  grad,file=fg) 
    fg.close()

    torlit = ['f_10','f_11','expv2','ETOR','Efcon']
    i = 0
    for tor in rn.tors:
        if rn.ntor[tor]>0:
           # if i<=100:
           if tor=='C-O-O-N' or tor=='C-O-O-H':
              for l in torlit:
                  fg = open('gradient.txt','a')
                  grad = rn.get_gradient(rn.__dict__[l][tor],rn.p[v]) 
                  print('- the gradient of %s/%s is: ' %(l+'_'+tor,v),
                         grad,file=fg) 
                  fg.close()
              i += 1
    
    rn.sess.close()


def plot(e1,e2,en,mol):
    plt.figure()
    plt.ylabel('Energies')
    plt.xlabel('Step')

    plt.plot(e1,label=r'$IRNN$', color='red', linewidth=2, linestyle='-.')
    plt.plot(e2,label=r'$IRNN$', color='blue', linewidth=2, linestyle='--')

    plt.legend()
    plt.savefig('%s_%s.eps' %(en,mol)) 
    plt.close()


def debug_plot():
    mol = 'ch4'
    direcs={mol:'/home/feng/siesta/train/ch4'}

    batch= {'others':300}
    rn = IRNN(libfile='ffield',direcs=direcs, 
                dft='cpmd',
                rc_scale='bo1',
                optword='all',
                batch_size=300,
                sort=False,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
    # rn.plot()

    ebondr = rn.get_value(rn.ebond[mol])
    eloner = rn.get_value(rn.elone[mol])
    eoverr = rn.get_value(rn.eover[mol])
    eunderr= rn.get_value(rn.eunder[mol])
    eangr  = rn.get_value(rn.eang[mol])
    epenr  = rn.get_value(rn.epen[mol])
    tconjr = rn.get_value(rn.tconj[mol])
    etorr  = rn.get_value(rn.etor[mol])
    efcon  = rn.get_value(rn.efcon[mol])
    evdw   = rn.get_value(rn.evdw[mol])

    rn.sess.close()
    del rn

    tc = cnn(libfile='ffield',direcs=direcs, 
             dft='cpmd',
             rc_scale='bo1',
             optword='all',
             batch_size=batch,
             sort=False,
             pkl=True,
             interactive=True)
    tc.session(learning_rate=1.0e-4,method='AdamOptimizer')  
    ebondt = tc.get_value(tc.M[mol].ebond)
    elonet = tc.get_value(tc.M[mol].elone)
    eovert = tc.get_value(tc.M[mol].eover)
    eundert = tc.get_value(tc.M[mol].eunder)
    eangt = tc.get_value(tc.M[mol].eang)
    epent = tc.get_value(tc.M[mol].epen)
    tconjt= tc.get_value(tc.M[mol].tconj)
    etort= tc.get_value(tc.M[mol].etor)
    etort= tc.get_value(tc.M[mol].etor)
    fconj= tc.get_value(tc.M[mol].fconj)
    evdwt= tc.get_value(tc.M[mol].evdw)


    tc.sess.close()

    plot(ebondr,ebondt,'ebond','mol')
    plot(eloner,elonet,'elone','mol')
    plot(eoverr,eovert,'eover','mol')
    plot(eunderr,eundert,'eunder','mol')
    plot(eangr,eangt,'eang','mol')
    plot(epenr,epent,'epen','mol')
    plot(tconjr,tconjt,'etcon','mol')
    plot(etorr,etort,'etor','mol')
    plot(efcon,fconj,'efcon','mol')
    plot(evdw,evdwt,'evdw','mol')
    

def compare_f(dire):
    mol = 'mol'
    direcs={mol:dire}
    batch= {'others':10}
    rn = IRNN(libfile='ffield',direcs=direcs, 
                dft='cpmd',
                rc_scale='bo1',
                optword='all',
                batch_size=10,
                sort=False,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
    rn.plot()
  
    ebondr = rn.get_value(rn.ebond[mol])
    bo = rn.get_value(rn.bo)
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
    print('-  shape of BOP: ',BOP.shape)
    print('-  number of bond: ',rn.lk.nbond)
    print('-  number of atoms: ',len(rn.lk.atom_lab))

    nbd = rn.nbd 
    bdlab = rn.lk.bdlab
    bonds = rn.bonds
    rn.sess.close()
    del rn

    ## another session
    tc = cnn(libfile='ffield',direcs=direcs, 
             dft='cpmd',
             rc_scale='bo1',
             optword='all',
             batch_size=batch,
             sort=False,
             pkl=True,
             interactive=True)
    tc.session(learning_rate=1.0e-4,method='AdamOptimizer')  

    ebondt = tc.get_value(tc.M[mol].ebond)
    r,BOp,BO,Bodiv1,Bodiv2,Bodiv3,BOp_pi    = \
            tc.get_value([tc.M[mol].r,tc.M[mol].BOp,tc.M[mol].BO,
                          tc.M[mol].bodiv1,tc.M[mol].bodiv2,tc.M[mol].bodiv3,
                          tc.M[mol].BOp_pi])

    fbosi = tc.get_value(tc.M[mol].fbosi)

    F,F_11,F_12,F_45,F_4,F_5 = \
             tc.get_value([tc.M[mol].F,tc.M[mol].F_11,tc.M[mol].F_12,tc.M[mol].F_45,
                           tc.M[mol].f_4,tc.M[mol].f_5])

    Db       = tc.get_value(tc.M[mol].Delta_boc)
    Vb       = tc.get_value(tc.M[mol].P['valboc'])
    print('-  shape of F_11: ',F_11.shape)
    print('-  shape of Delta_boc: ',Db.shape)

    for bd in bonds:
        if nbd[bd]>0:
           # print('-  shape of new style: ',bo[bd].shape)
           print('\n-  bd: %s \n' %bd)
           for nb in range(nbd[bd]):
               if bdlab[bd][nb][0] == mol:
                  iatom = int(bdlab[bd][nb][1])
                  jatom = int(bdlab[bd][nb][2])
                  if abs(BO[0][iatom][jatom]-bo[bd][nb][0])>0.0001:
                     print('- ',bd,iatom,jatom,
                          'BO:',BO[0][iatom][jatom],bo[bd][nb][0],
                           'F:',F[0][iatom][jatom],f[bd][nb][0],
                        'F_11:',F_11[0][iatom][jatom],f11[bd][nb][0],
                        'F_12:',F_12[0][iatom][jatom],f12[bd][nb][0],
                        'F_45:',F_45[0][iatom][jatom],f45[bd][nb][0],
                        'F_4:',F_4[0][iatom][jatom],f4[bd][nb][0],
                        'F_5:',F_5[0][iatom][jatom],f5[bd][nb][0],
                       'Dboci:',Db[0][iatom],dboci[bd][nb][0],
                       'Dbocj:',Db[0][jatom],dbocj[bd][nb][0])
    tc.sess.close()


def compare_bo(dire):
    # mol = 'mol'
    # direcs={mol:dire}

    mol = 'mol'
    direcs={mol:dire}
    # direcs={'nm13':'/home/gfeng/cpmd/train/nmr/nm13',
    #         'nm4':'/home/gfeng/cpmd/train/nmr/nm2004',
    #         'nm11':'/home/gfeng/cpmd/train/nmr/nm11'}

    batch= {'others':10}
    rn = IRNN(libfile='ffield',direcs=direcs, 
                dft='cpmd',
                rc_scale='bo1',
                optword='all',
                batch_size=10,
                sort=False,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

  
    ebondr = rn.get_value(rn.ebond[mol])
    bo = rn.get_value(rn.bo)
    bop = rn.get_value(rn.bop)
    rbd = rn.get_value(rn.rbd)
    bodiv1 = rn.get_value(rn.bodiv1)
    bop_pi = rn.get_value(rn.bop_pi)

    dboci= rn.get_value(rn.Di_boc)
    dbocj= rn.get_value(rn.Dj_boc)
    Dp   = rn.get_value(rn.Dp)
    BOP  = rn.get_value(rn.BOP)

    nbd = rn.nbd 
    bdlab = rn.lk.bdlab
    bonds = rn.bonds
    rn.sess.close()
    del rn

    ## another session
    tc = cnn(libfile='ffield',direcs=direcs, 
             dft='cpmd',
             rc_scale='bo1',
             optword='all',
             batch_size=batch,
             sort=False,
             pkl=True,
             interactive=True)
    tc.session(learning_rate=1.0e-4,method='AdamOptimizer')  

    ebondt = tc.get_value(tc.M[mol].ebond)
    r,BOp,BO,Bodiv1,Bodiv2,Bodiv3,BOp_pi    = \
            tc.get_value([tc.M[mol].r,tc.M[mol].BOp,tc.M[mol].BO,
                          tc.M[mol].bodiv1,tc.M[mol].bodiv2,tc.M[mol].bodiv3,
                          tc.M[mol].BOp_pi])


    Db       = tc.get_value(tc.M[mol].Delta_boc)
    Vb       = tc.get_value(tc.M[mol].P['valboc'])

    for bd in bonds:
        if nbd[bd]>0:
           # print('-  shape of new style: ',bo[bd].shape)
           print('\n-  bd: %s \n' %bd)
           for nb in range(nbd[bd]):
               if bdlab[bd][nb][0] == mol:
                  iatom = int(bdlab[bd][nb][1])
                  jatom = int(bdlab[bd][nb][2])
                  if abs(BO[0][iatom][jatom]-bo[bd][nb][0])>0.0001:
                     print('- ',bd,iatom,jatom,
                         'BOp:',BOp[0][iatom][jatom],bop[bd][nb][0],
                       'Dboci:',Db[0][iatom],dboci[bd][nb][0],
                       'Dbocj:',Db[0][jatom],dbocj[bd][nb][0],
                       'BO:',BO[0][iatom][jatom],bo[bd][nb][0],)
    tc.sess.close()
    print('-  shape of BOp: ',BOp.shape)
    print('-  shape of Dp: ',Dp.shape)


def compare_d(dire):
    mol = 'mol'
    direcs={mol:dire}
    batch= {'others':10}
    rn = IRNN(libfile='ffield',direcs=direcs, 
                dft='cpmd',
                rc_scale='bo1',
                optword='all',
                batch_size=10,
                sort=False,
                pkl=True,
                interactive=True)
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
    # rn.plot()

    dboci = rn.get_value(rn.Di_boc)
    dbocj = rn.get_value(rn.Dj_boc)
    dp    = rn.get_value(rn.Dp)
    deltap= rn.get_value(rn.Deltap)
    bop   = rn.get_value(rn.BOP)
    print('-  shape of BOP: ',bop.shape)
    print('-  number of bond: ',rn.lk.nbond)
    print('-  number of atoms: ',len(rn.lk.atom_lab))
    print('-  shape of deltap: ',deltap.shape)
    print('-  shape of dp: ',dp.shape)
    maxn = dp.shape[1]
    nbd = rn.nbd 
    bdlab = rn.lk.bdlab
    bonds = rn.bonds
    rn.sess.close()
    del rn

    ## another session
    tc = cnn(libfile='ffield',direcs=direcs, 
             dft='cpmd',
             rc_scale='bo1',
             optword='all',
             batch_size=batch,
             sort=False,
             pkl=True,
             interactive=True)
    tc.session(learning_rate=1.0e-4,method='AdamOptimizer')  

    natom = tc.M[mol].molecule.natom
    AN    = tc.M[mol].molecule.atom_name
    V     = tc.get_value(tc.P[mol]['val'])
    Dp,Db = tc.get_value([tc.M[mol].Deltap,tc.M[mol].Delta_boc])
    Dp = Dp+V

    for na in range(natom):
        if Dp[0][na]-deltap[na][0]>0.0001:
           print('- ',na,AN[na],
              'Deltap:',Dp[0][na],deltap[na][0])
           # for n in range(maxn):
           #     print('- bop:',dp[na][n][0])
    tc.sess.close()


def compare_eb(dire):
    system('rm *.pkl')
    mol = 'dp'
    # direcs={mol:dire}
    direcs={'dp':'/home/gfeng/cpmd/train/nmr/dop'}

    batch= {'others':10}
    rn = IRNN(libfile='ffield',direcs=direcs, 
                dft='cpmd',
                rc_scale='bo1',
                optword='all',
                batch_size=10,
                sort=False,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    dboci = rn.get_value(rn.Di_boc)
    dbocj = rn.get_value(rn.Dj_boc)
    dp    = rn.get_value(rn.Dp)
    deltap= rn.get_value(rn.Deltap)
    bop   = rn.get_value(rn.BOP)

    powb = rn.get_value(rn.powb)
    expb = rn.get_value(rn.expb)
    sieng= rn.get_value(rn.sieng)

    EBD= rn.get_value(rn.EBD)
    ebd= rn.get_value(rn.ebond)
    ebda = rn.get_value(rn.ebda)

    bdlink = rn.bdlink

    print('-  shape of BOP: ',bop.shape)
    print('-  number of bond: ',rn.lk.nbond)
    print('-  number of atoms: ',len(rn.lk.atom_lab))
    print('-  shape of deltap: ',deltap.shape)
    print('-  shape of ebda: ',ebda[mol].shape)

    nbd = rn.nbd 
    bdlab = rn.lk.bdlab
    bdlall = rn.lk.bdlall
    bonds = rn.bonds
    rn.sess.close()
    del rn

    ## another session
    tc = cnn(libfile='ffield',direcs=direcs, 
             dft='cpmd',
             rc_scale='bo1',
             optword='all',
             batch_size=batch,
             sort=False,
             pkl=True,
             interactive=True)
    tc.session(learning_rate=1.0e-4,method='AdamOptimizer')  

    natom = tc.M[mol].molecule.natom
    nbond = tc.M[mol].molecule.nbond
    AN    = tc.M[mol].molecule.atom_name
    V     = tc.get_value(tc.P[mol]['val'])
    Dp,Db = tc.get_value([tc.M[mol].Deltap,tc.M[mol].Delta_boc])
    Dp = Dp+V

    Powb,Expb,Sieng,EBOND = tc.get_value([tc.M[mol].powb,tc.M[mol].expb,
                             tc.M[mol].sieng,tc.M[mol].EBOND])
    ebond = tc.get_value(tc.M[mol].ebond)

    E = np.zeros([natom,natom])

    for bd in bonds:
        if nbd[bd]>0:
           # print('-  shape of new style: ',bo[bd].shape)
           print('\n-  bd: %s \n' %bd)
           for nb in range(nbd[bd]):
               if bdlab[bd][nb][0] == mol:
                  iatom = int(bdlab[bd][nb][1])
                  jatom = int(bdlab[bd][nb][2])

                  E[iatom][jatom] = EBD[bd][nb][0]
                  E[jatom][iatom] = EBD[bd][nb][0]

                  if abs(EBOND[0][iatom][jatom]-EBD[bd][nb][0])>0.0001:
                     print('- ',bd,iatom,jatom,
                     'powb:',Powb[0][iatom][jatom],powb[bd][nb][0],
                     'expb:',Expb[0][iatom][jatom],expb[bd][nb][0],
                    'sieng:',Sieng[0][iatom][jatom],sieng[bd][nb][0],
                    'EBOND:',EBOND[0][iatom][jatom],EBD[bd][nb][0])
    e = np.sum(E)
    Et= EBOND[0]
    ee= np.sum(Et)
    print('-  ebond:',ebond[0],ebd[mol][0],'e:',0.5*e,0.5*ee)
    print('-  number of bond:',len(bdlink[mol]),nbond)
    tc.sess.close()

    # for i,e in enumerate(ebda[mol]): 
    #     nb = bdlink[mol][i][0]
    #     bn = bdlall[nb]
    #     print('-  bond name %d:' %i,bn) 



def compare_u(dire):
    # mol = 'mol'
    # direcs={mol:dire}

    mol = 'nm13'
    #direcs={mol:dire}
    direcs={'nm13':'/home/gfeng/cpmd/train/nmr/nm13',
              'hc':'/home/gfeng/cpmd/train/nmr/hc1',
             'nm4':'/home/gfeng/cpmd/train/nmr/nm2004'}

    batch= {'others':10}
    rn = IRNN(libfile='ffield',direcs=direcs, 
                dft='cpmd',
                rc_scale='bo1',
                optword='all',
                batch_size=10,
                sort=False,
                pkl=True,
                interactive=True)
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
    # rn.plot()

    Dpi = rn.get_value(rn.Dpi)
    Dlp = rn.get_value(rn.Delta_lp)
    DLP = rn.get_value(rn.DLP)
    # bpi = rn.get_value(rn.bpi)
    Dlpc= rn.get_value(rn.Delta_lpcorr)

    bopi = rn.get_value(rn.bopi)
    bopp = rn.get_value(rn.bopp)
    D    = rn.get_value(rn.D)
 
    DPI  = rn.get_value(rn.DPI)

    print('-  shape of BOPI: ',Dpi['C'].shape)

    spec = rn.spec
    atlab= rn.lk.atlab
    bonds= rn.bonds
    # print('- bonds:\n',bonds)
    nbd  = rn.nbd
    bdlab= rn.lk.bdlab
    natom= rn.lk.natom

    rn.sess.close()
    del rn

    ## another session
    tc = cnn(libfile='ffield',direcs=direcs, 
             dft='cpmd',
             rc_scale='bo1',
             optword='all',
             batch_size=batch,
             sort=False,
             pkl=True,
             interactive=True)
    tc.session(learning_rate=1.0e-4,method='AdamOptimizer')  

    # natom = tc.M[mol].molecule.natom
    AN    = tc.M[mol].molecule.atom_name
    bonds = tc.bonds

    # print('- bonds:\n',bonds)

    DPI1,Dlp1    = tc.get_value([tc.M[mol].DPI,tc.M[mol].Delta_lp])
    DLP1          = tc.get_value(tc.M[mol].DLP)
    DPI1_          = tc.get_value(tc.M[mol].DPI_)
    Dlpc1         = tc.get_value(tc.M[mol].Delta_lpcorr)
    BO_pi         = tc.get_value(tc.M[mol].BO_pi)
    BO_pp         = tc.get_value(tc.M[mol].BO_pp)
    Delta         = tc.get_value(tc.M[mol].Delta)
    V             = tc.get_value(tc.P[mol]['val'])
    Delta  += V
    for sp in spec:
        for l,lab in enumerate(atlab[sp]):
            if lab[0]==mol:
               i = int(lab[1])
               print('- ',i,AN[i],
                 'DPI:',Dpi[sp][l][0],DPI1[0][i],
                 'Dlpc:',Dlpc[sp][l][0],Dlpc1[0][i],
                 'Delta:',D[sp][l][0],Delta[0][i])

                 # 'Dlp:',Dlp[sp][l][0],Dlp1[0][i],
 
    print('-  shape of DPI: ',DPI.shape)   
    tc.sess.close()


def torsion(direcs=None,
            dft='siesta'):
    for m in direcs:
        mol = m

    rn = IRNN(libfile='ffield',direcs=direcs, 
                dft=dft,
                rc_scale='none',
                optword='all',
                batch_size=200,
                sort=False,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
    # rn.plot()

    
    BOtij = rn.get_value(rn.BOtij) 
    BOtjk = rn.get_value(rn.BOtjk)
    BOtkl = rn.get_value(rn.BOtkl)  

    f10 = rn.get_value(rn.f_10)
    f11 = rn.get_value(rn.f_11)
    Etor= rn.get_value(rn.Etor)

    torp = rn.torp
    tors = rn.tors
    # print('-  shape of expv2: ',expv2['C'].shape)
    print('-  num of torp %d.' %len(torp))
    print('-  num of tors %d.' %len(tors))
    spec  = rn.spec
    torlab= rn.lk.torlab
    ntor  = rn.ntor
    tors  = rn.tors

    rn.sess.close()

    for tn in tors:
        if ntor[tn]>0:
           for t in range(ntor[tn]):
               if torlab[tn][t][0] == mol:
                  print('- ',t,tn,
                     'BOij:',BOtij[tn][t][0],
                     'BOjk:',BOtjk[tn][t][0],
                     'BOkl:',BOtkl[tn][t][0],
                      'f10:',f10[tn][t][0],
                      'f11:',f11[tn][t][0])

    print('-  shape of Etor:',Etor[mol].shape)
    print('-  num of torp %d.' %len(torp))
    print('-  num of tors %d.' %len(tors))


def get_v(direcs={'ch4cdeb':'/home/feng/siesta/ch4c4'},
          batch=50):
    for m in direcs:
        mol = m
    rn = IRNN(libfile='ffield',direcs=direcs, 
                dft='siesta',
                rc_scale='none',
                optword='all',
                batch_size=batch,
                sort=False,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')

    p      = rn.p_
    Dp_ = {}
    for sp in rn.spec:
        if rn.nsp[sp]>0:
           Dp_[sp] = tf.gather_nd(rn.Deltap,rn.atomlist[sp])
    Dp = rn.get_value(Dp_)


    Dpi    = rn.get_value(rn.Dpi)
    Dlp    = rn.get_value(rn.Dlp)

    powb   = rn.get_value(rn.powb)
    expb   = rn.get_value(rn.expb)

    bop_si = rn.get_value(rn.bop_si)
    bosi   = rn.get_value(rn.bosi)

    bop_pi = rn.get_value(rn.bop_pi)
    bopi   = rn.get_value(rn.bopi)

    bop_pp = rn.get_value(rn.bop_pp)
    bopp   = rn.get_value(rn.bopp)

    Fsi    = rn.get_value(rn.Fsi)
    Fpi    = rn.get_value(rn.Fpi)
    Fpp    = rn.get_value(rn.Fpp)

    fsi_1  = rn.get_value(rn.fsi_1)
    fpi_1  = rn.get_value(rn.fpi_1)
    fpp_1  = rn.get_value(rn.fpp_1)

    fsi_4  = rn.get_value(rn.fsi_4)
    fpi_4  = rn.get_value(rn.fpi_4)
    fpp_4  = rn.get_value(rn.fpp_4)

    f_2    = rn.get_value(rn.f_2)
    f_3    = rn.get_value(rn.f_3)

    Delta_lpcorr = rn.get_value(rn.Delta_lpcorr)
    Delta_lp     = rn.get_value(rn.Delta_lp)
    D            = rn.get_value(rn.D)
    slp          = rn.get_value(rn.slp)

    DPIL         = rn.get_value(rn.DPIL)
    DLP          = rn.get_value(rn.DLP)

    otrm1        = rn.get_value(rn.otrm1)
    EOV          = rn.get_value(rn.EOV)
    EUN          = rn.get_value(rn.EUN)

    nbd    = rn.nbd
    nsp    = rn.nsp
    spec   = rn.spec
    bonds  = rn.bonds
    bd     = 'H-H'
    # bonds  = [bd]
    bdlab  = rn.lk.bdlab
    atlab  = rn.lk.atlab
    atlall = rn.lk.atlall
    alist  = rn.atomlist

    fbo = open('bo.txt','w')
    for bd in bonds:
        if nbd[bd]>0:
           for i,pb in enumerate(powb[bd]):
               print('-bond %s:' %bd,
                     '-bosi- %10.8f' %bosi[bd][i][0],
                     '-bopi- %10.8f' %bopi[bd][i][0],
                     '-bopp- %10.8f' %bopp[bd][i][0],
                     '-bopsi- %10.8f' %bop_si[bd][i][0],
                     '-boppi- %10.8f' %bop_pi[bd][i][0],
                     '-boppp- %10.8f' %bop_pp[bd][i][0],
                     '-fsi_1- %11.8f' %fsi_1[bd][i][0],
                     '-fpi_1- %11.8f' %fpi_1[bd][i][0],
                     '-fpp_1- %11.8f' %fpp_1[bd][i][0],
                     '-fsi_4- %11.8f' %fsi_1[bd][i][0],
                     '-fpi_4- %11.8f' %fpi_1[bd][i][0],
                     '-fpp_4- %11.8f' %fpp_1[bd][i][0],
                     '-Fsi- %10.8f' %Fsi[bd][i][0],
                     '-Fpi- %10.8f' %Fpi[bd][i][0],
                     '-Fpp- %10.8f' %Fpp[bd][i][0],file=fbo)
    fbo.close()

    fa = open('a.txt','w')
    # spec=['H']
    for atom in spec:
        if nsp[atom]>0:
           for i,dl in enumerate(Delta_lpcorr[atom]):
               print('-atom %s:' %atom,
               	     '-Delta- %11.8f' %(D[atom][i][0]),
                     '-Deltap- %11.8f' %(Dp[atom][i][0]),
                     '-Delta_lpcorr- %11.8f' %Delta_lpcorr[atom][i][0],
                     '-Delta_lc+v- %11.8f' %(Delta_lpcorr[atom][i][0]+p['val_'+atom]),
                     '-Delta_lp- %11.8f' %Delta_lp[atom][i][0],
                     '-EOV- %11.8f' %EOV[atom][i][0],
                     '-EUN- %11.8f' %EUN[atom][i][0],file=fa)
    fa.close()
                                       
    fab = open('ab.txt','w')
    # mol = 'chw2-2-0'
    for bd in bonds:
        if nbd[bd]>0:
           [a1,a2] = bd.split('-')
           for i,pb in enumerate(powb[bd]):
               blb  = bdlab[bd][i]
               # if blb[0]==mol:
               mol = blb[0]
               na1   = atlab[a1].index([mol,blb[1]])
               na2   = atlab[a2].index([mol,blb[2]])
               na1_  = alist[a1][na1][0]
               na2_  = alist[a2][na2][0]
               # print(DLP.shape)
               print('-bond %s:' %bd,                  #  '-atm %d %d:' %(na1_,na2_),
                    '-fsi_1- %11.8f' %fsi_1[bd][i][0],
                    '-fpi_1- %11.8f' %fpi_1[bd][i][0],
                    '-fpp_1- %11.8f' %fpp_1[bd][i][0],
                    '-f_2- %11.8f' %f_2[bd][i][0],
                    '-f_3- %11.8f' %f_3[bd][i][0],
                    '-DPIL- %11.8f' %(DPIL[na1_][0]),
                    '-DPIL- %11.8f' %(DPIL[na2_][0]),
                    '-Dlp- %11.8f' %(Dlp[a1][na1][0]),
                    '-Dlp- %11.8f' %(Dlp[a2][na2][0]),
                    file=fab)  
    fab.close()
