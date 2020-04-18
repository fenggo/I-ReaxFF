from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import system, getcwd, chdir,listdir,environ
from irff.reax import ReaxFF
from irff.reax import logger
# from train_reaxff import cnn
import numpy as np

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logger(flog='debug.log')


def allgrad(direcs=None,batch=100):
    rn = ReaxFF(libfile='ffield',direcs=direcs,
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


def gradb(direcs,v='bo5',bd='H-H',nn=False,bo_layer=[9,2],
          debd=True,deba=True,deang=True,
          dft='siesta',batch=100):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
    '''
    v_ = v
    v  = v+'_'+bd  
    print('-  grading ... ...')
    ffield = 'ffield.json' if nn else 'ffield'
    
    rn = ReaxFF(libfile=ffield,direcs=direcs,
                dft=dft,
                nn=nn,bo_layer=bo_layer,
                batch_size=batch,
                pkl=True)
    rn.initialize()
    rn.session(learning_rate=3.0e-4,method='AdamOptimizer') 

    if nn:
       bdlit = ['bop','bop_si','bop_pi','bop_pp',
                'F',
                'bosi','bopi','bopp',
                'powb','expb','sieng','EBD']
    else:
       bdlit = ['bop','bop_si','bop_pi','bop_pp',
                'f_1','f_2','f_3','f_4','f_5',
                'bosi','bopi','bopp',
                'powb','expb','sieng','EBD']

    if debd:
       bonds = rn.bonds
       for b in bonds:
           v  = v_ +'_'+ b 

           grad = rn.get_gradient(rn.Loss,rn.p[v]) 
           text_ = '-  the gradient of Loss/%s is ' %v
           logger.info( text_+str(grad))

           if grad is None:
              continue
           if not np.isnan(grad):
              continue

           if rn.nbd[b]>0:
              grad = rn.get_gradient(rn.__dict__['EBD'][b],rn.p[v]) 
              logger.info('-  the gradient of %s/%s is: %s' %('EBD'+'_'+b,v,str(grad))) 
              if not grad is None:
                 if np.isnan(grad):
                    for l in bdlit:
                        grad = rn.get_gradient(rn.__dict__[l][bd],rn.p[v]) 
                        logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+b,v,str(grad))) 
    v  = v_ +'_'+ bd 
    if deba:
       sl = ['EL','EOV','EUN']
       alist = {'EL':['Delta_lp','Delta_e','D','explp'],
                'EOV':['Delta_lpcorr','so','otrm1','otrm2'],
                'EUN':[]}
       for sp in rn.spec:
           for l in sl:
               if sp in rn.__dict__[l]:
                  grad = rn.get_gradient(rn.__dict__[l][sp],rn.p[v]) 
                  logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+sp,v,str(grad))) 

                  if not grad is None:
                     if np.isnan(grad):
                        for al in alist[l]:
                            grad = rn.get_gradient(rn.__dict__[al][sp],rn.p[v]) 
                            logger.info('-  the gradient of %s/%s is: %s' %(al+'_'+sp,v,str(grad))) 

    if deang:
       al = ['EANG','EPEN','ETC']
       for ang in rn.angs:
           # v  = 'val1'+'_'+ang  
           if rn.nang[ang]>0:
              for l in al:
                  grad = rn.get_gradient(rn.__dict__[l][ang],rn.p[v]) 
                  logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+ang,v,str(grad))) 

    tl = ['ETOR','Efcon']
    for tor in rn.tors:
        # v  = 'tor2' # +'_'+tor  
        if rn.ntor[tor]>0:
           for l in tl:
               grad = rn.get_gradient(rn.__dict__[l][tor],rn.p[v]) 
               logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+tor,v,str(grad))) 
    rn.sess.close()


def gradbm(direcs,v='bo5',bd='H-H',dft='siesta',batch=100):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
    '''
    v_ = v
    v  = v+'_'+bd  
    print('-  grading ... ...')

    rn = ReaxFF(libfile='ffield',direcs=direcs,
                dft=dft,
                batch_size=batch,
                pkl=True)
    rn.initialize()
    rn.session(learning_rate=3.0e-4,method='AdamOptimizer') 

    bdlit = ['bop','bop_si','bop_pi','bop_pp',
             'f_1','f_2','f_3','f_4','f_5',
             'bosi','bopi','bopp',
             'powb','expb','sieng','EBD']

    gl = rn.get_gradient(rn.Loss,rn.p[v]) 
    logger.info('-  the gradient of Loss/%s is %f.' %(v,gl))

    bonds = rn.bonds
    bonds = [bd]
    for bd in bonds:
        v  = v_ +'_'+ bd  
        if rn.nbd[bd]>0:
           grad = rn.get_gradient(rn.__dict__['EBD'][bd],rn.p[v]) 
           logger.info('-  the gradient of %s/%s is: %s' %('EBD'+'_'+bd,v,str(grad))) 
           if not grad is None:
              if np.isnan(grad):
                 for l in bdlit:
                     grad = rn.get_gradient(rn.__dict__[l][bd],rn.p[v]) 
                     logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+bd,v,str(grad))) 

    ml = ['ebond','elone','eover','eunder','eang','epen','tconj','etor','efcon','evdw','ehb']
    for bd in bonds:
        v  = v_ +'_'+ bd  
        if rn.nbd[bd]>0:
           for m in direcs:
               mol = m
               grad = rn.get_gradient(rn.__dict__['loss'][mol],rn.p[v]) 
               logger.info('-  the gradient of %s_%s/%s is: %s' %('loss',mol,v,str(grad)))

               if not grad is None:
                  if np.isnan(grad):
                     for l in ml:
                         grad = rn.get_gradient(rn.__dict__[l][mol],rn.p[v]) 
                         logger.info('-  the gradient of %s_%s/%s is: %s' %(l,mol,v,str(grad)))
    rn.sess.close()


def grada(direcs,v='valang_O',dft='siesta',batch=100):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
    '''
    print('-  grading ... ... ')
    for m in direcs:
        mol = m

    rn = ReaxFF(libfile='ffield',direcs=direcs,
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
    anglit = ['EANG']
    angs = rn.angs
    # angs = ['C-C-C']
    for ang in angs:
        if rn.nang[ang]>0:
           for l in anglit:
               grad = rn.get_gradient(rn.__dict__[l][ang],rn.p[v]) 
               print('- the gradient of %s/%s is: ' %(l+'_'+ang,v),grad) 
    rn.sess.close()


def gradt(v='bo2_C-N',direcs={},
          torlit=['f_10','f_11','expv2','ETOR','Efcon'],
          batch=100):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
        torlit = ['f_10','f_11','expv2','ETOR','Efcon']
    '''
    print('-  grading ... ... ')
    rn = ReaxFF(libfile='ffield',
    	          direcs=direcs,
                dft='siesta',
                batch_size=batch,
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

    i = 0
    for tor in rn.tors:
        if rn.ntor[tor]>0:
           # if tor=='C-H-H-C':
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

    plt.plot(e1,label=r'$ReaxFF$', color='red', linewidth=2, linestyle='-.')
    plt.plot(e2,label=r'$ReaxFF$', color='blue', linewidth=2, linestyle='--')

    plt.legend()
    plt.savefig('%s_%s.eps' %(en,mol)) 
    plt.close()


def debug_plot():
    mol = 'ch4'
    direcs={mol:'/home/feng/siesta/train/ch4'}

    batch= {'others':300}
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
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
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
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


def get_v(direcs={'ch4':'/home/gfeng/siesta/train/ch4'},batch=1):
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                optword='nocoul',
                batch_size=batch,
                sort=False,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
  
    p   = rn.p_
    bo  = rn.get_value(rn.bo)
    bo0 = rn.get_value(rn.bo0)
    D   = rn.get_value(rn.Deltap)
    Dlp = rn.get_value(rn.Delta_lpcorr)
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
    df4    = rn.get_value(rn.df4)
    df5    = rn.get_value(rn.df5)

    dboci= rn.get_value(rn.Di_boc)
    dbocj= rn.get_value(rn.Dj_boc)
    Dp   = rn.get_value(rn.Dp)
    BOP  = rn.get_value(rn.BOP)
    print('-  shape of BOP: ',BOP.shape)

    nbd = rn.nbd 
    bdlab = rn.lk.bdlab
    bonds = rn.bonds

    nang = rn.nang
    anglab = rn.lk.anglab
    angs = rn.angs

    fbo = open('bo.txt','w')

    for bd in bonds:
        if nbd[bd]>0:
           print('\n-  bd: %s \n' %bd,file=fbo)
           for nb in range(nbd[bd]):
               atomi,atomj = bd.split('-')
               print('- ',bd,
                    'r:',rbd[bd][nb][0],
                    'BOp:',bop[bd][nb][0],
                    'F:',f[bd][nb][0],
                    'F_11:',f11[bd][nb][0],
                    'F_12:',f12[bd][nb][0],
                    'F_4:',f4[bd][nb][0],
                    'F_5:',f5[bd][nb][0],
                    'Di:',dboci[bd][nb][0]+p['valboc_'+atomi],
                    'Dj:',dbocj[bd][nb][0]+p['valboc_'+atomj],
                    'BO:',bo0[bd][nb][0],file=fbo)
    fbo.close()

    fa = open('a.txt','w')
    for a in rn.spec:
        if rn.nsp[a]>0:
           print('\n-  specices: %s \n' %a,file=fa)
           for na in range(rn.nsp[a]):
               print('- ',a,
                    'Dlpc:',Dlp[a][na][0], # +p['val_'+a],
                    file=fa)
    fa.close()

    ff  = open('f.txt','w')
    f1   = rn.get_value(rn.f_1)
    f2   = rn.get_value(rn.f_2)
    f3   = rn.get_value(rn.f_3)
    f4   = rn.get_value(rn.f_4)
    f5   = rn.get_value(rn.f_5)
    f4r   = rn.get_value(rn.f4r)
    f5r   = rn.get_value(rn.f5r)

    dexpf3 = rn.get_value(rn.dexpf3)
    dexpf3t= rn.get_value(rn.dexpf3t)
    f3log  = rn.get_value(rn.f3log)

    for bd in bonds:
        if nbd[bd]>0:
           print('\n-  bd: %s \n' %bd,file=ff)
           for nb in range(nbd[bd]):
               print('- ',bd,
                    'r:',rbd[bd][nb][0],
                  'f_1:',f1[bd][nb][0],
                  'f_2:',f2[bd][nb][0],
                  'f_3:',f3[bd][nb][0],
                  'dexpf3:',dexpf3[bd][nb][0],
                  'dexpf3t:',dexpf3t[bd][nb][0],
                  'f_4:',f4[bd][nb][0],
                  'f_5:',f5[bd][nb][0],
                  'df4:',df4[bd][nb][0],
                  'df5:',df5[bd][nb][0],
                  'f_4r:',f4r[bd][nb][0],
                  'f_5r:',f5r[bd][nb][0],
                  file=ff)
    ff.close()
    rn.sess.close()    
    del rn

    # for i,d in enumerate(D):
    #     print('-  atom:',i,d[0],Dlp[i])
    # for i,bop in enumerate(BOP):
    #     print('-  atom:',i,bop)


def compare_bo(dire):
    # mol = 'mol'
    # direcs={mol:dire}

    mol = 'mol'
    direcs={mol:dire}
    # direcs={'nm13':'/home/gfeng/cpmd/train/nmr/nm13',
    #         'nm4':'/home/gfeng/cpmd/train/nmr/nm2004',
    #         'nm11':'/home/gfeng/cpmd/train/nmr/nm11'}

    batch= {'others':10}
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
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


def get_spv(direcs={'ch4':'/home/gfeng/siesta/train/ch4'},batch=50):
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft='siesta',
                optword='all',
                batch_size=batch,
                sort=False,
                pkl=True,
                interactive=True)
    rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
  
    # ebond = rn.get_value(rn.ebond[mol])
    diffa  = rn.get_value(rn.diffa)
    diffb  = rn.get_value(rn.diffb)
    diffe  = rn.get_value(rn.diffe)
    bosip  = rn.get_value(rn.bosip)
 
    nbd    = rn.nbd 
    bdlab  = rn.lk.bdlab
    bonds  = rn.bonds

    fspv   = open('spv.txt','w')

    for bd in bonds:
        if nbd[bd]>0:
           print('\n-  spv: %s \n' %bd,file=fspv)
           # for nb in range(nbd[bd]):
           print('- ',bd,
                 'a:',diffa[bd],
                 'b:',diffb[bd],
                 'e:',diffe[bd],
                 's:',bosip[bd],file=fspv)
    fspv.close()

    rn.sess.close()    
    del rn


def compare_eb(dire):
    system('rm *.pkl')
    mol = 'dp'
    # direcs={mol:dire}
    direcs={'dp':'/home/gfeng/cpmd/train/nmr/dop'}

    batch= {'others':10}
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
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
    rn = ReaxFF(libfile='ffield',direcs=direcs, 
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

    rn = ReaxFF(libfile='ffield',direcs=direcs, 
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


def ecoul(direcs=None,batch=1,
          dft='siesta'):
    for m in direcs:
        mol = m

    rn = ReaxFF(libfile='ffield',direcs=direcs, 
                dft=dft,
                rc_scale='none',
                optword='all',
                batch_size=batch,
                sort=False,
                pkl=True,
                interactive=True)
    molecule = rn.initialize()
    rn.session(learning_rate=1.0-4,method='AdamOptimizer')
    # rn.plot()
    vlab = rn.lk.vlab 
    
    ecoul = rn.get_value(rn.ecoul)
    ecou  = rn.get_value(rn.ECOU)
    rv    = rn.get_value(rn.rv)
    tpv   = rn.get_value(rn.tpv)
    qij   = rn.get_value(rn.qij)
    rth   = rn.get_value(rn.rth)


    bonds = rn.bonds
    nvb   = rn.nvb

    for vb in bonds:
        if nvb[vb]>0:
           for i,r in enumerate(rv[vb]):
               vl    = vlab[vb][i]
               vi,vj = vl[1],vl[2]

               found = False
               for j in range(27):
                   R_ = np.array(molecule[mol].R_)
                   tp = np.array(molecule[mol].tp_)
                   qij_= np.array(molecule[mol].qij)
                   ecoul_= np.array(molecule[mol].ecoul_)
                   rth_= np.array(molecule[mol].rth_)
    
                   if abs(r[0]-R_[j][0][vi][vj])<0.000001:
                      nj,i_,j_ = j,vi,vj
                      found = True
                   elif abs(r[0] - R_[j][0][vj][vi])<0.000001:
                      nj,i_,j_ = j,vj,vi
                      found = True

               if found:
                  print('-  GULP:',i_,j_, r[0],tpv[vb][i][0],qij[vb][i][0],
                                    rth[vb][i][0],ecou[vb][i][0])
                  print('-    IR:',i_,j_,R_[nj][0][i_][j_],tp[nj][0][i_][j_],qij_[0][i_][j_],
                                   rth_[nj][0][i_][j_],ecoul_[nj][0][i_][j_],ecoul_[nj][0][j_][i_],
                                   R_[nj][0][j_][i_])
               else:
                  print('- not found',vb,i,r[0])
    Ecoul_ = molecule[mol].ecoul


    print('- energys:',ecoul[mol],Ecoul_)
    plt.figure()
    plt.ylabel('Energies')
    plt.xlabel('Step')

    plt.plot(ecoul[mol],label=r'$ReaxFF$', color='red', linewidth=2, linestyle='-.')
    plt.plot(Ecoul_,label=r'$GULP$', color='blue', linewidth=2, linestyle='--')

    plt.legend()
    plt.savefig('energies_%s.eps' %mol) 
    plt.close()

