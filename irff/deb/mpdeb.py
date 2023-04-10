#!/usr/bin/env python
import numpy as np
from os import system, getcwd, chdir,listdir,environ
from irff.mpnn import MPNN
from irff.data.ColData import ColData
from irff.reax import logger

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logger(flog='debug.log')


def gradb(dataset,v='bo5',bd='H-H',
          nn=True,
          debd=False,deba=True,deang=True,
          batch=50):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
    '''
    v_ = v
    v  = v+'_'+bd  
    print('-  grading ... ...')
    
    rn = MPNN(libfile='ffield.json',
              dataset=dataset,            
              weight={'hmx-r':20.0,'others':2.0},
              optword='nocoul',
              regularize_mf=1,regularize_be=1,regularize_bias=1,
              lambda_reg=0.005,lambda_bd=1000.0,lambda_me=0.01,
              mf_layer=[9,1],be_layer=[9,1],
              EnergyFunction=1,MessageFunction=3,
              batch_size=batch,
              fixrcbo=False,
              convergence=0.99) 
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
       alist = {'EL':['Delta_lp','Delta_e','explp'],
                'EOV':['Delta_lpcorr','Delta_lp','nlp','so','otrm1','otrm2'],
                'EUN':['expeu1','expeu3','Delta_lpcorr','Delta_lp','Delta_e','explp','nlp']}
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


if __name__ == '__main__':
   getdata = ColData()
   dataset = {}
   strucs = ['al64',
            'AlO', 
            'o22']

   batchs = {'others':50}

   for mol in strucs:
      b = batchs[mol] if mol in batchs else batchs['others']
      trajs = getdata(label=mol,batch=b)
      dataset.update(trajs)

   gradb(dataset,v='bo1',bd='O-O',
         debd=False,deba=True,deang=True,batch=50)


