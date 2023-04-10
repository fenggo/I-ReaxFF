import numpy as np


def init_chk_bonds(p_,pns,bonds):
    for pn in pns:
        for bd in bonds:
            pn_ = pn + '_' + bd
            if not pn_ in p_:
               print('-  warning: parameter %s is not found in lib ...' %pn_)
               p_[pn_] = 0.010
    return p_

def init_bonds(p_):
    spec,bonds,offd,angs,torp,hbs = [],[],[],[],[],[]
    for key in p_:
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           if len(kk)==2:
              if kk[0]!=kk[1]:
                 offd.append(k[1])
           elif len(kk)==1:
              spec.append(k[1])
        elif k[0]=='theta0':
           angs.append(k[1])
        elif k[0]=='tor1':
           torp.append(k[1])
        elif k[0]=='rohb':
           hbs.append(k[1])
    return spec,bonds,offd,angs,torp,hbs

def value(p,key):
    fc = open('intcheck.log','a')
    fc.write('-  %s change to %f\n' %(key,p))
    fc.close()
    return p

suggestion_ = {'tor1':np.random.normal(loc=-2.0, scale=0.01),
               'V1':np.random.normal(loc=0.0, scale=0.1),
               'V2':np.random.normal(loc=0.0, scale=0.1),
               'V3':np.random.normal(loc=0.0, scale=0.1),
               'cot1':0.0}

class Intelligent_Check(object):
  def __init__(self,re=None,nanv=None,clip=None,spec=None,bonds=None,
               offd=None,angs=None,tors=None,ptor=None):
      ''' Etcon not continuous --> coa1 
          Eang  not continuous --> val1 
      '''
      self.re = re
      self.nanv=nanv
      self.clip=clip
      self.spec  = spec
      self.bonds = bonds
      self.offd  = offd
      self.angs  = angs
      self.tors  = tors
      self.ptor  = ptor

      with open('intcheck.log','w') as fc:
           fc.write('Values have clipped:\n')
           fc.write('\n')

  def suggest_parameter(self,key,suggestion=None):
      ''' return suggedted parameters as hyper-parameters as used by ReaxFF-nn '''
      if suggestion is None:
         suggestion = suggestion_
      key_  = key.split('_')[0]

      if key in suggestion:
         value = suggestion[key]
      elif key_ in suggestion:
         value = suggestion[key_]
      else:
         value = 0.0
      return value
      
  def auto(self,p):
      kmax = None
      vmax = None

      for key in self.nanv:
          # k = key.split('_')
          if key=='lp1':
             if p[key]>30.0:
                p[key]  = p[key] + self.nanv[key]
             elif p[key]<10.0:
                p[key]  = p[key] - self.nanv[key]
             continue

          if kmax is None:
             kmax = key
             vmax = p[key]
          else:
             if p[key]>vmax:
                kmax  = key
                vmax  = p[key]

      p[kmax]  = p[kmax] + self.nanv[kmax]
      with open('intcheck.log','a') as fc:
           fc.write('- to avoid nan error, %s is change to %f \n' %(kmax,p[kmax]))
      # 'lp1':-2.0,
      return p

  def check(self,p,m=None,resetDeadNeuron=False):
      print('-  check parameters if reasonable ...')
      unit = 4.3364432032e-2
      
      if self.clip is None:
         self.clip = {}

      for key in p:
          k = key.split('_')[0]
          if k in self.clip: 
             if p[key]>self.clip[k][1]:
                p[key] = value(self.clip[k][1],key)
             if p[key]<self.clip[k][0]:
                p[key] = value(self.clip[k][0],key)

          if k in ['val1','val7']: 
             pr = key.split('_')[1]
             ang= pr.split('-')
             if  ang[1]=='H':
                p[key] = value(0.0,key)
 
          if k in ['V1','V2','V3']: 
             pr = key.split('_')[1]
             tor= pr.split('-')
             if tor[1]=='H' or tor[2]=='H':
                p[key] = value(0.0,key)

      for key in self.ptor:               ## check four-body parameters
          for tor in self.tors:
              torsion = key+'_'+tor
              if torsion not in p:
                 p[torsion] = self.suggest_parameter(key)
                 print('-  fourbody parameter {:s} not in ffield.json, using suggested value.'.format(torsion))
                 with open('intcheck.log','a') as fc:
                      fc.write('-  fourbody parameter {:s} set to {:f}\n'.format(torsion,p[torsion]))

      if not m is None and resetDeadNeuron:
         m = self.check_m(m)

      self.p = p
      return p,m
  
  def check_m(self,m):
      ''' reset the dead neuron '''
      print('-  check neurons that are zero ...')
      for key in m:
          k = key.split('_')[0]  
          if k in  ['f1w','few','fvw']: #'f1b' 'feb' 'fvb' 
             for m_,n in enumerate(m[key]):
                 if np.mean(m[key])<0.1 and np.var(m[key])<0.1:
                    print('-  The variance of m[{:s}] is zero, regenerate with Gaussian distribution ...'.format(key))
                    s = np.array(m[key][m_]).shape
                    m[key][m_] = np.random.normal(0.0,1.0,s).astype(np.float32)  # 高斯分布
          elif k in  ['f1wi','fewi','fvwi','f1wo','fewo','fvwo']:
             if np.mean(m[key])<0.1 and np.var(m[key])<0.1:
                print('-  The variance of m[{:s}] is zero, regenerate with Gaussian distribution ...'.format(key))
                s = np.array(m[key]).shape
                m[key] = np.random.normal(0.0,1.0,s).astype(np.float32)         # 高斯分布

          if k in  ['f1b','feb','fvb','f1bi','f1bo','febi','febo','fvb','fvbo']: #'f1b' 'feb' 'fvb'
             for i,mmm in enumerate(m[key]):
                if isinstance(mmm, list):
                   for j,mm in enumerate(mmm):
                         if isinstance(mm, list):
                            for l,m_ in enumerate(mm):
                               if m[key][i][j][l]>= 5.0:
                                  m[key][i][j][l] = 2.0 
                               if m[key][i][j][l]<=-5.0:
                                  m[key][i][j][l] =-2.0 
                         else:
                            if m[key][i][j]>= 5.0:
                               m[key][i][j] = 2.0 
                            if m[key][i][j]<=-5.0:
                               m[key][i][j] =-2.0 
                else:
                   if m[key][i]>= 5.0:
                      m[key][i] = 2.0 
                   if m[key][i]<=-5.0:
                      m[key][i] =-2.0 
      return m 

  def close(self):
      self.re   = None
      self.nanv = None


