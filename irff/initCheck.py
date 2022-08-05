import numpy as np
from pydoc import cli


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
    fc = open('clip.log','a')
    fc.write('-  %s change to %f\n' %(key,p))
    fc.close()
    return p


class Init_Check(object):
  def __init__(self,re=None,nanv=None,clip=None):
      ''' Etcon not continuous --> coa1 
          Eang  not continuous --> val1 
      '''
      self.re = re
      self.nanv=nanv
      self.clip=clip
      with open('clip.log','w') as fc:
           fc.write('Values have clipped:\n')
           fc.write('\n')

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
      with open('clip.log','a') as fc:
           fc.write('- to avoid nan error, %s is change to %f \n' %(kmax,p[kmax]))
      # 'lp1':-2.0,
      return p

  def check(self,p,m=None,resetDeadNeuron=False):
      print('-  check parameters if reasonable ...')
      unit = 4.3364432032e-2
      if self.clip is None:
         return p,m
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
                
      if not m is None and resetDeadNeuron:
         m = self.check_m(m)


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


