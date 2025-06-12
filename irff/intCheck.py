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

def check_tors(spec,torp):
    tors = []   ### check torsion parameter  
    for spi in spec:
        for spj in spec:
            for spk in spec:
                for spl in spec:
                     tor = spi+'-'+spj+'-'+spk+'-'+spl
                     torr= spl+'-'+spk+'-'+spj+'-'+spi
                     tor1= spi+'-'+spk+'-'+spj+'-'+spl
                     tor2= spl+'-'+spj+'-'+spk+'-'+spi
                     tor3= 'X-'+spj+'-'+spk+'-X'
                     tor4= 'X-'+spk+'-'+spj+'-X'
                     if (tor in torp) or (torr in torp) or (tor1 in torp) \
                        or (tor2 in torp) or (tor3 in torp) or (tor4 in torp):
                        if (not tor in tors) and (not torr in tors):
                           if tor in torp:
                              tors.append(tor)
                           elif torr in torp:
                              tors.append(torr)
                           else:
                              tors.append(tor)
    return tors

def value(p,p_,key):
    fc = open('intcheck.log','a')
    fc.write('-  {:s} change from {:f} to {:f}\n'.format(key,p_,p))
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
      # punit    = ('Desi','Depi','Depp','lp2','ovun5','val1',
      #             'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb') 
      # unit     = 4.3364432032e-2
      self.clip  = {'acut':(0.0000999,0.05),'hbtol':(0.0000999,0.05),'cutoff':(0.0000999,0.012),
                    'ovun5':(0.0,900.0),'ovun1':(0.0,2.0),'gammaw':(0.001,16.0),
                    'vdw1':(0.001,9.0),'gamma':(0.001,6.0),'tor1':(-9.001,-0.0001),
                    'pen1':(-60.0,196.0),'Devdw':(0.0001,1.0),
                    'Dehb':(-3.0,3.0),'Desi':(0.10,990.0),
                    'ovun2':(-50.001,0.0),'tor3':(0.0,10.0),'val7':(0.0,10.0),
                    'tor2':(0.0,30.0),'tor4':(0.0,30.0),
                    'coa1':(-99.0,0.0),'cot1':(-99.0,0.0),
                    'V1':(-99.0,199.0),'V2':(-99.0,199.0),
                    'V3':(-99.0,199.0),
                    'bo1':(-0.25,-0.005),'bo2':(-0.25,-0.005),'bo3':(-0.25,-0.005),
                    'bo2':(1.0,16.0),'bo4':(1.0,16.0),'bo6':(1.0,16.0),
                    'alfa':(1.0,16.0),'lp1':(1.0,50.0),'lp2':(0.0,999.0),
                    'coa2':(-10.0,30.0),'coa2':(-10.0,30.0),'rohb':(1.5,3.6),
                    'val1':(0.0,300.0),'val2':(0.0,8.0),'val3':(0.0,8.0),
                    'val5':(0.0,20.0),'val9':(0.0,6.0) }
      if clip:
         for key in clip:
             k = key.split('_')
             if len(k)>1:
                if k[0] in ['Devdw','rvdw','alfa','rosi','ropi','ropp']:
                   k_ = k[1].split('-')
                   if len(k_)==1:
                      self.clip[k[0]+'_'+k_[0]+'-'+k_[0]] = clip[key]
         self.clip.update(clip)

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
         v = suggestion[key]
      elif key_ in suggestion:
         v = suggestion[key_]
      else:
         v = 0.0
      return v
      
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
      # unit = 4.3364432032e-2
      cons = []
      for key in p:
          k = key.split('_')[0]
          if key in self.clip: 
             if p[key]>self.clip[key][1]:
                p[key] = value(self.clip[key][1],p[key],key)
                cons.append(key)
             if p[key]<self.clip[key][0]:
                p[key] = value(self.clip[key][0],p[key],key)
                cons.append(key)
          elif k in self.clip: 
             if p[key]>self.clip[k][1]:
                p[key] = value(self.clip[k][1],p[key],key)
                cons.append(key)
             if p[key]<self.clip[k][0]:
                p[key] = value(self.clip[k][0],p[key],key)
                cons.append(key)

          if k in ['val1','val7']: 
             pr = key.split('_')[1]
             ang= pr.split('-')
             if  ang[1]=='H':
                p[key] = value(0.0,p[key],key)
          if k == 'valboc': 
             p[key] = value(p['valang_'+key.split('_')[1]],p[key],key)
          if k in ['V1','V2','V3']: 
             pr = key.split('_')[1]
             tor= pr.split('-')
             if tor[1]=='H' or tor[2]=='H':
                p[key] = value(0.0,p[key],key)

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
      return p,m,cons
  
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


