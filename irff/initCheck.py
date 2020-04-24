from __future__ import print_function



def init_chk_bonds(p_,pns,bonds):
    for pn in pns:
        for bd in bonds:
            pn_ = pn + '_' + bd
            if not pn_ in p_:
               print('-  warning: parameter %s is not found in lib ...' %pn_)
               p_[pn_] = 0.010
    return p_


def init_bonds(p_):
    bonds,offd,angs,torp,hbs = [],[],[],[],[]
    for key in p_:
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           if len(kk)==2:
              if kk[0]!=kk[1]:
                 offd.append(k[1])
        elif k[0]=='theta0':
           angs.append(k[1])
        elif k[0]=='tor1':
           torp.append(k[1])
        elif k[0]=='rohb':
           hbs.append(k[1])
    return bonds,offd,angs,torp,hbs


def value(p,key):
    fc = open('check.log','a')
    fc.write('-  %s change to %f\n' %(key,p))
    fc.close()
    return p


class Init_Check(object):
  def __init__(self,re=None,nanv=None):
      self.re = re
      self.nanv=nanv
      with open('check.log','w') as fc:
           fc.write('Values have changed:\n')
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
      with open('check.log','a') as fc:
           fc.write('- to avoid nan error, %s is change to %f \n' %(kmax,p[kmax]))
      # 'lp1':-2.0,
      return p


  def check(self,p):
      print('-  check parameters if reasonable ...')
      unit = 4.3364432032e-2
      for key in p:
          k = key.split('_')[0]
          if key == 'boc1':
             if p[key]>50.0:
                p[key] = value(40.0,key)
          if key == 'boc2':
             if p[key]<=0.05:
                p[key]= value(1.0,key)
          if k == 'boc3':
             if p[key]<=0.001:
                p[key]= value(0.001,key)
             if p[key]>=40.0:
                p[key]= value(20.0,key)  

          if key == 'lp1':
             if p[key]>=28.0:
                p[key] = value(20.0,key)
             if p[key]<=12.0:
                p[key] = value(16.0,key)

          if k=='lp2':
             sp = key.split('_')[1]
             if p['val_'+sp]!=p['vale_'+sp]:
                if p[key] <= 0.001:
                   p[key] = value(4.00,key)
             if p[key]>=60.0:
                p[key] = value(20.0,key)

          # if k=='coa1': 
          #    if p[key]<=-90.0:
          #       p[key] = value(0.0,key)

          if k=='val1': 
             if p[key]>=500.0:
                p[key] = value(100.0,key)
          if k=='val2': 
             if p[key]>=30.0:
                p[key] = value(15.0,key)

          if k in ['V1','V2','V3']: 
             if p[key]>=500.0:
                p[key] = value(100.0,key)
             # if p[key]<=-90.0:
             #    p[key] = value(0.0,key)

          if k=='val4': 
             if p[key]>=30.0:
                p[key] = value(15.0,key)
          if k=='val7': 
             if p[key]>=30.0:
                p[key] = value(15.0,key)

          if key in ['pen3','pen4']:
             if p[key]==0.0:
                p[key] = value(1.0,key)
             if p[key]>16.0:
                p[key] = value(8.0,key)

          if k=='be2': 
             # if p[key]<=0.01:
             #    p[key] = value(1.00,key)
             if p[key]>=10.0:
                p[key] = value(3.0,key)
          if k=='be1': 
             if p[key]<=-10.0:
                p[key] = value(-3.00,key)
             if p[key]>=10.0:
                p[key] = value(3.0,key)

          if k=='alfa': 
             if p[key]>=40.0:
                p[key] = value(4.0,key)
             if p[key]<=0.01:
                p[key] = value(2.0,key)

          if k=='gammaw': 
             if p[key]>=40.0:
                p[key] = value(4.0,key)
             if p[key]<=0.01:
                p[key] = value(2.0,key)

          if k=='Desi': 
             if p[key]<100.0:
                p[key]= value(100.0,key)
             if p[key]>=400.0:
                p[key]= value(200.0,key)
          if k=='Depi': 
             if p[key]==0.0:
                p[key]= value(0.1,key)
             if p[key]>=400.0:
                p[key]= value(200.0,key)
          if k=='Depp': 
             if p[key]==0.0:
                p[key]= value(0.1,key)
             if p[key]>=400.0:
                p[key]= value(200.0,key)
          if k=='Devdw': 
             if p[key]>=5.0:
                p[key]= value(1.8,key)
             if p[key]<=0.001:
                p[key]= value(0.005,key)
          if k=='vdw1': 
             if p[key]>=15.0:
                p[key]= value(1.8,key)
          if k=='ovun8': 
             if p[key]>=15.0:
                p[key] = value(10.0,key)
          if k=='val7': 
             if p[key]>=20.0:
                p[key] = value(18.0,key)
          if k=='ovun1': 
             if p[key]<=0.10:
                p[key]= value(1.0,key)
          if k=='ovun2': 
             if p[key]>=0.0:
                p[key]= value(-3.0,key)
          if k=='ovun4': 
             if p[key]<=0.001:
                p[key]= value(1.0,key)
             if p[key]>=20.00:
                p[key]= value(2.0,key)
          if k=='ovun5': 
             if p[key]>=699.0:
                p[key]= value(699.0,key)
             if p[key]<=0.001:
                p[key]= value(6.0,key)
          if k in ['bo2','bo4','bo6']:
             if p[key]>=40.0:
                p[key] = value(30.0,key)
          if k=='theta0':
             if p[key]>=200.0:
                p[key] = value(109.09,key)
          if k=='pen1':
             if p[key]>=200.0:
                p[key] = value(100.0,key) 
             if p[key]<=-69.0:
                p[key] = value(-1.0,key)
          if key == 'cot2':
             if p[key]<0.001:
                p[key] = value(2.0,key)
          if key == 'tor1':
             if p[key]<-40.0:
                p[key] = value(-12.0,key)
          if key == 'tor2':
             if p[key]>15.0:
                p[key] = value(5.0,key)
          if key == 'tor4':
             if p[key]>11.0:
                p[key] = value(11.0,key)
          if key == 'tor3':
             if p[key]>15.0:
                p[key] = value(5.0,key)

          if k=='ropi': 
             bd = key.split('_')[1]
             b  = bd.split('-')
             ofd= bd
             if p[key]<=0.0:
                p[key]= value(0.98*p['rosi_'+ofd],key)
                if len(b)==2:
                   p['bo3_'+bd]= value(-10.0,'bo3_'+bd)
                   p['bo4_'+bd]= value(0.0,'bo4_'+bd)

          elif k=='ropp': 
             bd = key.split('_')[1]
             b  = bd.split('-')
             ofd= bd
             if p[key]<=0.0:
                p[key]= value(0.9*p['rosi_'+ofd],key)
                if len(b)==2:
                   p['bo5_'+bd]= value(-10.0,'bo5_'+bd)
                   p['bo6_'+bd]= value(0.0,'bo6_'+bd)

          # if not self.re is None:
          #    if k in ['rosi','ropi','ropp']:
          #       bd = key.split('_')[1]
          #       b  = bd.split('-')
          #       ofd= bd
          #       if len(b)==1:
          #          bd = b[0] +'-' +b[0]
          #          ofd= b[0] 

          #       if k=='rosi': 
          #          if p[key]>=1.30*self.re[bd]:
          #             p[key]= value(1.20*self.re[bd],key)
          #          if p[key]<0.750*self.re[bd]:
          #             p[key]= value(0.850*self.re[bd],key)
          #       elif k=='ropi': 
          #          if p[key]>=0.98*p['rosi_'+ofd]:
          #             p[key]= value(0.98*p['rosi_'+ofd],key)
          #          if p[key]<0.55*p['rosi_'+ofd]:
          #             p[key]= value(0.55*p['rosi_'+ofd],key)
          #       elif k=='ropp': 
          #          if p[key]>=0.98*p['ropi_'+ofd]:
          #             p[key]= value(0.98*p['ropi_'+ofd],key)
          #          if p[key]<0.55*p['ropi_'+ofd]:
          #             p[key]= value(0.55*p['ropi_'+ofd],key)
      return p


  def close(self):
      self.re   = None
      self.nanv = None


