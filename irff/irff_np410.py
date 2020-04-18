from __future__ import print_function
from .reaxfflib import read_lib,write_lib
from ase import Atoms
from ase.io import read,write
import numpy as np
from .qeq import qeq
from .setRcut import setRcut
import json as js
from ase.calculators.calculator import Calculator, all_changes
# tf.compat.v1.enable_eager_execution()


try:
   from .neighbor import get_neighbors,get_pangle,get_ptorsion,get_phb
except ImportError:
   from .neighbors import get_neighbors,get_pangle,get_ptorsion,get_phb


def rtaper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r<rmin,1.0,0.0) # r > rmax then 1 else 0

    ok    = np.logical_and(r<=rmax,r>rmin)      # rmin < r < rmax  = r else 0
    r2    = np.where(ok,r,0.0)
    r20   = np.where(ok,1.0,0.0)

    rterm = np.divide(1.0,np.power(rmax-rmin,3))
    rm    = rmax*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmin*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3


def taper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r>rmax,1.0,0.0) # r > rmax then 1 else 0

    ok    = np.logical_and(r<=rmax,r>rmin)      # rmin < r < rmax  = r else 0
    r2    = np.where(ok,r,0.0)
    r20   = np.where(ok,1.0,0.0)

    rterm = np.divide(1.0,np.power(rmin-rmax,3)+0.0000001)
    rm    = rmin*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmax*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3
    

def fvr(x):
    xi  = np.expand_dims(x,axis=0)
    xj  = np.expand_dims(x,axis=1) 
    vr  = xj - xi
    return vr


def fr(vr):
    R   = np.sqrt(np.sum(vr*vr,axis=2))
    return R


def sigmoid(x):
    s = 1.0/(1.0+np.exp(-x))
    return s


def relu(x):
    return np.where(x>0.0,x,0.0)  



class IRFF_NP(object):
  '''Intelligent Machine-Learning ASE calculator'''
  name = "IRFF-NP"
  implemented_properties = ["energy", "forces"] # , "stress"]
  def __init__(self,atoms=None,
               libfile='ffield',
               rcut=None,rcuta=None,
               vdwcut=10.0,
               nn=False,
               massages=1,
               hbshort=6.75,hblong=7.5,
               label="IRFF", **kwargs):
      # Calculator.__init__(self,label=label, **kwargs)
      self.atoms        = atoms
      self.cell         = atoms.get_cell()
      self.atom_name    = self.atoms.get_chemical_symbols()
      self.natom        = len(self.atom_name)
      self.spec         = []
      self.nn           = nn
      self.massages     = massages + 1
      self.safety_value = 0.000000001

      for sp in self.atom_name:
          if sp not in self.spec:
             self.spec.append(sp)

      self.p_ang  = ['theta0','val1','val2','coa1','val7','val4','pen1'] 
      self.p_hb   = ['rohb','Dehb','hb1','hb2']
      self.p_tor  = ['V1','V2','V3','tor1','cot1']  

      if libfile.endswith('.json'):
         lf = open(libfile,'r')
         j = js.load(lf)
         self.p  = j['p']
         m       = j['m']
         self.zpe_= j['zpe']
         self.massages = j['massages']
         if 'bo_layer' in j:
            self.bo_layer = j['bo_layer']
         else:
            self.bo_layer = None
         lf.close()
         self.init_bonds()
      else:
         self.p,zpe_,self.spec,self.bonds,self.offd,self.Angs,self.torp,self.Hbs= \
                       read_lib(libfile=libfile,zpe=True)
         m                = None
         self.bo_layer    = None

      self.torp      = self.checkTors(self.torp)
      self.check_tors(self.p_tor)
      self.botol     = 0.01*self.p['cutoff']
      self.atol      = self.p['acut']
      self.hbtol     = self.p['hbtol']
      self.hbshort   = hbshort
      self.hblong    = hblong
      self.set_rcut(rcut,rcuta)
      self.vdwcut    = vdwcut
      self.d1  = np.triu(np.ones([self.natom,self.natom],dtype=np.float32),k=0)
      self.d2  = np.triu(np.ones([self.natom,self.natom],dtype=np.float32),k=1)
      self.eye = 1.0 - np.eye(self.natom,dtype=np.float32)
      self.get_rcbo()
      self.set_p(m,self.bo_layer)
      self.Qe= qeq(p=self.p,atoms=self.atoms)

    
  def get_charge(self,cell,positions):
      self.Qe.calc(cell,positions)
      self.q   = self.Qe.q[:-1]
      qij      = np.expand_dims(self.q,axis=0)*np.expand_dims(self.q,axis=1)
      self.qij = qij*14.39975840 


  def get_neighbor(self,cell,rcell,positions):
      xi    = np.expand_dims(positions,axis=0)
      xj    = np.expand_dims(positions,axis=1)
      vr    = xj-xi
      
      vrf   = np.dot(vr,rcell)
      vrf   = np.where(vrf-0.5>0,vrf-1.0,vrf)
      vrf   = np.where(vrf+0.5<0,vrf+1.0,vrf)  
      vr    = np.dot(vrf,cell)
      r     = np.sqrt(np.sum(vr*vr,axis=2))

      angs,tors,hbs = get_neighbors(self.natom,self.atom_name,self.r_cuta,r)

      self.angs  = np.array(angs)
      self.tors  = np.array(tors)
      self.hbs   = np.array(hbs)

      self.nang  = len(self.angs)
      self.ntor  = len(self.tors)
      self.nhb   = len(self.hbs)
    
      if self.nang>0:
         self.angi  = np.expand_dims(self.angs[:,0],axis=1)
         self.angj  = np.expand_dims(self.angs[:,1],axis=1)
         self.angk  = np.expand_dims(self.angs[:,2],axis=1)

         # self.angij = np.transpose([self.angs[:,0],self.angs[:,1]])
         # self.angjk = np.transpose([self.angs[:,1],self.angs[:,2]])
         # self.angik = np.transpose([self.angs[:,0],self.angs[:,2]])

      if self.ntor>0:
         self.tori  = np.expand_dims(self.tors[:,0],axis=1)
         self.torj  = np.expand_dims(self.tors[:,1],axis=1)
         self.tork  = np.expand_dims(self.tors[:,2],axis=1)
         self.torl  = np.expand_dims(self.tors[:,3],axis=1)

         # self.torij = np.transpose([self.tors[:,0],self.tors[:,1]])
         # self.torjk = np.transpose([self.tors[:,1],self.tors[:,2]])
         # self.torkl = np.transpose([self.tors[:,2],self.tors[:,3]])

      if self.nhb>0:
         self.hbi     = np.expand_dims(self.hbs[:,0],axis=1)
         self.hbj     = np.expand_dims(self.hbs[:,1],axis=1)
         self.hbk     = np.expand_dims(self.hbs[:,2],axis=1)
         # self.hbij  = np.transpose([self.hbs[:,0],self.hbs[:,1]])
         # self.hbjk  = np.transpose([self.hbs[:,1],self.hbs[:,2]])

      P_ = get_pangle(self.p,self.atom_name,len(self.p_ang),self.p_ang,self.nang,angs)
      self.P.update(P_)

      P_ = get_ptorsion(self.p,self.atom_name,len(self.p_tor),self.p_tor,self.ntor,tors)
      self.P.update(P_)

      P_ = get_phb(self.p,self.atom_name,len(self.p_hb),self.p_hb,self.nhb,hbs)
      self.P.update(P_)


  def set_rcut(self,rcut,rcuta): 
      rcut_,rcuta_,re_ = setRcut(self.bonds)
      if rcut is None:  ## bond order compute cutoff
         self.rcut = rcut_
      if rcuta is None: ## angle term cutoff
         self.rcuta = rcuta_

      self.r_cut = np.zeros([self.natom,self.natom],dtype=np.float32)
      self.r_cuta = np.zeros([self.natom,self.natom],dtype=np.float32)
      self.re = np.zeros([self.natom,self.natom],dtype=np.float32)
      for i in range(self.natom):
          for j in range(self.natom):
              bd = self.atom_name[i] + '-' + self.atom_name[j]
              if i!=j:
                 self.r_cut[i][j]  = self.rcut[bd]  
                 self.r_cuta[i][j] = self.rcuta[bd] 
                 self.re[i][j]     = re_[bd] 
                 # print(i,j,bd,re_[bd])


  def get_rcbo(self):
      ''' get cut-offs for individual bond '''
      self.rc_bo = {}
      for bd in self.bonds:
          b= bd.split('-')
          ofd=bd if b[0]!=b[1] else b[0]

          log_ = np.log((self.botol/(1.0+self.botol)))
          rr = log_/self.p['bo1_'+bd] 
          self.rc_bo[bd]=self.p['rosi_'+ofd]*np.power(log_/self.p['bo1_'+bd],1.0/self.p['bo2_'+bd])
  

  def get_bondorder_uc(self):
      self.frc = np.where(np.logical_or(self.r>self.rcbo,self.r<=0.001), 0.0,1.0)

      self.bodiv1 = self.r/self.P['rosi']
      self.bopow1 = np.power(self.bodiv1,self.P['bo2'])
      self.eterm1 = (1.0+self.botol)*np.exp(self.P['bo1']*self.bopow1)*self.frc # consist with GULP

      self.bodiv2 = self.r/self.P['ropi']
      self.bopow2 = np.power(self.bodiv2,self.P['bo4'])
      self.eterm2 = np.exp(self.P['bo3']*self.bopow2)*self.frc

      self.bodiv3 = self.r/self.P['ropp']
      self.bopow3 = np.power(self.bodiv3,self.P['bo6'])
      self.eterm3 = np.exp(self.P['bo5']*self.bopow3)*self.frc

      self.bop_si = taper(self.eterm1,rmin=self.botol,rmax=2.0*self.botol)*(self.eterm1-self.botol) # consist with GULP
      self.bop_pi = taper(self.eterm2,rmin=self.botol,rmax=2.0*self.botol)*self.eterm2
      self.bop_pp = taper(self.eterm3,rmin=self.botol,rmax=2.0*self.botol)*self.eterm3
      self.bop    = self.bop_si+self.bop_pi+self.bop_pp


  def f1(self):
      Dv  = np.expand_dims(self.Deltap - self.P['val'],axis=0)
      self.f2(Dv)
      self.f3(Dv)
      VAL      = np.expand_dims(self.P['val'],axis=1)
      VALt     = np.expand_dims(self.P['val'],axis=0)
      self.f_1 = 0.5*(np.divide(VAL+self.f_2,  VAL+self.f_2+self.f_3)  + 
                      np.divide(VALt+self.f_2, VALt+self.f_2+self.f_3))


  def f2(self,Dv):
      self.dexpf2  = np.exp(-self.P['boc1']*Dv)
      self.f_2     = self.dexpf2 + np.transpose(self.dexpf2,[1,0])


  def f3(self,Dv):
      self.dexpf3 = np.exp(-self.P['boc2']*Dv)
      delta_exp   = self.dexpf3 + np.transpose(self.dexpf3,[1,0])

      self.f3log  = np.log(0.5*delta_exp )
      self.f_3    = -1.0/(self.P['boc2']*self.f3log)


  def f45(self):
      self.D_boc = self.Deltap - self.P['valboc'] # + self.p['val_'+atomi]
      
      self.DELTA  = np.expand_dims(self.D_boc,axis=1)
      self.DELTAt = np.transpose(self.DELTA,[1,0])
      
      self.df4 = self.P['boc4']*np.square(self.bop)-self.DELTA
      self.f4r = np.exp(-self.P['boc3']*(self.df4)+self.P['boc5'])

      self.df5 = self.P['boc4']*np.square(self.bop)-self.DELTAt
      self.f5r = np.exp(-self.P['boc3']*(self.df5)+self.P['boc5'])

      self.f_4 = 1.0/(1.0+self.f4r)
      self.f_5 = 1.0/(1.0+self.f5r)


  def get_bondorder(self):
      self.f1()
      self.f45()

      self.F        = self.f_1*self.f_1*self.f_4*self.f_5 
      self.bo0      = self.bop*self.f_1*self.f_4*self.f_5   #-0.001        # consistent with GULP
     
      bo_           = self.bo0 - self.atol
      self.bo       = np.where(bo_>0.0,bo_,0.0)      #bond-order cut-off 0.001 reaxffatol
      self.bopi     = self.bop_pi*self.F
      self.bopp     = self.bop_pp*self.F
      bosi_         = self.bo0 - self.bopi - self.bopp
      self.bosi     = np.where(bosi_>0.0,bosi_,0.0)  
      self.bso      = self.P['ovun1']*self.P['Desi']*self.bo0 
      self.Delta    = np.sum(self.bo0,axis=1)   


  def f_nn(self,pre,x,layer=5):
      X   = np.expand_dims(np.stack(x,axis=2),2)

      o   = []
      o.append(sigmoid(np.matmul(X,self.m[pre+'wi'])+self.m[pre+'bi']))  
                                                                    # input layer
      for l in range(layer):                                        # hidden layer      
          o.append(sigmoid(np.matmul(o[-1],self.m[pre+'w'][l])+self.m[pre+'b'][l]))
      
      o_  = sigmoid(np.matmul(o[-1],self.m[pre+'wo']) + self.m[pre+'bo']) 
      out = np.squeeze(o_)                                          # output layer
      return out


  def massage_passing(self):
      self.H         = []    # hiden states (or embeding states)
      self.D         = []    # degree matrix
      self.Hsi       = []
      self.Hpi       = []
      self.Hpp       = []
      self.F         = []
      self.H.append(self.bop)                   # 
      self.Hsi.append(self.bop_si)              #
      self.Hpi.append(self.bop_pi)              #
      self.Hpp.append(self.bop_pp)              # 
      self.D.append(self.Deltap)                # get the initial hidden state H[0]

      for t in range(1,self.massages):
          Di_        = np.expand_dims(self.D[t-1],axis=0)*self.eye
          Dj_        = np.expand_dims(self.D[t-1],axis=1)*self.eye
        
          Dbi        = Di_ - self.H[t-1]
          Dbj        = Dj_ - self.H[t-1]

          # self.Fi  = self.f_nn('f',[Di,Dj,Dbi,Dbj],layer=self.bo_layer[1])
          Fi    = self.f_nn('f'+str(t),[Dbj,Dbi,self.H[t-1]],layer=self.bo_layer[1])
          Fj    = np.transpose(Fi,[1,0])
          F     = 2.0*Fi*Fj
          self.F.append(F)

          self.Hsi.append(self.Hsi[t-1]*F)
          self.Hpi.append(self.Hpi[t-1]*F)
          self.Hpp.append(self.Hpp[t-1]*F)
          self.H.append(self.Hsi[t]+self.Hpi[t]+self.Hpp[t])
          self.D.append(np.sum(self.H[t],axis=1))  


  def get_bondorder_nn(self):
      self.massage_passing()
      self.bosi  = self.Hsi[-1]       # getting the final state
      self.bopi  = self.Hpi[-1]
      self.bopp  = self.Hpp[-1]

      self.bo0   = self.bosi + self.bopi + self.bopp
      self.bo    = relu(self.bo0 - self.atol*self.eye)     # bond-order cut-off 0.001 reaxffatol
      self.bso   = self.P['ovun1']*self.P['Desi']*self.bo0  
      self.Delta = np.sum(self.bo0,axis=1)   

      Di_        = np.expand_dims(self.Delta,axis=0)*self.eye          # get energy layer
      Dj_        = np.expand_dims(self.Delta,axis=1)*self.eye
      Dbi        = Di_ - self.bo0
      Dbj        = Dj_ - self.bo0

      Fi         = self.f_nn('fe',[Dbj,Dbi,self.bosi],layer=self.bo_layer[1])
      Fj         = np.transpose(Fi,[1,0])
      F          = 2.0*Fi*Fj
      self.esi   = self.bosi*F


  def get_ebond(self,cell,rcell,positions):
      self.vr    = fvr(positions)
      vrf        = np.dot(self.vr,rcell)

      vrf        = np.where(vrf-0.5>0,vrf-1.0,vrf)
      vrf        = np.where(vrf+0.5<0,vrf+1.0,vrf) 

      self.vr    = np.dot(vrf,cell)
      self.r     = np.sqrt(np.sum(self.vr*self.vr,axis=2)+self.safety_value)

      self.get_bondorder_uc()
      self.Deltap= np.sum(self.bop,axis=1)  

      if self.nn:
         self.get_bondorder_nn()
      else:
         self.get_bondorder()

      self.Dv    = self.Delta - self.P['val']
      self.Dpi   = np.sum(self.bopi+self.bopp,axis=1) 

      self.so    = np.sum(self.P['ovun1']*self.P['Desi']*self.bo0,axis=1)  
      self.fbo   = taper(self.bo0,rmin=self.atol,rmax=2.0*self.atol) 
      self.fhb   = taper(self.bo0,rmin=self.hbtol,rmax=2.0*self.hbtol) 

      if self.nn:
         self.sieng = np.multiply(self.P['Desi'],self.esi)
      else:
         self.powb  = np.power(self.bosi+self.safety_value,self.P['be2'])
         self.expb  = np.exp(np.multiply(self.P['be1'],1.0-self.powb))
         self.sieng = self.P['Desi']*self.bosi*self.expb 

      self.pieng = np.multiply(self.P['Depi'],self.bopi)
      self.ppeng = np.multiply(self.P['Depp'],self.bopp)
      self.ebond = - self.sieng - self.pieng - self.ppeng
      self.Ebond = 0.5*np.sum(self.ebond)
      return self.Ebond


  def get_elone(self):
      self.NLPOPT  = 0.5*(self.P['vale'] - self.P['val'])
      self.Delta_e = 0.5*(self.Delta - self.P['vale'])
      self.DE      = relu(-np.ceil(self.Delta_e))  # number of lone pair electron
      self.nlp     = self.DE + np.exp(-self.P['lp1']*4.0*np.square(1.0+self.Delta_e+self.DE))
      
      self.Delta_lp= self.NLPOPT-self.nlp   
      self.Dlp     = self.Delta - self.P['val'] - self.Delta_lp   
      self.Dpil    = np.sum(np.expand_dims(self.Dlp,axis=0)*(self.bopi+self.bopp),1)

      self.explp   = 1.0+np.exp(-75.0*self.Delta_lp)
      self.elone   = self.P['lp2']*self.Delta_lp/self.explp
      self.Elone   = np.sum(self.elone)


  def get_eover(self):
      self.lpcorr= self.Delta_lp/(1.0+self.P['ovun3']*np.exp(self.P['ovun4']*self.Dpil))
      self.Delta_lpcorr = self.Dv - self.lpcorr

      self.otrm1 = 1.0/(self.Delta_lpcorr+self.P['val'])
      self.otrm2 = 1.0/(1.0+np.exp(self.P['ovun2']*self.Delta_lpcorr))
      self.eover = self.so*self.otrm1*self.Delta_lpcorr*self.otrm2
      self.Eover = np.sum(self.eover)


  def get_eunder(self):
      self.expeu1 = np.exp(self.P['ovun6']*self.Delta_lpcorr)
      self.eu1    = sigmoid(self.P['ovun2']*self.Delta_lpcorr)

      self.expeu3 = np.exp(self.P['ovun8']*self.Dpil)
      self.eu2    = 1.0/(1.0+self.P['ovun7']*self.expeu3)
      self.eunder = -self.P['ovun5']*(1.0-self.expeu1)*self.eu1*self.eu2   
      self.Eunder = np.sum(self.eunder)


  def get_theta(self):
      Rij = self.r[self.angi,self.angj]  
      Rjk = self.r[self.angj,self.angk]  
      Rik = self.r[self.angi,self.angk]  

      Rij2= Rij*Rij
      Rjk2= Rjk*Rjk
      Rik2= Rik*Rik

      self.cos_theta = np.squeeze((Rij2+Rjk2-Rik2)/(2.0*Rij*Rjk))
      self.theta     = np.arccos(self.cos_theta)


  def get_theta0(self,dang):
      sbo   = np.squeeze(self.Dpi[self.angj])
      pbo   = np.squeeze(self.PBO[self.angj])
      rnlp  = np.squeeze(self.nlp[self.angj])

      SBO   = sbo - (1.0-pbo)*(dang+self.P['val8']*rnlp)    
      
      ok    = np.logical_and(SBO<=1.0,SBO>0.0)
      S1    = np.where(ok,SBO,0.0)                             #  0< sbo < 1                  
      SBO01 = np.where(ok,np.power(S1,self.P['val9']),0.0)

      ok    = np.logical_and(SBO<2.0,SBO>1.0)
      S2    = np.where(ok,SBO,0.0)                 
      F2    = np.where(ok,1.0,0.0)                              #  1< sbo <2
      # print('F2',F2.shape)
      S2    = 2.0*F2-S2  
      # print('S2',S2.shape)
      SBO12 = np.where(ok,2.0-np.power(S2,self.P['val9']),0.0)  #  1< sbo <2
      SBO2  = np.where(SBO>2.0,1.0,0.0)                         #     sbo >2

      self.SBO3   = SBO01+SBO12+2.0*SBO2
      thet_ = 180.0 - self.P['theta0']*(1.0-np.exp(-self.P['val10']*(2.0-self.SBO3)))
      self.thet0 = thet_/57.29577951


  def get_eangle(self):
      self.Dang  = self.Delta - self.P['valang']
      self.boaij = np.squeeze(self.bo[self.angi,self.angj])
      self.boajk = np.squeeze(self.bo[self.angj,self.angk])
      fij        = self.fbo[self.angi,self.angj]   
      fjk        = self.fbo[self.angj,self.angk]   
      self.fijk  = np.squeeze(fij*fjk)

      dang       = np.squeeze(self.Dang[self.angj])
      PBOpow     = -np.power(self.bo+self.safety_value,8)  # bo0
      PBOexp     = np.exp(PBOpow)
      self.PBO   = np.prod(PBOexp,axis=1)

      self.get_theta()
      self.get_theta0(dang)

      self.thet  = self.thet0-self.theta
      self.expang= np.exp(-self.P['val2']*np.square(self.thet))
      self.f7(self.boaij,self.boajk)
      self.f8(dang)

      self.eang  = self.fijk*self.f_7*self.f_8*(self.P['val1']-self.P['val1']*self.expang) 
      self.Eang  = np.sum(self.eang)

      self.get_epenalty(self.boaij,self.boajk)
      self.get_three_conj(self.boaij,self.boajk)


  def f7(self,boij,bojk):
      self.expaij = np.exp(-self.P['val3']*np.power(boij+self.safety_value,self.P['val4']))
      self.expajk = np.exp(-self.P['val3']*np.power(bojk+self.safety_value,self.P['val4']))
      fi          = 1.0 - self.expaij
      fk          = 1.0 - self.expajk
      self.f_7    = fi*fk


  def f8(self,dang):
      exp6      = np.exp( self.P['val6']*dang)
      exp7      = np.exp(-self.P['val7']*dang)
      self.f_8  = self.P['val5'] - (self.P['val5'] - 1.0)*(2.0+exp6)/(1.0+exp6+exp7)


  def get_epenalty(self,boij,bojk):
      self.f9()
      expi      = np.exp(-self.P['pen2']*np.square(boij-2.0))
      expk      = np.exp(-self.P['pen2']*np.square(bojk-2.0))
      self.epen = self.P['pen1']*self.f_9*expi*expk*self.fijk
      self.Epen = np.sum(self.epen)


  def f9(self):
      D         = np.squeeze(self.Dv[self.angj])
      exp3      = np.exp(-self.P['pen3']*D)
      exp4      = np.exp( self.P['pen4']*D)
      self.f_9  = np.divide(2.0+exp3,1.0+exp3+exp4)


  def get_three_conj(self,boij,bojk):
      Dcoa_ = self.Delta-self.P['valboc']
      Dcoa  = np.squeeze(Dcoa_[self.angj])
      Di    = np.squeeze(self.Delta[self.angi])
      Dk    = np.squeeze(self.Delta[self.angk])
      self.expcoa1 = np.exp(self.P['coa2']*Dcoa)

      texp0 = np.divide(self.P['coa1'],1.0+self.expcoa1)  
      texp1 = np.exp(-self.P['coa3']*np.square(Di-boij))
      texp2 = np.exp(-self.P['coa3']*np.square(Dk-bojk))
      texp3 = np.exp(-self.P['coa4']*np.square(boij-1.5))
      texp4 = np.exp(-self.P['coa4']*np.square(bojk-1.5))
      self.etcon = texp0*texp1*texp2*texp3*texp4*self.fijk
      # print(texp0.shape)
      # print(self.etcon.shape)
      self.Etcon = np.sum(self.etcon)
  

  def get_torsion_angle(self):
      rij = np.squeeze(self.r[self.tori,self.torj])
      rjk = np.squeeze(self.r[self.torj,self.tork])
      rkl = np.squeeze(self.r[self.tork,self.torl])

      vrjk= np.squeeze(self.vr[self.torj,self.tork])
      vrkl= np.squeeze(self.vr[self.tork,self.torl])

      vrjl= vrjk + vrkl
      rjl = np.sqrt(np.sum(np.square(vrjl),axis=1))

      vrij= np.squeeze(self.vr[self.tori,self.torj])
      vril= vrij + vrjl
      ril = np.sqrt(np.sum(np.square(vril),axis=1))

      vrik= vrij + vrjk
      rik = np.sqrt(np.sum(np.square(vrik),axis=1))

      rij2= np.square(rij)
      rjk2= np.square(rjk)
      rkl2= np.square(rkl)
      rjl2= np.square(rjl)
      ril2= np.square(ril)
      rik2= np.square(rik)

      c_ijk = (rij2+rjk2-rik2)/(2.0*rij*rjk)
      c2ijk = np.square(c_ijk)
      # tijk  = np.arccos(c_ijk)
      cijk  = 1.0 - c2ijk
      self.s_ijk = np.sqrt(cijk)

      c_jkl = (rjk2+rkl2-rjl2)/(2.0*rjk*rkl)
      c2jkl = np.square(c_jkl)
      cjkl  = 1.0 - c2jkl
      self.s_jkl = np.sqrt(cjkl)

      c_ijl = (rij2+rjl2-ril2)/(2.0*rij*rjl)
      c_kjl = (rjk2+rjl2-rkl2)/(2.0*rjk*rjl)

      c2kjl = np.square(c_kjl)
      ckjl  = 1.0-c2kjl
      s_kjl = np.sqrt(ckjl)

      fz    = rij2+rjl2-ril2-2.0*rij*rjl*c_ijk*c_kjl
      fm    = rij*rjl*self.s_ijk*s_kjl

      fm    = np.where(np.logical_and(fm<=0.000001,fm>=-0.000001),1.0,fm)
      fac   = np.where(np.logical_and(fm<=0.000001,fm>=-0.000001),0.0,1.0)
      cos_w = 0.5*fz*fac/fm
      #cos_w= cos_w*ccijk*ccjkl
      cos_w = np.where(cos_w>0.9999999,1.0,cos_w)   
      self.cos_w = np.where(cos_w<-0.999999,-1.0,cos_w)
      self.w= np.arccos(self.cos_w)
      self.cos2w = np.cos(2.0*self.w)


  def get_etorsion(self):
      self.get_torsion_angle()

      self.botij = np.squeeze(self.bo[self.tori,self.torj])
      self.botjk = np.squeeze(self.bo[self.torj,self.tork])
      self.botkl = np.squeeze(self.bo[self.tork,self.torl])
      fij        = np.squeeze(self.fbo[self.tori,self.torj])
      fjk        = np.squeeze(self.fbo[self.torj,self.tork])
      fkl        = np.squeeze(self.fbo[self.tork,self.torl])
      self.fijkl = fij*fjk*fkl

      Dj         = np.squeeze(self.Dang[self.torj])
      Dk         = np.squeeze(self.Dang[self.tork])

      self.f10(self.botij,self.botjk,self.botkl)
      self.f11(Dj,Dk)

      self.bopjk = np.squeeze(self.bopi[self.torj,self.tork])  #   different from reaxff manual
      self.expv2 = np.exp(self.P['tor1']*np.square(2.0-self.bopjk-self.f_11)) 
      self.cos3w = np.cos(3.0*self.w)

      self.v1    = 0.5*self.P['V1']*(1.0+self.cos_w)  
      self.v2    = 0.5*self.P['V2']*self.expv2*(1.0-self.cos2w)
      self.v3    = 0.5*self.P['V3']*(1.0+self.cos3w)

      self.etor  = self.fijkl*self.f_10*self.s_ijk*self.s_jkl*(self.v1+self.v2+self.v3)
      self.Etor  = np.sum(self.etor)
      # print('Etor',self.etor.shape)
      self.get_four_conj(self.botij,self.botjk,self.botkl)


  def f10(self,boij,bojk,bokl):
      exp1 = 1.0 - np.exp(-self.P['tor2']*boij)
      exp2 = 1.0 - np.exp(-self.P['tor2']*bojk)
      exp3 = 1.0 - np.exp(-self.P['tor2']*bokl)
      self.f_10 = exp1*exp2*exp3


  def f11(self,Dj,Dk):
      delt      = Dj+Dk
      f11exp3   = np.exp(-self.P['tor3']*delt)
      f11exp4   = np.exp( self.P['tor4']*delt)
      self.f_11 = np.divide(2.0+f11exp3,1.0+f11exp3+f11exp4)


  def get_four_conj(self,boij,bojk,bokl):
      exptol= np.exp(-self.P['cot2']*np.square(self.atol - 1.5))
      expij = np.exp(-self.P['cot2']*np.square(boij-1.5))-exptol
      expjk = np.exp(-self.P['cot2']*np.square(bojk-1.5))-exptol 
      expkl = np.exp(-self.P['cot2']*np.square(bokl-1.5))-exptol

      self.f_12  = expij*expjk*expkl
      self.prod  = 1.0+(np.square(np.cos(self.w))-1.0)*self.s_ijk*self.s_jkl
      self.efcon = self.fijkl*self.f_12*self.P['cot1']*self.prod  
      # print('efcon',self.efcon.shape)
      self.Efcon = np.sum(self.efcon)


  def f13(self,r):
      rr = np.power(r,self.P['vdw1'])+np.power(np.divide(1.0,self.P['gammaw']),self.P['vdw1'])
      f_13 = np.power(rr,np.divide(1.0,self.P['vdw1']))  
      return f_13


  def get_tap(self,r):
      tp = 1.0+np.divide(-35.0,np.power(self.vdwcut,4.0))*np.power(r,4.0)+ \
           np.divide(84.0,np.power(self.vdwcut,5.0))*np.power(r,5.0)+ \
           np.divide(-70.0,np.power(self.vdwcut,6.0))*np.power(r,6.0)+ \
           np.divide(20.0,np.power(self.vdwcut,7.0))*np.power(r,7.0)
      return tp


  def get_evdw(self):
      self.evdw = 0.0
      self.ecoul= 0.0
      nc = 0
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell = self.cell[0]*i + self.cell[1]*j+self.cell[2]*k
                  vr_  = self.vr + cell
                  r    = np.sqrt(np.sum(np.square(vr_),axis=2)+self.safety_value)

                  gm3  = np.power(np.divide(1.0,self.P['gamma']),3.0)
                  r3   = np.power(r,3.0)

                  fv_   = np.where(np.logical_or(r<=0.0000001,r>self.vdwcut),0.0,1.0)
                  if nc<13:
                     fv = fv_*self.d1
                  else:
                     fv = fv_*self.d2

                  f_13 = self.f13(r)
                  tpv  = self.get_tap(r)

                  expvdw1 = np.exp(0.5*self.P['alfa']*(1.0-np.divide(f_13,2.0*self.P['rvdw'])))
                  expvdw2 = np.square(expvdw1) 
                  self.evdw  += fv*tpv*self.P['Devdw']*(expvdw2-2.0*expvdw1)

                  rth         = np.power(r3+gm3,1.0/3.0)                                      # ecoul
                  self.ecoul += np.divide(fv*tpv*self.qij,rth)
                  nc += 1

      self.Evdw  = np.sum(self.evdw)
      self.Ecoul = np.sum(self.ecoul)

  
  def get_ehb(self):
      self.BOhb   = np.squeeze(self.bo0[self.hbi,self.hbj])
      fhb         = np.squeeze(self.fhb[self.hbi,self.hbj])

      rij         = np.squeeze(self.r[self.hbi,self.hbj])
      rij2        = np.square(rij)
      vrij        = np.squeeze(self.vr[self.hbi,self.hbj]) 
      vrjk_       = np.squeeze(self.vr[self.hbj,self.hbk]) 
      self.Ehb    = 0.0
      
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell   = self.cell[0]*i + self.cell[1]*j+self.cell[2]*k
                  vrjk   = vrjk_ + cell
                  rjk2   = np.sum(np.square(vrjk),axis=1)
                  rjk    = np.sqrt(rjk2)
                  
                  vrik   = vrij + vrjk
                  rik2   = np.sum(np.square(vrik),axis=1)
                  rik    = np.sqrt(rik2)

                  cos_th = (rij2+rjk2-rik2)/(2.0*rij*rjk)
                  hbthe  = 0.5-0.5*cos_th
                  frhb   = rtaper(rik,rmin=self.hbshort,rmax=self.hblong)

                  exphb1 = 1.0-np.exp(-self.P['hb1']*self.BOhb)
                  hbsum  = np.divide(self.P['rohb'],rjk)+np.divide(rjk,self.P['rohb'])-2.0
                  exphb2 = np.exp(-self.P['hb2']*hbsum)
               
                  sin4   = np.square(hbthe)
                  ehb    = fhb*frhb*self.P['Dehb']*exphb1*exphb2*sin4 
                  self.Ehb += np.sum(ehb)


  def get_eself(self):
      chi    = np.expand_dims(self.P['chi'],axis=0)
      mu     = np.expand_dims(self.P['mu'],axis=0)
      self.eself = self.q*(chi+self.q*mu)
      # print(self.eself.shape)
      self.Eself = np.sum(self.eself)


  def get_total_energy(self,cell,rcell,positions):
      self.get_ebond(cell,rcell,positions)
      self.get_elone()
      self.get_eover()
      self.get_eunder()

      if self.nang>0:
         self.get_eangle()
      else:
         self.Eang  = 0.0
         self.Epen  = 0.0
         self.Etcon = 0.0

      if self.ntor>0:
         self.get_etorsion()
      else:
         self.Etor  = 0.0
         self.Efcon = 0.0

      self.get_evdw()

      if self.nhb>0:
         self.get_ehb()
      else:
         self.Ehb   = 0.0
         
      self.get_eself()

      E = self.Ebond + self.Elone + self.Eover + self.Eunder + \
               self.Eang + self.Epen + self.Etcon + \
               self.Etor + self.Efcon + self.Evdw + self.Ecoul + \
               self.Ehb + self.Eself + self.zpe
      return E


  def update_p(self,libfile='ffield.json'):
      if libfile.endswith('.json'):
         lf = open(libfile,'r')
         j = js.load(lf)
         self.p  = j['p']
         m       = j['m']
         self.zpe_= j['zpe']
         if 'bo_layer' in j:
            self.bo_layer = j['bo_layer']
         else:
            self.bo_layer = None
         lf.close()
      else:
         self.p,zpe_,self.spec,self.bonds,self.offd,self.Angs,self.torp,self.Hbs= \
                       read_lib(libfile=libfile,zpe=True)
         m           = None
         bo_layer    = None
      self.check_tors(self.p_tor)
      self.botol     = 0.01*self.p['cutoff']
      self.atol      = self.p['acut']
      self.hbtol     = self.p['hbtol']

      self.get_rcbo()
      self.set_p(m,self.bo_layer)


  def calculate_Delta(self,atoms=None,updateP=False):
      if updateP:
         self.update_p()

      cell      = atoms.get_cell()                    # cell is object now
      cell      = cell[:].astype(dtype=np.float32)
      rcell     = np.linalg.inv(cell).astype(dtype=np.float32)

      positions = atoms.get_positions()
      xf        = np.dot(positions,rcell)
      xf        = np.mod(xf,1.0)
      positions = np.dot(xf,cell).astype(dtype=np.float32)

      # self.get_charge(cell,positions)
      self.get_neighbor(cell,rcell,positions)
      self.get_ebond(cell,rcell,positions)

      n       = np.where(np.logical_and(self.r>0.0001,self.r<self.r_cuta),1.0,0.0)
      N       = np.sum(n,axis=1)                      # nearest neighbor matrix
      self.ND = N-self.P['val']                       # 

      
  def calculate(self,atoms=None):
      cell      = atoms.get_cell()                    # cell is object now
      cell      = cell[:].astype(dtype=np.float32)
      rcell     = np.linalg.inv(cell).astype(dtype=np.float32)

      positions = atoms.get_positions()
      xf        = np.dot(positions,rcell)
      xf        = np.mod(xf,1.0)
      positions = np.dot(xf,cell).astype(dtype=np.float32)

      self.get_charge(cell,positions)
      self.get_neighbor(cell,rcell,positions)

      self.E    = self.get_total_energy(cell,rcell,positions)
      
      # self.results['energy'] = self.E.numpy()[0]
      # self.results['forces'] = -self.grad.numpy()
      # self.results['stress'] = v


  def get_pot_energy(self,atoms):
      cell      = atoms.get_cell()                    # cell is object now
      cell      = cell[:].astype(dtype=np.float32)
      rcell     = np.linalg.inv(cell).astype(dtype=np.float32)

      positions = atoms.get_positions()
      xf        = np.dot(positions,rcell)
      xf        = np.mod(xf,1.0)
      positions = np.dot(xf,cell).astype(dtype=np.float32)

      self.get_charge(cell,positions)
      self.get_neighbor(cell,rcell,positions)

      self.E = self.get_total_energy(cell,rcell,positions)
      return self.E


  def set_p(self,m,bo_layer):
      ''' setting up parameters '''
      self.unit   = 4.3364432032e-2
      self.punit  = ['Desi','Depi','Depp','lp2','ovun5','val1',
                     'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb']
      p_bond = ['Desi','Depi','Depp','be1','bo5','bo6','ovun1',
                'be2','bo3','bo4','bo1','bo2',
                'Devdw','rvdw','alfa','rosi','ropi','ropp']
      p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
      self.P = {}

      self.rcbo = np.zeros([self.natom,self.natom],dtype=np.float32)

      for key in p_offd:
          for sp in self.spec:
              try:
                 self.p[key+'_'+sp+'-'+sp]  = self.p[key+'_'+sp]  
              except KeyError:
                 print('-  warning: key not in dict') 

      for i in range(self.natom):
          for j in range(self.natom):
              bd = self.atom_name[i] + '-' + self.atom_name[j]
              if not bd in self.bonds:
                 bd = self.atom_name[j] + '-' + self.atom_name[i]
              self.rcbo[i][j] = min(self.rcut[bd],self.rc_bo[bd])   #  ###### TODO #####

      p_spec = ['valang','valboc','val','vale',
                'lp2','ovun5',                 # 'val3','val5','boc3','boc4','boc5'
                'ovun2','atomic',
                'mass','chi','mu']             # 'gamma','gammaw','Devdw','rvdw','alfa'

      for key in p_spec:
          unit_ = self.unit if key in self.punit else 1.0
          self.P[key] = np.zeros([self.natom],dtype=np.float32)
          for i in range(self.natom):
                sp = self.atom_name[i]
                self.P[key][i] = self.p[key+'_'+sp]*unit_
      self.zpe = -np.sum(self.P['atomic'])

      for key in ['boc3','boc4','boc5','gamma','gammaw']:
          self.P[key] = np.zeros([self.natom,self.natom],dtype=np.float32)
          for i in range(self.natom):
              for j in range(self.natom):
                  self.P[key][i][j] = np.sqrt(self.p[key+'_'+self.atom_name[i]]*self.p[key+'_'+self.atom_name[j]],
                                              dtype=np.float32)
      
      for key in p_bond:
          unit_ = self.unit if key in self.punit else 1.0
          self.P[key] = np.zeros([self.natom,self.natom],dtype=np.float32)
          for i in range(self.natom):
              for j in range(self.natom):
                  bd = self.atom_name[i] + '-' + self.atom_name[j]
                  if bd not in self.bonds:
                     bd = self.atom_name[j] + '-' + self.atom_name[i]
                  self.P[key][i][j] = self.p[key+'_'+bd]*unit_
      
      p_g  = ['boc1','boc2','coa2','ovun6',
              'ovun7','ovun8','val6','lp1','val9','val10','tor2',
              'tor3','tor4','cot2','coa4','ovun4',               
              'ovun3','val8','coa3','pen2','pen3','pen4','vdw1'] 
      for key in p_g:
          self.P[key] = self.p[key]
 
      for key in self.p_ang:
          unit_ = self.unit if key in self.punit else 1.0
          for a in self.Angs:
              pn = key + '_' + a
              self.p[pn] = self.p[pn]*unit_

      for key in self.p_tor:
          unit_ = self.unit if key in self.punit else 1.0
          for t in self.Tors:
              pn = key + '_' + t
              self.p[pn] = self.p[pn]*unit_

      for h in self.Hbs:
          pn = 'Dehb_' + h
          self.p[pn] = self.p[pn]*self.unit

      if self.nn:
         self.set_m(m,bo_layer)


  def set_m(self,m,bo_layer):
      self.m = {}
      for t in range(1,self.massages+1):
          for k in ['wi','bi','wo','bo']:
              if t==self.massages:
                 key = 'fe'+k
              else:
                 key = 'f'+str(t)+k
              self.m[key] = []
              for i in range(self.natom):
                  mi_ = []
                  for j in range(self.natom):
                      bd = self.atom_name[i] + '-' + self.atom_name[j]
                      if k in ['bi','bo']:
                         mi_.append(np.expand_dims(m[key+'_'+bd],axis=0))
                      else:
                         mi_.append(m[key+'_'+bd])
                  self.m[key].append(mi_)
              self.m[key] = np.array(self.m[key],dtype=np.float32)

          for k in ['w','b']:
              if t==self.massages:
                 key = 'fe'+k
              else:
                 key = 'f'+str(t)+k
              self.m[key] = []
              for l in range(bo_layer[1]):
                  m_ = []
                  for i in range(self.natom):
                      mi_ = []
                      for j in range(self.natom):
                          bd = self.atom_name[i] + '-' + self.atom_name[j]
                          if k == 'b':
                             mi_.append(np.expand_dims(m[key+'_'+bd][l],axis=0))
                          else:
                             mi_.append(m[key+'_'+bd][l])
                      m_.append(mi_)
                  self.m[key].append(np.array(m_,dtype=np.float32))


  def init_bonds(self):
      self.bonds,self.offd,self.Angs,self.torp,self.Hbs = [],[],[],[],[]
      for key in self.p:
          k = key.split('_')
          if k[0]=='bo1':
             self.bonds.append(k[1])
          elif k[0]=='rosi':
             kk = k[1].split('-')
             if len(kk)==2:
                self.offd.append(k[1])
          elif k[0]=='theta0':
             self.Angs.append(k[1])
          elif k[0]=='tor1':
             self.torp.append(k[1])
          elif k[0]=='rohb':
             self.Hbs.append(k[1])


  def checkTors(self,torp):
      tors_ = torp
      for tor in tors_:
          [t1,t2,t3,t4] = tor.split('-')
          tor1 = t1+'-'+t3+'-'+t2+'-'+t4
          tor2 = t4+'-'+t3+'-'+t2+'-'+t1
          tor3 = t4+'-'+t2+'-'+t3+'-'+t1

          if tor1 in torp and tor1!=tor:
             # print('-  dict %s is repeated, delteting ...' %tor1)
             torp.remove(tor1)
          elif tor2 in self.torp and tor2!=tor:
             # print('-  dict %s is repeated, delteting ...' %tor2)
             torp.remove(tor2)
          elif tor3 in self.torp and tor3!=tor:
             # print('-  dict %s is repeated, delteting ...' %tor3)
             torp.remove(tor3)  

      self.Tors = []          ### check torsion parameter
      for spi in self.spec:
          for spj in self.spec:
              for spk in self.spec:
                  for spl in self.spec:
                      tor = spi+'-'+spj+'-'+spk+'-'+spl
                      if tor not in self.Tors:
                         self.Tors.append(tor)
      return torp 


  def check_tors(self,p_tor):
      for key in p_tor:
          for tor in self.Tors:
              if tor not in self.torp:
                 [t1,t2,t3,t4] = tor.split('-')
                 tor1 = t1+'-'+t3+'-'+t2+'-'+t4
                 tor2 = t4+'-'+t3+'-'+t2+'-'+t1
                 tor3 = t4+'-'+t2+'-'+t3+'-'+t1
                 if tor1 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor1]
                 elif tor2 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor2]
                 elif tor3 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor3]    
                 else:
                    print('-  an error case for %s .........' %tor)
      

  def logout(self):
      with open('irff.log','w') as fmd:
         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n-                Energies From Machine Learning MD                     -\n')
         fmd.write('\n------------------------------------------------------------------------\n')

         fmd.write('-  Ebond =%f  ' %self.Ebond)
         fmd.write('-  Elone =%f  ' %self.Elone)
         fmd.write('-  Eover =%f  \n' %self.Eover)
         fmd.write('-  Eunder=%f  ' %self.Eunder)
         fmd.write('-  Eang  =%f  ' %self.Eang)
         fmd.write('-  Epen  =%f  \n' %self.Epen)
         fmd.write('-  Etcon =%f  ' %self.Etcon)
         fmd.write('-  Etor  =%f  ' %self.Etor)
         fmd.write('-  Efcon =%f  \n' %self.Efcon)
         fmd.write('-  Evdw  =%f  ' %self.Evdw)
         fmd.write('-  Ecoul =%f  ' %self.Ecoul)
         fmd.write('-  Ehb   =%f  \n' %self.Ehb)
         fmd.write('-  Eself =%f  ' %self.Eself)
         fmd.write('-  Ezpe  =%f  \n' %self.zpe)
         
         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n-              Atomic Information  (Delta and Bond order)              -\n')
         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n  AtomID Sym  Delta      NLP        DLPC          Bond-Order \n')
         for i in range(self.natom):
             fmd.write('%6d  %2s %9.6f %9.6f %9.6f' %(i,
                                      self.atom_name[i],
                                      self.Delta[i],
                                      self.nlp[i],
                                      self.Delta_lpcorr[i]))
             for j in range(self.natom):
                   if self.bo0[i][j]>self.botol:
                      fmd.write(' %3d %2s %9.6f' %(j,self.atom_name[j],
                                                 self.bo0[i][j]))
             fmd.write(' \n')

         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n-                          Atomic Energies                             -\n')
         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n  AtomID Sym  Explp     Delta_lp     Elone     Eover      Eunder      Fx        Fy         Fz\n')
         for i in range(self.natom):
             fmd.write('%6d  %2s  %9.6f  %9.6f  %9.6f  %9.6f  %9.6f ' %(i,
                       self.atom_name[i],
                       self.explp[i],
                       self.Delta_lp[i],
                       self.elone[i],
                       self.eover[i],
                       self.eunder[i]))
             fmd.write(' \n')

         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n- Machine Learning MD Completed!\n')


  def close(self):
      self.P  = None
      self.m  = None
      self.Qe = None




