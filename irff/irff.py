import json as js
import numpy as np
from ase import Atoms
from ase.io import read,write
from ase.calculators.calculator import Calculator, all_changes
from .qeq import qeq
from .RadiusCutOff import setRcut
from .reaxfflib import read_ffield,write_lib
from .neighbors import get_neighbors,get_pangle,get_ptorsion,get_phb
from torch.autograd import Variable
import torch


try:
   from .neighbor import get_neighbors,get_pangle,get_ptorsion,get_phb
except ImportError:
   from .neighbors import get_neighbors,get_pangle,get_ptorsion,get_phb


def rtaper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = torch.where(r<rmin,torch.full_like(r,1.0),torch.full_like(r,0.0)) # r > rmax then 1 else 0

    ok    = torch.logical_and(r<=rmax,r>rmin)     # rmin < r < rmax  = r else 0
    r2    = torch.where(ok,r,torch.full_like(r,0.0))
    r20   = torch.where(ok,torch.full_like(r,1.0),torch.full_like(r,0.0))

    rterm = 1.0/(rmax-rmin)**3.0
    rm    = rmax*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmin*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3


def taper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = torch.where(r>rmax,torch.full_like(r,1.0),torch.full_like(r,0.0)) # r > rmax then 1 else 0

    ok    = torch.logical_and(r<=rmax,r>rmin)      # rmin < r < rmax  = r else 0
    r2    = torch.where(ok,r,torch.full_like(r,0.0))
    r20   = torch.where(ok,torch.full_like(r,1.0),torch.full_like(r,0.0))

    rterm = 1.0/(rmin-rmax)**3.0
    rm    = rmin*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmax*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3
    

def fvr(x):
    xi  = x.unsqueeze(0)
    xj  = x.unsqueeze(1) 
    vr  = xj - xi
    return vr


def fr(vr):
    R   = torch.sqrt(torch.sum(vr*vr,2))
    return R


def DIV(y,x):
    xok = (x!=0.0)
    f = lambda x: y/x
    safe_x = torch.where(xok,x,torch.full_like(x,1.0))
    return torch.where(xok, f(safe_x), torch.full_like(x,0.0))


def DIV_IF(y,x):
    xok = (x!=0.0)
    f = lambda x: y/x
    safe_x = torch.where(xok,x,torch.full_like(x,0.00000001))
    return torch.where(xok, f(safe_x), f(safe_x))


def relu(x):
    return torch.where(x>0.0,x,torch.full_like(x,0.0))  


class IRFF(Calculator):
  '''Intelligent Machine-Learning ASE calculator'''
  name = "IRFF"
  implemented_properties = ["energy", "forces", "stress","pressure"]
  def __init__(self,atoms=None,
               mol=None,
               libfile='ffield',
               vdwcut=10.0,
               atol=0.001,
               hbtol=0.001,
               nn=False,# vdwnn=False,
               messages=1,
               hbshort=6.75,hblong=7.5,
               nomb=False,  # this option is used when deal with metal system
               autograd=True,
               CalStress=False,
               label="IRFF", **kwargs):
      Calculator.__init__(self,label=label, **kwargs)
      self.atoms        = atoms
      self.cell         = atoms.get_cell()
      self.atom_name    = self.atoms.get_chemical_symbols()
      self.natom        = len(self.atom_name)
      self.spec         = []
      self.nn           = nn
      # self.vdwnn      = vdwnn
      self.EnergyFunction = 0
      self.autograd     = autograd
      self.nomb         = nomb # without angle, torsion and hbond manybody term
      self.messages     = messages 
      self.safety_value = 0.000000001
      self.GPa          = 1.60217662*1.0e2
      self.CalStress    = CalStress

      if libfile.endswith('.json'):
         lf                  = open(libfile,'r')
         j                   = js.load(lf)
         self.p              = j['p']
         m                   = j['m']
         self.MolEnergy_     = j['MolEnergy']
         self.messages       = j['messages']
         self.BOFunction     = j['BOFunction']
         self.EnergyFunction = j['EnergyFunction']
         self.MessageFunction= j['MessageFunction']
         self.VdwFunction    = j['VdwFunction']
         self.bo_layer       = j['bo_layer']
         self.mf_layer       = j['mf_layer']
         self.be_layer       = j['be_layer']
         self.vdw_layer      = j['vdw_layer']
         if not self.vdw_layer is None:
            self.vdwnn       = True
         else:
            self.vdwnn       = False
         rcut                = j['rcut']
         rcuta               = j['rcutBond']
         re                  = j['rEquilibrium']
         lf.close()
         self.init_bonds()
         if mol is None:
            self.emol = 0.0
         else:
            mol_ = mol.split('-')[0]
            if mol_ in self.MolEnergy_:
               self.emol = self.MolEnergy_[mol_]
            else:
               self.emol = 0.0
      else:
         self.p,zpe_,self.spec,self.bonds,self.offd,self.angs,self.torp,self.Hbs= \
                       read_ffield(libfile=libfile,zpe=False)
         m             = None
         self.bo_layer = None
         self.emol     = 0.0
         rcut          = None
         rcuta         = None
         self.vdwnn    = False
         self.EnergyFunction  = 0
         self.MessageFunction = 0
         self.VdwFunction     = 0
         self.p['acut']   = 0.0001
         self.p['hbtol']  = 0.0001
      if m is None:
         self.nn=False
         
      for sp in self.atom_name:
          if sp not in self.spec:
             self.spec.append(sp)

      self.hbshort   = hbshort
      self.hblong    = hblong
      self.set_rcut(rcut,rcuta,re)
      self.vdwcut    = vdwcut
      self.botol     = 0.01*self.p['cutoff']
      self.atol      = self.p['acut']   # atol
      self.hbtol     = self.p['hbtol']  # hbtol
      self.check_offd()
      self.check_hb()
      self.get_rcbo()
      self.set_p(m,self.bo_layer)
      self.Qe= qeq(p=self.p,atoms=self.atoms)

  def get_charge(self,cell,positions):
      self.Qe.calc(cell,positions)
      self.q   = self.Qe.q[:-1]
      qij      = np.expand_dims(self.q,axis=0)*np.expand_dims(self.q,axis=1)
      self.qij = torch.tensor(qij*14.39975840)

  def get_neighbor(self,cell,rcell,positions):
      xi    = np.expand_dims(positions,axis=0)
      xj    = np.expand_dims(positions,axis=1)
      vr    = xj-xi
      
      vrf   = np.dot(vr,rcell)
      vrf   = np.where(vrf-0.5>0,vrf-1.0,vrf)
      vrf   = np.where(vrf+0.5<0,vrf+1.0,vrf)  
      vr    = np.dot(vrf,cell)
      r     = np.sqrt(np.sum(vr*vr,axis=2))
      
      if self.nomb:
         angs,tors,hbs = [],[],[]
      else:
         angs,tors,hbs = get_neighbors(self.natom,self.atom_name,self.r_cuta,r)

      self.angs  = np.array(angs)
      self.tors  = np.array(tors)
      self.hbs   = np.array(hbs)

      self.nang  = len(self.angs)
      self.ntor  = len(self.tors)
      self.nhb   = len(self.hbs)
    
      if self.nang>0:
         self.angj  = self.angs[:,1]
         self.angi  = self.angs[:,0]
         self.angk  = self.angs[:,2]
         P_ = get_pangle(self.p,self.atom_name,len(self.p_ang),self.p_ang,self.nang,angs)
         for key in P_:
             self.P[key] = torch.from_numpy(P_[key])

      if self.ntor>0:
         self.tori  = self.tors[:,0]
         self.torj  = self.tors[:,1]
         self.tork  = self.tors[:,2]
         self.torl  = self.tors[:,3]
         P_ = get_ptorsion(self.p,self.atom_name,len(self.p_tor),self.p_tor,self.ntor,tors)
         for key in P_:
             self.P[key] = torch.from_numpy(P_[key])

      if self.nhb>0:
         self.hbi     = self.hbs[:,0]
         self.hbj     = self.hbs[:,1]
         self.hbk     = self.hbs[:,2]
         P_ = get_phb(self.p,self.atom_name,len(self.p_hb),self.p_hb,self.nhb,hbs)
         for key in P_:
             self.P[key] = torch.from_numpy(P_[key])

  def set_rcut(self,rcut,rcuta,re): 
      rcut_,rcuta_,re_ = setRcut(self.bonds,rcut,rcuta,re)
      self.rcut  = rcut_    ## bond order compute cutoff
      self.rcuta = rcuta_   ## angle term cutoff

      # self.r_cut = np.zeros([self.natom,self.natom],dtype=np.float32)
      # self.r_cuta = np.zeros([self.natom,self.natom],dtype=np.float32)
      self.re = np.zeros([self.natom,self.natom],dtype=np.float32)
      for i in range(self.natom):
          for j in range(self.natom):
              bd = self.atom_name[i] + '-' + self.atom_name[j]
              if i!=j:
                 # self.r_cut[i][j]  = self.rcut[bd]  
                 # self.r_cuta[i][j] = self.rcuta[bd] 
                 self.re[i][j]     = re_[bd] 

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
      if self.nn:
         self.frc = 1.0
      else:
         self.frc = torch.where(torch.logical_and(self.r<self.rcbo_tensor,self.r>0.0001),1.0,0.0)
      self.bodiv1 = torch.div(self.r,self.P['rosi'])
      self.bopow1 = torch.pow(self.bodiv1,self.P['bo2'])
      self.eterm1 = (1.0+self.botol)*torch.exp(torch.mul(self.P['bo1'],self.bopow1))*self.frc # consist with GULP

      self.bodiv2 = torch.div(self.r,self.P['ropi'])
      self.bopow2 = torch.pow(self.bodiv2,self.P['bo4'])
      self.eterm2 = torch.exp(torch.mul(self.P['bo3'],self.bopow2))*self.frc

      self.bodiv3 = torch.div(self.r,self.P['ropp'])
      self.bopow3 = torch.pow(self.bodiv3,self.P['bo6'])
      self.eterm3 = torch.exp(torch.mul(self.P['bo5'],self.bopow3))*self.frc

      if self.nn:
         if self.BOFunction==0:
            fsi_  = taper(self.eterm1,rmin=self.botol,rmax=2.0*self.botol)*(self.eterm1-self.botol)
            fpi_  = taper(self.eterm2,rmin=self.botol,rmax=2.0*self.botol)*self.eterm2
            fpp_  = taper(self.eterm3,rmin=self.botol,rmax=2.0*self.botol)*self.eterm3
         elif self.BOFunction==1:
            fsi_  = self.f_nn('fsi',[self.eterm1],layer=self.bo_layer[1])
            fpi_  = self.f_nn('fpi',[self.eterm2],layer=self.bo_layer[1])
            fpp_  = self.f_nn('fpp',[self.eterm3],layer=self.bo_layer[1])  
         elif self.BOFunction==2:
            fsi_  = self.f_nn('fsi',[-self.eterm1],layer=self.bo_layer[1])
            fpi_  = self.f_nn('fpi',[-self.eterm2],layer=self.bo_layer[1])
            fpp_  = self.f_nn('fpp',[-self.eterm3],layer=self.bo_layer[1])
         else:
            raise NotImplementedError('-  BO function not supported yet!')
         self.bop_si = fsi_*self.eye #*self.frc #*self.eterm1
         self.bop_pi = fpi_*self.eye #*self.frc #*self.eterm2
         self.bop_pp = fpp_*self.eye #*self.frc #*self.eterm3
      else:
         self.bop_si = taper(self.eterm1,rmin=self.botol,rmax=2.0*self.botol)*(self.eterm1-self.botol) # consist with GULP
         self.bop_pi = taper(self.eterm2,rmin=self.botol,rmax=2.0*self.botol)*self.eterm2
         self.bop_pp = taper(self.eterm3,rmin=self.botol,rmax=2.0*self.botol)*self.eterm3

      self.bop    = self.bop_si + self.bop_pi+self.bop_pp
      self.Deltap = torch.sum(self.bop,1)  

      if self.MessageFunction==1:
         self.D_si = [torch.sum(self.bop_si,1)]  
         self.D_pi = [torch.sum(self.bop_pi,1)]
         self.D_pp = [torch.sum(self.bop_pp,1)]

  def f1(self):
      Dv  = torch.unsqueeze(self.Deltap - self.P['val'],0)    # expand_dims(self.Deltap - self.P['val'],0)
      self.f2(Dv)
      self.f3(Dv)
      VAL      = torch.unsqueeze(self.P['val'],1)
      VALt     = torch.unsqueeze(self.P['val'],0)
      self.f_1 = 0.5*(DIV(VAL+self.f_2,  VAL+self.f_2+self.f_3)  + 
                      DIV(VALt+self.f_2, VALt+self.f_2+self.f_3))

  def f2(self,Dv):
      self.dexpf2  = torch.exp(-self.P['boc1']*Dv)
      self.f_2     = torch.add(self.dexpf2,torch.transpose(self.dexpf2,1,0))

  def f3(self,Dv):
      self.dexpf3 = torch.exp(-self.P['boc2']*Dv)
      delta_exp   = self.dexpf3+torch.transpose(self.dexpf3,1,0)

      self.f3log  = torch.log(0.5*delta_exp )
      self.f_3    = torch.div(-1.0,self.P['boc2'])*self.f3log

  def f45(self):
      self.D_boc = self.Deltap - self.P['valboc'] # + self.p['val_'+atomi]
      
      self.DELTA  = torch.unsqueeze(self.D_boc,1)
      self.DELTAt = torch.transpose(self.DELTA,1,0)
      
      self.df4 = self.P['boc4']*torch.square(self.bop)-self.DELTA
      self.f4r = torch.exp(-self.P['boc3']*(self.df4)+self.P['boc5'])

      self.df5 = self.P['boc4']*torch.square(self.bop)-self.DELTAt
      self.f5r = torch.exp(-self.P['boc3']*(self.df5)+self.P['boc5'])

      self.f_4 = torch.div(1.0,1.0+self.f4r)
      self.f_5 = torch.div(1.0,1.0+self.f5r)

  def get_bondorder(self):
      self.f1()
      self.f45()

      f11    = torch.where(self.P['ovcorr']>=0.0001,self.f_1,1.0)
      f12    = torch.where(torch.logical_and(self.P['ovcorr']>=0.0001,self.P['corr13']>=0.0001),
                         self.f_1,1.0)
      F11        = f11*f12
      F45_       = self.f_4 * self.f_5
      F45        = torch.where(self.P['corr13']>=0.0001,F45_,1.0)
      self.F     = F11*F45
      self.bo0   = self.bop*f11*F45  # -0.001  # consistent with GULP ## here is very strange!!

      self.bo    = torch.nn.functional.relu(self.bo0 - self.atol) # bond-order cut-off 0.001 reaxffatol
      self.bopi  = self.bop_pi * self.F
      self.bopp  = self.bop_pp * self.F
      self.bosi  = torch.nn.functional.relu(self.bo0 - self.bopi - self.bopp)
      self.bso   = self.P['ovun1'] * self.P['Desi'] * self.bo0
      self.Delta = torch.sum(self.bo0, 1)

      self.powb  = torch.pow(self.bosi+self.safety_value,self.P['be2'])
      self.expb  = torch.exp(torch.mul(self.P['be1'],1.0-self.powb))
      self.sieng = self.P['Desi']*self.bosi*self.expb 

      self.pieng = torch.mul(self.P['Depi'],self.bopi)
      self.ppeng = torch.mul(self.P['Depp'],self.bopp)
      self.esi   = self.sieng + self.pieng + self.ppeng

  def f_nn(self,pre,x,layer=5):
      X   = torch.unsqueeze(torch.stack(x,dim=2),2)

      o   =  []
      o.append(torch.sigmoid(torch.matmul(X,self.m[pre+'wi'])+self.m[pre+'bi']))  
                                                                       # input layer
      for l in range(layer):                                           # hidden layer  
          o.append(torch.sigmoid(torch.matmul(o[-1],self.m[pre+'w'][l])+self.m[pre+'b'][l]))
      
      o_  = torch.sigmoid(torch.matmul(o[-1],self.m[pre+'wo']) + self.m[pre+'bo']) 
      out = torch.squeeze(o_)                                          # output layer
      return out

  def message_passing(self):
      self.H         = []    # hiden states (or embeding states)
      self.D         = []    # degree matrix
      self.Hsi       = []
      self.Hpi       = []
      self.Hpp       = []
      self.H.append(self.bop)                   # 
      self.Hsi.append(self.bop_si)              #
      self.Hpi.append(self.bop_pi)              #
      self.Hpp.append(self.bop_pp)              # 
      self.D.append(self.Deltap)                # get the initial hidden state H[0]

      for t in range(1,self.messages+1):
          Di        = torch.unsqueeze(self.D[t-1],0)*self.eye
          Dj        = torch.unsqueeze(self.D[t-1],1)*self.eye

          if self.MessageFunction==1:
             Dsi_i = torch.unsqueeze(self.D_si[t-1],0)*self.eye - self.Hsi[t-1]
             Dsi_j = torch.unsqueeze(self.D_si[t-1],1)*self.eye - self.Hsi[t-1]

             Dpi_i = torch.unsqueeze(self.D_pi[t-1],0)*self.eye - self.Hpi[t-1]
             Dpi_j = torch.unsqueeze(self.D_pi[t-1],1)*self.eye - self.Hpi[t-1]

             Dpp_i = torch.unsqueeze(self.D_pp[t-1],0)*self.eye - self.Hpp[t-1]
             Dpp_j = torch.unsqueeze(self.D_pp[t-1],1)*self.eye - self.Hpp[t-1]

             Dpii  = Dpi_i + Dpp_i
             Dpij  = Dpi_j + Dpp_j
             Fi  = self.f_nn('fm',[Dsi_i,Dpii,self.H[t-1],Dsi_j,Dpij],layer=self.mf_layer[1])
             #  Fi  = self.f_nn('fm',[Dsi_j,Dsi_i,self.Hsi[t-1],  # +str(t)
             #                        Dpi_j,Dpi_i,self.Hpi[t-1],
             #                        Dpp_j,Dpp_i,self.Hpp[t-1]],
             #                        layer=self.mf_layer[1])
             Fj  = torch.transpose(Fi,1,0)
             F   = Fi*Fj
             Fsi = F[:,:,0]
             Fpi = F[:,:,1]
             Fpp = F[:,:,2]
             self.Hsi.append(self.Hsi[t-1]*Fsi)
             self.Hpi.append(self.Hpi[t-1]*Fpi)
             self.Hpp.append(self.Hpp[t-1]*Fpp)
          elif self.MessageFunction==2:
             Dbi = Di - self.H[t-1]
             Dbj = Dj - self.H[t-1]
             Fi  = self.f_nn('fm',[Dbj,Dbi,self.Hsi[t-1],self.Hpi[t-1],self.Hpp[t-1]], # +str(t)
                             layer=self.mf_layer[1])
             Fj  = torch.transpose(Fi,1,0)
             F   = Fi*Fj
             Fsi = F[:,:,0]
             Fpi = F[:,:,1]
             Fpp = F[:,:,2]
             self.Hsi.append(self.Hsi[t-1]*Fsi)
             self.Hpi.append(self.Hpi[t-1]*Fpi)
             self.Hpp.append(self.Hpp[t-1]*Fpp)
          elif self.MessageFunction==3:
             Dbi = Di  - self.H[t-1] # torch.unsqueeze(self.P['valboc'],0) 
             Dbj = Dj  - self.H[t-1] # torch.unsqueeze(self.P['valboc'],1)   
             Fi  = self.f_nn('fm',[Dbj,self.H[t-1],Dbi],layer=self.mf_layer[1]) # +str(t)
             #Fj = self.f_nn('f'+str(t),[Dbi,self.H[t-1],Dbj],layer=self.mf_layer[1])
             Fj  = torch.transpose(Fi,1,0)
             F   = Fi*Fj
             Fsi = F[:,:,0]
             Fpi = F[:,:,1]
             Fpp = F[:,:,2]
             self.Hsi.append(self.Hsi[t-1]*Fsi)
             self.Hpi.append(self.Hpi[t-1]*Fpi)
             self.Hpp.append(self.Hpp[t-1]*Fpp)
          elif self.MessageFunction==4:
             Dbi = Di  - torch.unsqueeze(self.P['val'],0)
             Dbj = Dj  - torch.unsqueeze(self.P['val'],1)
             Fi  = self.f_nn('fm',[Dbj,self.H[t-1],Dbi],layer=self.mf_layer[1]) # +str(t)
             Fj  = self.f_nn('fm',[Dbi,self.H[t-1],Dbj],layer=self.mf_layer[1]) # +str(t)

            #  self.f1()
            #  f11 = self.f_1*self.f_1
            #  F11 = torch.where(self.P['ovcorr']>=0.0001,f11,1.0)

             F   = Fi*Fj #*F11
             self.Hsi.append(self.Hsi[t-1]*F)
             self.Hpi.append(self.Hpi[t-1]*F)
             self.Hpp.append(self.Hpp[t-1]*F)
          elif self.MessageFunction==5:
             Dbi = Di  - torch.unsqueeze(self.P['val'],0) # Di  - self.H[t-1]
             Dbj = Dj  - torch.unsqueeze(self.P['val'],1) # Dj  - self.H[t-1]
             Fi  = self.f_nn('fm',[Dbj,self.H[t-1],Dbi],layer=self.mf_layer[1]) # +str(t)
             Fj  = self.f_nn('fm',[Dbi,self.H[t-1],Dbj],layer=self.mf_layer[1]) # +str(t)
             #Fj  = torch.transpose(Fi,2,0)
             F   = Fi*Fj
             Fsi = F[:,:,0]
             Fpi = F[:,:,1]
             Fpp = F[:,:,2]
             self.Hsi.append(Fsi)
             self.Hpi.append(Fpi)
             self.Hpp.append(Fpp)
          else:
             raise NotImplementedError('-  Message function not supported yet!')
          self.H.append(self.Hsi[t]+self.Hpi[t]+self.Hpp[t])
          self.D.append(torch.sum(self.H[t],1) )
          if self.MessageFunction==1:
             self.D_si.append(torch.sum(self.Hsi[t],1))
             self.D_pi.append(torch.sum(self.Hpi[t],1))
             self.D_pp.append(torch.sum(self.Hpp[t],1))

  def get_bondorder_nn(self):
      self.message_passing()
      self.bosi  = self.Hsi[-1]       # getting the final state
      self.bopi  = self.Hpi[-1]
      self.bopp  = self.Hpp[-1]

      self.bo0   = self.H[-1] # torch.where(self.H[-1]<0.000001,0.0,self.H[-1])
      # self.fbo   = taper(self.bo0,rmin=self.botol,rmax=2.0*self.botol)
      self.bo    = torch.nn.functional.relu(self.bo0 - self.atol*self.eye)      #bond-order cut-off 0.001 reaxffatol
      self.bso   = self.P['ovun1']*self.P['Desi']*self.bo0  
      self.Delta = torch.sum(self.bo0,1)   

      self.Di    = torch.unsqueeze(self.Delta,0)*self.eye          # get energy layer
      self.Dj    = torch.unsqueeze(self.Delta,1)*self.eye
      Dbi        = self.Di - self.bo0
      Dbj        = self.Dj - self.bo0

      if self.EnergyFunction==0:
         self.powb  = torch.pow(self.bosi+self.safety_value,self.P['be2'])
         self.expb  = torch.exp(torch.mul(self.P['be1'],1.0-self.powb))
         self.sieng = self.P['Desi']*self.bosi*self.expb 

         self.pieng = torch.mul(self.P['Depi'],self.bopi)
         self.ppeng = torch.mul(self.P['Depp'],self.bopp)
         self.esi   = self.sieng + self.pieng + self.ppeng
      elif self.EnergyFunction==1: 
         esi      = self.f_nn('fe',[self.bosi,self.bopi,self.bopp],layer=self.be_layer[1])
         self.esi = esi*torch.where(self.bo0<0.0000001,0.0,1.0)
      elif self.EnergyFunction==2:
         self.esi = self.f_nn('fe',[-self.bosi,-self.bopi,-self.bopp],layer=self.be_layer[1])
         self.esi = self.esi*tf.where(self.bo0<0.0000001,0.0,1.0)
      elif self.EnergyFunction==3: 
         e_ = self.f_nn('fe',[self.bosi,self.bopi,self.bopp],layer=self.be_layer[1])
         self.esi = self.bo0*e_
      elif self.EnergyFunction==4:
         Fi = self.f_nn('fe',[Dbj,Dbi,self.bo0],layer=self.be_layer[1])
         Fj = torch.transpose(Fi,1,0)
         self.esi  = Fi*Fj*self.bo0
      elif self.EnergyFunction==5:
         # r_        = self.bodiv1
         # mors_exp1 = torch.exp(self.P['be2']*(1.0-r_))
         # mors_exp2 = torch.square(mors_exp1) 

         # mors_exp10= torch.exp(self.P['be2']*self.P['be1'])
         # mors_exp20= torch.square(mors_exp10) 
         # emorse    = 2.0*mors_exp1 - mors_exp2 + mors_exp20 - 2.0*mors_exp10
         # self.esi  = torch.nn.ReLU(emorse)
         self.sieng = torch.mul(self.P['Desi'],self.bosi)
         self.pieng = torch.mul(self.P['Depi'],self.bopi)
         self.ppeng = torch.mul(self.P['Depp'],self.bopp)
         self.esi   = self.sieng + self.pieng - self.ppeng
      else:
         raise NotImplementedError('-  This method is not implimented!')

  def get_ebond(self,cell,rcell,positions):
      self.vr    = fvr(positions)
      vrf        = torch.matmul(self.vr,rcell)

      vrf        = torch.where(vrf-0.5>0,vrf-1.0,vrf)
      vrf        = torch.where(vrf+0.5<0,vrf+1.0,vrf) 

      self.vr    = torch.matmul(vrf,cell)
      self.r     = torch.sqrt(torch.sum(self.vr*self.vr,2)+0.0000000001)

      self.get_bondorder_uc()

      if self.nn:
         self.get_bondorder_nn()
      else:
         self.get_bondorder()

      self.Dv     = self.Delta - self.P['val']
      self.Dpi   = torch.sum(self.bopi+self.bopp,1) 

      self.so    = torch.sum(self.P['ovun1']*self.P['Desi']*self.bo0,1)  
      self.fbo   = taper(self.bo0,rmin=self.atol,rmax=2.0*self.atol) 
      self.fhb   = taper(self.bo0,rmin=self.hbtol,rmax=2.0*self.hbtol) 

      if self.EnergyFunction>=1: # or self.EnergyFunction==2 or self.EnergyFunction==3:
         self.ebond = - self.P['Desi']*self.esi
      else:
         self.ebond = -self.esi

      self.Ebond = 0.5*torch.sum(self.ebond)
      return self.Ebond


  def get_elone(self):
      self.NLPOPT  = 0.5*(self.P['vale'] - self.P['val'])

      # if self.nn:
      #    self.Delta_e = 0.5*(self.P['vale'] - self.Delta)
      #    self.nlp     = self.Delta_e
      # else:
      self.Delta_e = 0.5*(self.Delta - self.P['vale'])
      self.DE      = torch.nn.functional.relu(-torch.ceil(self.Delta_e))  # number of lone pair electron
      self.nlp     = self.DE + torch.exp(-self.P['lp1']*4.0*torch.square(1.0+self.Delta_e+self.DE))
      
      self.Delta_lp= self.NLPOPT- self.nlp   
      self.Dlp     = self.Delta - self.P['val'] - self.Delta_lp   
      self.Dpil    = torch.sum(torch.unsqueeze(self.Dlp,0)*(self.bopi+self.bopp),1)
      
      Delta_lp     = torch.nn.functional.relu(self.Delta_lp+1.0) -1.0
      self.explp   = 1.0+torch.exp(-75.0*Delta_lp) # -self.P['lp3']
      self.elone   = torch.div(self.P['lp2']*self.Delta_lp,self.explp)
      self.Elone   = torch.sum(self.elone)


  def get_eover(self):
      self.lpcorr= self.Delta_lp/(1.0+self.P['ovun3']*torch.exp(self.P['ovun4']*self.Dpil))
      self.Delta_lpcorr = self.Dv - self.lpcorr

      D_         = self.Delta_lpcorr+self.P['val']

      self.otrm1 = DIV_IF(1.0,D_)
      self.otrm2 = 1.0/(1.0+torch.exp(self.P['ovun2']*self.Delta_lpcorr))
      self.eover = self.so*self.otrm1*self.Delta_lpcorr*self.otrm2
      self.Eover = torch.sum(self.eover)


  def get_eunder(self):
      self.expeu1 = torch.exp(self.P['ovun6']*self.Delta_lpcorr)
      self.eu1    = torch.sigmoid(self.P['ovun2']*self.Delta_lpcorr)

      self.expeu3 = torch.exp(self.P['ovun8']*self.Dpil)
      self.eu2    = 1.0/(1.0+self.P['ovun7']*self.expeu3)
      self.eunder = -self.P['ovun5']*(1.0-self.expeu1)*self.eu1*self.eu2   
      self.Eunder = torch.sum(self.eunder)


  def get_theta(self):
      Rij = self.r[self.angi,self.angj]  
      Rjk = self.r[self.angj,self.angk]  
      # Rik = self.r[self.angi,self.angk]  
      vik = self.vr[self.angi,self.angj] + self.vr[self.angj,self.angk]
      Rik = torch.sqrt(torch.sum(torch.square(vik),1))

      Rij2= Rij*Rij
      Rjk2= Rjk*Rjk
      Rik2= Rik*Rik

      self.cos_theta = (Rij2+Rjk2-Rik2)/(2.0*Rij*Rjk)
      self.theta     = torch.acos(self.cos_theta)


  def get_theta0(self,dang):
      sbo   = self.Dpi[self.angj]
      pbo   = self.PBO[self.angj]
      rnlp  = self.nlp[self.angj]
      self.pbo = pbo
      self.rnlp= rnlp
      
      # if self.nn:
      #    SBO= sbo   
      # else:
      SBO= sbo - (1.0-pbo)*(dang+self.P['val8']*rnlp)  
      self.SBO= SBO 
      
      ok    = torch.logical_and(SBO<=1.0,SBO>0.0)
      S1    = torch.where(ok,SBO,torch.full_like(SBO,0.0))         #  0< sbo < 1                  
      SBO01 = torch.where(ok,torch.pow(S1,self.P['val9']),torch.full_like(S1,0.0)) 

      ok    = torch.logical_and(SBO<2.0,SBO>1.0)
      S2    = torch.where(ok,SBO,torch.full_like(SBO,0.0))                     
      F2    = torch.where(ok,torch.full_like(S2,1.0),torch.full_like(S2,0.0))          #  1< sbo <2
     
      S2    = 2.0*F2-S2  
      SBO12 = torch.where(ok,2.0-torch.pow(S2,self.P['val9']),torch.full_like(S2,0.0)) #  1< sbo <2
      SBO2  = torch.where(SBO>2.0,torch.full_like(S2,1.0),torch.full_like(S2,0.0))                         #     sbo >2

      self.SBO3  = SBO01+SBO12+2.0*SBO2
      # if self.nn:
      #    thet_ = torch.mul(self.P['theta0'],(1.0-torch.exp(-self.P['val10']*(2.0-self.SBO3))))
      # else:
      thet_      = 180.0 - torch.mul(self.P['theta0'],(1.0-torch.exp(-self.P['val10']*(2.0-self.SBO3))))
      self.thet0 = thet_/57.29577951


  def get_eangle(self):
      self.Dang  = self.Delta - self.P['valang']
      self.boaij = self.bo[self.angi,self.angj]
      self.boajk = self.bo[self.angj,self.angk]
      fij        = self.fbo[self.angi,self.angj]   
      fjk        = self.fbo[self.angj,self.angk]   
      self.fijk  = fij*fjk
      
      dang       = self.Dang[self.angj]
      PBOpow     = -torch.pow(self.bo+self.safety_value,8)  # bo0
      PBOexp     = torch.exp(PBOpow)
      self.PBO   = torch.prod(PBOexp,1)

      self.get_theta()
      self.get_theta0(dang)

      self.thet  = self.thet0-self.theta
      self.expang= torch.exp(-self.P['val2']*torch.square(self.thet))
      self.f7(self.boaij,self.boajk)
      self.f8(dang)
      self.eang  = self.fijk*self.f_7*self.f_8*(self.P['val1']-self.P['val1']*self.expang) 
      self.Eang  = torch.sum(self.eang)

      self.get_epenalty(self.boaij,self.boajk)
      self.get_three_conj(self.boaij,self.boajk)


  def f7(self,boij,bojk):
      self.expaij = torch.exp(-self.P['val3']*torch.pow(boij+self.safety_value,self.P['val4']))
      self.expajk = torch.exp(-self.P['val3']*torch.pow(bojk+self.safety_value,self.P['val4']))
      fi          = 1.0 - self.expaij
      fk          = 1.0 - self.expajk
      self.f_7    = fi*fk


  def f8(self,dang):
      exp6     = torch.exp( self.P['val6']*dang)
      exp7     = torch.exp(-self.P['val7']*dang)
      self.f_8 = self.P['val5'] - (self.P['val5'] - 1.0)*(2.0+exp6)/(1.0+exp6+exp7)


  def get_epenalty(self,boij,bojk):
      self.f9()
      expi = torch.exp(-self.P['pen2']*torch.square(boij-2.0))
      expk = torch.exp(-self.P['pen2']*torch.square(bojk-2.0))
      self.epen = self.P['pen1']*self.f_9*expi*expk*self.fijk
      self.Epen = torch.sum(self.epen)


  def f9(self):
      D    = torch.squeeze(self.Dv[self.angj])
      exp3 = torch.exp(-self.P['pen3']*D)
      exp4 = torch.exp( self.P['pen4']*D)
      self.f_9 = torch.div(2.0+exp3,1.0+exp3+exp4)


  def get_three_conj(self,boij,bojk):
      Dcoa_ = self.Delta-self.P['valboc']
      Dcoa  = Dcoa_[self.angj]
      Di    = self.Delta[self.angi]
      Dk    = self.Delta[self.angk]
      self.expcoa1 = torch.exp(self.P['coa2']*Dcoa)

      texp0 = torch.div(self.P['coa1'],1.0+self.expcoa1)  
      texp1 = torch.exp(-self.P['coa3']*torch.square(Di-boij))
      texp2 = torch.exp(-self.P['coa3']*torch.square(Dk-bojk))
      texp3 = torch.exp(-self.P['coa4']*torch.square(boij-1.5))
      texp4 = torch.exp(-self.P['coa4']*torch.square(bojk-1.5))
      self.etcon = texp0*texp1*texp2*texp3*texp4*self.fijk
      self.Etcon = torch.sum(self.etcon)
  

  def get_torsion_angle(self):
      rij = self.r[self.tori,self.torj]
      rjk = self.r[self.torj,self.tork]
      rkl = self.r[self.tork,self.torl]

      vrjk= self.vr[self.torj,self.tork]
      vrkl= self.vr[self.tork,self.torl]

      vrjl= vrjk + vrkl
      rjl = torch.sqrt(torch.sum(torch.square(vrjl),1))

      vrij= self.vr[self.tori,self.torj]
      vril= vrij + vrjl
      ril = torch.sqrt(torch.sum(torch.square(vril),1))

      vrik= vrij + vrjk
      rik = torch.sqrt(torch.sum(torch.square(vrik),1))

      rij2= torch.square(rij)
      rjk2= torch.square(rjk)
      rkl2= torch.square(rkl)
      rjl2= torch.square(rjl)
      ril2= torch.square(ril)
      rik2= torch.square(rik)

      c_ijk = (rij2+rjk2-rik2)/(2.0*rij*rjk)
      c2ijk = torch.square(c_ijk)
      # tijk  = tf.acos(c_ijk)
      cijk  =  1.000001 - c2ijk
      self.s_ijk = torch.sqrt(cijk)

      c_jkl = (rjk2+rkl2-rjl2)/(2.0*rjk*rkl)
      c2jkl = torch.square(c_jkl)
      cjkl  = 1.000001  - c2jkl 
      self.s_jkl = torch.sqrt(cjkl)

      c_ijl = (rij2+rjl2-ril2)/(2.0*rij*rjl)
      c_kjl = (rjk2+rjl2-rkl2)/(2.0*rjk*rjl)

      c2kjl = torch.square(c_kjl)
      ckjl  = 1.000001 - c2kjl 
      s_kjl = torch.sqrt(ckjl)

      fz    = rij2+rjl2-ril2-2.0*rij*rjl*c_ijk*c_kjl
      fm    = rij*rjl*self.s_ijk*s_kjl

      fm    = torch.where(torch.logical_and(fm<=0.000001,fm>=-0.000001),torch.full_like(fm,1.0),fm)
      fac   = torch.where(torch.logical_and(fm<=0.000001,fm>=-0.000001),torch.full_like(fm,0.0),
                                                                     torch.full_like(fm,1.0))
      cos_w = 0.5*fz*fac/fm
      #cos_w= cos_w*ccijk*ccjkl
      cos_w = torch.where(cos_w>0.9999999,torch.full_like(cos_w,1.0),cos_w)   
      self.cos_w = torch.where(cos_w<-0.999999,torch.full_like(cos_w,-1.0),cos_w)
      self.w= torch.acos(self.cos_w)
      self.cos2w = torch.cos(2.0*self.w)


  def get_etorsion(self):
      self.get_torsion_angle()

      self.botij = self.bo[self.tori,self.torj]
      self.botjk = self.bo[self.torj,self.tork]
      self.botkl = self.bo[self.tork,self.torl]
      fij        = self.fbo[self.tori,self.torj]
      fjk        = self.fbo[self.torj,self.tork]
      fkl        = self.fbo[self.tork,self.torl]
      self.fijkl = fij*fjk*fkl

      Dj         = self.Dang[self.torj]
      Dk         = self.Dang[self.tork]

      self.f10(self.botij,self.botjk,self.botkl)
      self.f11(Dj,Dk)

      self.bopjk = self.bopi[self.torj,self.tork]   #   different from reaxff manual
      self.expv2 = torch.exp(self.P['tor1']*torch.square(2.0-self.bopjk-self.f_11)) 

      self.cos3w = torch.cos(3.0*self.w)
      self.v1 = 0.5*self.P['V1']*(1.0+self.cos_w)  
      self.v2 = 0.5*self.P['V2']*self.expv2*(1.0-self.cos2w)
      self.v3 = 0.5*self.P['V3']*(1.0+self.cos3w)

      self.etor = self.fijkl*self.f_10*self.s_ijk*self.s_jkl*(self.v1+self.v2+self.v3)
      self.Etor = torch.sum(self.etor)
      self.get_four_conj(self.botij,self.botjk,self.botkl)


  def f10(self,boij,bojk,bokl):
      exp1 = 1.0 - torch.exp(-self.P['tor2']*boij)
      exp2 = 1.0 - torch.exp(-self.P['tor2']*bojk)
      exp3 = 1.0 - torch.exp(-self.P['tor2']*bokl)
      self.f_10 = exp1*exp2*exp3


  def f11(self,Dj,Dk):
      delt    = Dj+Dk
      f11exp3 = torch.exp(-self.P['tor3']*delt)
      f11exp4 = torch.exp( self.P['tor4']*delt)
      self.f_11 = torch.div(2.0+f11exp3,1.0+f11exp3+f11exp4)


  def get_four_conj(self,boij,bojk,bokl):
      exptol= torch.exp(-self.P['cot2']*torch.square(torch.tensor(self.atol - 1.5)))
      expij = torch.exp(-self.P['cot2']*torch.square(boij-1.5))-exptol
      expjk = torch.exp(-self.P['cot2']*torch.square(bojk-1.5))-exptol 
      expkl = torch.exp(-self.P['cot2']*torch.square(bokl-1.5))-exptol

      self.f_12  = expij*expjk*expkl
      self.prod  = 1.0+(torch.square(torch.cos(self.w))-1.0)*self.s_ijk*self.s_jkl
      self.efcon = self.fijkl*self.f_12*self.P['cot1']*self.prod  
      self.Efcon = torch.sum(self.efcon)


  def f13(self,r):
      rr = torch.pow(r,self.P['vdw1'])+torch.pow(torch.div(1.0,self.P['gammaw']),self.P['vdw1'])
      f_13 = torch.pow(rr,torch.div(1.0,self.P['vdw1']))  
      return f_13


  def get_tap(self,r):
      tpc = 1.0+torch.div(-35.0,self.vdwcut**4.0)*torch.pow(r,4.0)+ \
            torch.div(84.0,self.vdwcut**5.0)*torch.pow(r,5.0)+ \
            torch.div(-70.0,self.vdwcut**6.0)*torch.pow(r,6.0)+ \
            torch.div(20.0,self.vdwcut**7.0)*torch.pow(r,7.0)
      if self.vdwnn:
         if self.VdwFunction==1:
            tp = self.f_nn('fv',[r],layer=self.vdw_layer[1]) # /self.P['rvdw']
         elif self.VdwFunction==2:
            tpi = self.f_nn('fv',[r,self.Di,self.Dj],layer=self.vdw_layer[1]) 
            # tpj = self.f_nn('fv',[r,dj,di],layer=self.vdw_layer[1]) 
            tpj = torch.transpose(tpi,1,0)
            tp  = tpi*tpj
         else:
            raise RuntimeError('-  This method not implimented!')
      else:
         tp = tpc
      return tp,tpc


  def get_evdw(self,cell_tensor):
      self.evdw = 0.0
      self.ecoul= 0.0
      nc = 0
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell = cell_tensor[0]*i + cell_tensor[1]*j+cell_tensor[2]*k
                  vr_  = self.vr + cell
                  r    = torch.sqrt(torch.sum(torch.square(vr_),2)+self.safety_value)

                  gm3  = torch.pow(torch.div(1.0,self.P['gamma']),3.0)
                  r3   = torch.pow(r,3.0)
                  fv_   = torch.where(torch.logical_and(r>0.0000001,r<=self.vdwcut),torch.full_like(r,1.0),
                                                                                   torch.full_like(r,0.0))
                  if nc<13:
                     fv = fv_*self.d1
                  else:
                     fv = fv_*self.d2

                  f_13 = self.f13(r)
                  tpv,tpc  = self.get_tap(r)

                  expvdw1 = torch.exp(0.5*self.P['alfa']*(1.0-torch.div(f_13,2.0*self.P['rvdw'])))
                  expvdw2 = torch.square(expvdw1) 
                  self.evdw  += fv*tpv*self.P['Devdw']*(expvdw2-2.0*expvdw1)

                  rth         = torch.pow(r3+gm3,1.0/3.0)                                      # ecoul
                  self.ecoul += torch.div(fv*tpc*self.qij,rth)
                  nc += 1

      self.Evdw  = torch.sum(self.evdw)
      self.Ecoul = torch.sum(self.ecoul)

  
  def get_ehb(self,cell_tensor):
      self.BOhb   = self.bo0[self.hbi,self.hbj]
      fhb         = self.fhb[self.hbi,self.hbj]

      rij         = self.r[self.hbi,self.hbj]
      rij2        = torch.square(rij)
      vrij        = self.vr[self.hbi,self.hbj]
      vrjk_       = self.vr[self.hbj,self.hbk]
      self.Ehb    = 0.0
      
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell   = cell_tensor[0]*i + cell_tensor[1]*j+cell_tensor[2]*k
                  vrjk   = vrjk_ + cell
                  rjk2   = torch.sum(torch.square(vrjk),axis=1)
                  rjk    = torch.sqrt(rjk2)
                  
                  vrik   = vrij + vrjk
                  rik2   = torch.sum(torch.square(vrik),axis=1)
                  rik    = torch.sqrt(rik2)

                  cos_th = (rij2+rjk2-rik2)/(2.0*rij*rjk)
                  hbthe  = 0.5-0.5*cos_th
                  frhb   = rtaper(rik,rmin=self.hbshort,rmax=self.hblong)

                  exphb1 = 1.0-torch.exp(-self.P['hb1']*self.BOhb)
                  hbsum  = torch.div(self.P['rohb'],rjk)+torch.div(rjk,self.P['rohb'])-2.0
                  exphb2 = torch.exp(-self.P['hb2']*hbsum)
               
                  sin4   = torch.square(hbthe)
                  ehb    = fhb*frhb*self.P['Dehb']*exphb1*exphb2*sin4 
                  self.Ehb += torch.sum(ehb)


  def get_eself(self):
      chi    = np.expand_dims(self.P['chi'],axis=0)
      mu     = np.expand_dims(self.P['mu'],axis=0)
      self.eself = self.q*(chi+self.q*mu)
      self.Eself = torch.from_numpy(np.sum(self.eself,axis=1))


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

      self.get_evdw(cell)

      if self.nhb>0:
         self.get_ehb(cell)
      else:
         self.Ehb   = 0.0
         
      self.get_eself()

      E = self.Ebond + self.Elone + self.Eover + self.Eunder + \
               self.Eang + self.Epen + self.Etcon + \
               self.Etor + self.Efcon + self.Evdw + self.Ecoul + \
               self.Ehb + self.Eself + self.zpe
      return E


  def calculate(self,atoms=None,properties=["energy", "forces", "stress","pressure"], 
                system_changes=all_changes):
      Calculator.calculate(self, atoms, properties, system_changes)
      
      cell      = atoms.get_cell()                    # cell is object now
      cell      = cell[:].astype(dtype=np.float64)
      rcell     = np.linalg.inv(cell).astype(dtype=np.float64)

      positions = atoms.get_positions()
      xf        = np.dot(positions,rcell)
      xf        = np.mod(xf,1.0)
      positions = np.dot(xf,cell).astype(dtype=np.float64)

      self.get_charge(cell,positions)
      self.get_neighbor(cell,rcell,positions)

      cell = torch.tensor(cell)
      rcell= torch.tensor(rcell)

      if self.autograd:
         self.positions = torch.tensor(positions,requires_grad=True)
         E = self.get_total_energy(cell,rcell,self.positions)
         grad = torch.autograd.grad(outputs=E,
                                    inputs=self.positions,
                                    only_inputs=True)
      
         self.grad              = grad[0].numpy()
         self.E                 = E.detach().numpy()[0]
      else:
         self.positions = torch.from_numpy(positions)
         E              = self.get_total_energy(cell,rcell,self.positions)
         self.E         = E.numpy()[0]
         self.grad      = 0.0

      self.results['energy'] = self.E
      self.results['forces'] = -self.grad
      if self.CalStress:
         self.results['stress'] = self.calculate_numerical_stress(atoms,d=5e-5,voigt=True)
         stress  = self.results['stress']
         nonzero = 0
         stre_   = 0.0
         for _ in range(3):
             if abs(stress[_])>0.0000001:
                nonzero += 1
                stre_   += -stress[_]
         self.results['pressure'] = stre_*self.GPa/nonzero


  def get_free_energy(self,atoms=None,BuildNeighbor=False):
      cell      = atoms.get_cell()                    # cell is object now
      cell      = cell[:].astype(dtype=np.float64)
      rcell     = np.linalg.inv(cell).astype(dtype=np.float64)

      positions = atoms.get_positions()
      xf        = np.dot(positions,rcell)
      xf        = np.mod(xf,1.0)
      positions = np.dot(xf,cell).astype(dtype=np.float64)

      self.get_charge(cell,positions)
      if BuildNeighbor:
         self.get_neighbor(cell,rcell,positions)

      cell = torch.tensor(cell)
      rcell= torch.tensor(rcell)

      self.positions = torch.from_numpy(positions)
      E              = self.get_total_energy(cell,rcell,self.positions)
      return E


  def calculate_numerical_stress(self,atoms,d=1e-6,voigt=True,scale_atoms=False):
      """Calculate numerical stress using finite difference."""
      stress = np.zeros((3, 3), dtype=float)
      cell   = atoms.cell.copy()
      V      = atoms.get_volume()

      for i in range(3):
          x = np.eye(3)
          x[i, i] += d
          atoms.set_cell(np.dot(cell, x), scale_atoms=scale_atoms)
          eplus = self.get_free_energy(atoms=atoms)

          x[i, i] -= 2 * d
          atoms.set_cell(np.dot(cell, x), scale_atoms=scale_atoms)
          eminus = self.get_free_energy(atoms=atoms)

          stress[i, i] = (eplus - eminus) / (2 * d * V)
          x[i, i] += d

          j = i - 2
          x[i, j] = d
          x[j, i] = d
          atoms.set_cell(np.dot(cell, x), scale_atoms=scale_atoms)
          eplus = self.get_free_energy(atoms=atoms)

          x[i, j] = -d
          x[j, i] = -d
          atoms.set_cell(np.dot(cell, x), scale_atoms=scale_atoms)
          eminus = self.get_free_energy(atoms=atoms)

          stress[i, j] = (eplus - eminus) / (4 * d * V)
          stress[j, i] = stress[i, j]
      atoms.set_cell(cell, scale_atoms=True)

      if voigt:
         return stress.flat[[0, 4, 8, 5, 2, 1]]
      else:
         return stress


  def check_hb(self):
      if 'H' in self.spec:
         for sp1 in self.spec:
             if sp1 != 'H':
                for sp2 in self.spec:
                    if sp2 != 'H':
                       hb = sp1+'-H-'+sp2
                       if hb not in self.Hbs:
                          self.Hbs.append(hb) # 'rohb','Dehb','hb1','hb2'
                          self.p['rohb_'+hb] = 1.9
                          self.p['Dehb_'+hb] = 0.0
                          self.p['hb1_'+hb]  = 2.0
                          self.p['hb2_'+hb]  = 19.0


  def check_offd(self):
      p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
      for key in p_offd:
          for sp in self.spec:
              try:
                 self.p[key+'_'+sp+'-'+sp]  = self.p[key+'_'+sp]  
              except KeyError:
                 print('-  warning: key not in dict') 

      for bd in self.bonds:             # check offd parameters
          b= bd.split('-')
          if 'rvdw_'+bd not in self.p:
             for key in p_offd:        # set offd parameters according combine rules
                 if self.p[key+'_'+b[0]]>0.0 and self.p[key+'_'+b[1]]>0.0:
                    self.p[key+'_'+bd] = np.sqrt(self.p[key+'_'+b[0]]*self.p[key+'_'+b[1]])
                 else:
                    self.p[key+'_'+bd] = -1.0

      for bd in self.bonds:             # check minus ropi ropp parameters
          if self.p['ropi_'+bd]<0.0:
             self.p['ropi_'+bd] = 0.3*self.p['rosi_'+bd]
             self.p['bo3_'+bd]  = -50.0
             self.p['bo4_'+bd]  = 0.0
          if self.p['ropp_'+bd]<0.0:
             self.p['ropp_'+bd] = 0.2*self.p['rosi_'+bd]
             self.p['bo5_'+bd]  = -50.0
             self.p['bo6_'+bd]  = 0.0


  def set_p(self,m,bo_layer):
      ''' setting up parameters '''
      self.unit   = 4.3364432032e-2
      self.punit  = ['Desi','Depi','Depp','lp2','ovun5','val1',
                     'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb']
      p_bond = ['Desi','Depi','Depp','be1','bo5','bo6','ovun1',
                'be2','bo3','bo4','bo1','bo2',
                'Devdw','rvdw','alfa','rosi','ropi','ropp',
                'corr13','ovcorr']
      p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
      self.P = {}

      if not self.nn:
         self.p['lp3'] = 75.0
      # else:
      #    self.hbtol = self.p['hbtol']
      #    self.atol = self.p['acut']  

      self.rcbo = np.zeros([self.natom,self.natom],dtype=np.float64)
      self.r_cut = np.zeros([self.natom,self.natom],dtype=np.float64)
      self.r_cuta = np.zeros([self.natom,self.natom],dtype=np.float64)

      for i in range(self.natom):
          for j in range(self.natom):
              bd = self.atom_name[i] + '-' + self.atom_name[j]
              if not bd in self.bonds:
                 bd = self.atom_name[j] + '-' + self.atom_name[i]
              self.rcbo[i][j] = min(self.rcut[bd],self.rc_bo[bd])   #  ###### TODO #####

              if i!=j:
                 self.r_cut[i][j]  = self.rcut[bd]  
                 self.r_cuta[i][j] = self.rcuta[bd] 
              # if i<j:  self.nbe0[bd] += 1
      self.rcbo_tensor = torch.from_numpy(self.rcbo)

      p_spec = ['valang','valboc','val','vale',
                'lp2','ovun5',                 # 'val3','val5','boc3','boc4','boc5'
                'ovun2','atomic',
                'mass','chi','mu']             # 'gamma','gammaw','Devdw','rvdw','alfa'

      for key in p_spec:
          unit_ = self.unit if key in self.punit else 1.0
          self.P[key] = np.zeros([self.natom],dtype=np.float64)
          for i in range(self.natom):
                sp = self.atom_name[i]
                self.P[key][i] = self.p[key+'_'+sp]*unit_
          self.P[key] = torch.tensor(self.P[key])

      self.zpe = -torch.sum(self.P['atomic']) + self.emol


      for key in ['boc3','boc4','boc5','gamma','gammaw']:
          self.P[key] = np.zeros([self.natom,self.natom],dtype=np.float64)
          for i in range(self.natom):
              for j in range(self.natom):
                  self.P[key][i][j] = np.sqrt(self.p[key+'_'+self.atom_name[i]]*self.p[key+'_'+self.atom_name[j]],
                                              dtype=np.float64)
          self.P[key] = torch.tensor(self.P[key])
      
      for key in p_bond:
          unit_ = self.unit if key in self.punit else 1.0
          self.P[key] = np.zeros([self.natom,self.natom],dtype=np.float64)
          for i in range(self.natom):
              for j in range(self.natom):
                  bd = self.atom_name[i] + '-' + self.atom_name[j]
                  if bd not in self.bonds:
                     bd = self.atom_name[j] + '-' + self.atom_name[i]
                  self.P[key][i][j] = self.p[key+'_'+bd]*unit_
          self.P[key] = torch.tensor(self.P[key])
       
      p_g  = ['boc1','boc2','coa2','ovun6','lp1','lp3',
              'ovun7','ovun8','val6','val9','val10','tor2',
              'tor3','tor4','cot2','coa4','ovun4',               
              'ovun3','val8','coa3','pen2','pen3','pen4','vdw1'] 
      for key in p_g:
          self.P[key] = self.p[key]

      self.p_ang  = ['theta0','val1','val2','coa1','val7','val4','pen1'] 
      self.p_hb   = ['rohb','Dehb','hb1','hb2']
      self.p_tor  = ['V1','V2','V3','tor1','cot1']  
      tors        = self.check_tors(self.p_tor)
 
      for key in self.p_ang:
          unit_ = self.unit if key in self.punit else 1.0
          for a in self.angs:
              pn = key + '_' + a
              self.p[pn] = self.p[pn]*unit_

      for key in self.p_tor:
          unit_ = self.unit if key in self.punit else 1.0
          for t in tors:
              pn = key + '_' + t
              self.p[pn] = self.p[pn]*unit_

      for h in self.Hbs:
          pn = 'Dehb_' + h
          self.p[pn] = self.p[pn]*self.unit

      self.d1  = torch.tensor(np.triu(np.ones([self.natom,self.natom],dtype=np.float64),k=0))
      self.d2  = torch.tensor(np.triu(np.ones([self.natom,self.natom],dtype=np.float64),k=1))
      self.eye = torch.tensor(1.0 - np.eye(self.natom,dtype=np.float64))

      if self.nn:
         self.set_m(m)


  def set_m(self,m):
      self.m = {}
      if self.BOFunction==0:
         pres = ['fe','fm']
      else:
         pres = ['fe','fsi','fpi','fpp','fm']
         
      if self.vdwnn:
         pres.append('fv')
      # for t in range(1,self.messages+1):
      #     pres.append('fm')
      for k_ in pres:
          for k in ['wi','bi','wo','bo']:
              key = k_+k
              self.m[key] = []
              for i in range(self.natom):
                  mi_ = []
                  for j in range(self.natom):
                      if k_ in ['fe','fsi','fpi','fpp','fv']:
                         bd = self.atom_name[i] + '-' + self.atom_name[j]
                         if bd not in self.bonds:
                            bd = self.atom_name[j] + '-' + self.atom_name[i]
                      else:
                         bd = self.atom_name[i] 
                      key_ = key+'_'+bd
                      if key_ in m:
                         if k in ['bi','bo']:
                            mi_.append(np.expand_dims(m[key_],axis=0))
                         else:
                            mi_.append(m[key_])
                  self.m[key].append(mi_)
              self.m[key] = torch.tensor(np.array(self.m[key]),dtype=torch.double)

          for k in ['w','b']:
              key = k_+k
              self.m[key] = []

              if k_ in ['fesi','fepi','fepp','fe']:
                 layer_ = self.be_layer[1]
              elif k_ in ['fsi','fpi','fpp']:
                 layer_ = self.bo_layer[1]
              elif k_ =='fv':
                 layer_ = self.vdw_layer[1]
              else:
                 layer_ = self.mf_layer[1]

              for l in range(layer_):
                  m_ = []
                  for i in range(self.natom):
                      mi_ = []
                      for j in range(self.natom):
                          if k_ in ['fe','fsi','fpi','fpp','fv']:
                             bd = self.atom_name[i] + '-' + self.atom_name[j]
                             if bd not in self.bonds:
                                bd = self.atom_name[j] + '-' + self.atom_name[i]
                          else:
                             bd = self.atom_name[i] 
                          key_ = key+'_'+bd
                          if key_ in m:
                             if k == 'b':
                                mi_.append(np.expand_dims(m[key+'_'+bd][l],axis=0))
                             else:
                                mi_.append(m[key+'_'+bd][l])
                      m_.append(mi_)
                  self.m[key].append(torch.tensor(np.array(m_),dtype=torch.double))

  def init_bonds(self):
      self.bonds,self.offd,self.angs,self.torp,self.Hbs = [],[],[],[],[]
      for key in self.p:
          k = key.split('_')
          if k[0]=='bo1':
             self.bonds.append(k[1])
          elif k[0]=='rosi':
             kk = k[1].split('-')
             if len(kk)==2:
                self.offd.append(k[1])
          elif k[0]=='theta0':
             self.angs.append(k[1])
          elif k[0]=='tor1':
             self.torp.append(k[1])
          elif k[0]=='rohb':
             self.Hbs.append(k[1])
      self.torp = self.checkTors(self.torp)

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
      return torp 

  def check_tors(self,p_tor):
      tors = []          ### check torsion parameter
      for spi in self.spec:
          for spj in self.spec:
              for spk in self.spec:
                  for spl in self.spec:
                      tor = spi+'-'+spj+'-'+spk+'-'+spl
                      if tor not in tors:
                         tors.append(tor)

      for key in p_tor:
          for tor in tors:
              if tor not in self.torp:
                 [t1,t2,t3,t4] = tor.split('-')
                 tor1 =  t1+'-'+t3+'-'+t2+'-'+t4
                 tor2 =  t4+'-'+t3+'-'+t2+'-'+t1
                 tor3 =  t4+'-'+t2+'-'+t3+'-'+t1
                 tor4 = 'X'+'-'+t2+'-'+t3+'-'+'X'
                 tor5 = 'X'+'-'+t3+'-'+t2+'-'+'X'
                 if tor1 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor1]
                 elif tor2 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor2]
                 elif tor3 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor3]    
                 elif tor4 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor4]  
                 elif tor5 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor5]     
                 else:
                    self.p[key+'_'+tor] = 0.0
      return tors
      
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
         fmd.write('\nAtomID Sym   Delta   NLP    DLPC   -\n')
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

if __name__ == '__main__':
   A  = read('poscar.gen')
   rf = IRFF(atoms=A,libfile='ffield')


