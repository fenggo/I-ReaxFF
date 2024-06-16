import json as js
import numpy as np
from ase import Atoms
from ase.io import read,write
from ase.calculators.calculator import Calculator, all_changes
from .qeq import qeq
from .RadiusCutOff import setRcut
from .reaxfflib import read_ffield,write_lib
from .reax_force_data import reax_force_data,Dataset
from .neighbors import get_neighbors,get_pangle,get_ptorsion,get_phb
from .set_matrix_tensor import set_matrix
#from torch.autograd import Variable
import torch
from torch import nn

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
    xi  = x.unsqueeze(1)
    xj  = x.unsqueeze(2) 
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

def fmessage(pre,bd,nbd,x,m,batch=50,layer=5):
    ''' Dimention: (nbatch,3) input = 3
                Wi:  (4,8) 
                Wh:  (8,8)
                Wo:  (8,3)  output = 3
    '''
    X   = torch.unsqueeze(torch.stack(x,dim=2),dim=2)
    # print(X.shape)
    # print(m[pre+'wi_'+bd].shape)
    # X   = tf.stack(x_,axis=1)       # Dimention: (nbatch,4)
    #                                 #        Wi:  (4,8) 
    o   =  []                         #        Wh:  (8,8)
    o.append(torch.sigmoid(torch.matmul(X,m[pre+'wi_'+bd])))   # input layer

    for l in range(layer):                                                   # hidden layer      
        o.append(torch.sigmoid(torch.matmul(o[-1],m[pre+'w_'+bd][l])+m[pre+'b_'+bd][l]))

    out = torch.sigmoid(torch.matmul(o[-1],m[pre+'wo_'+bd]) + m[pre+'bo_'+bd])  # output layer
    # print(out.shape)
    return  out.squeeze(dim=2) 

def fnn(pre,bd,x,m,layer=5):
    ''' Dimention: (nbatch,3) input = 3
                Wi:  (3,8) 
                Wh:  (8,8)
                Wo:  (8,1)  output = 3
    '''
    X   = torch.unsqueeze(torch.stack(x,dim=2),dim=2)
    #                                 #        Wi:  (3,8) 
    o   =  []                         #        Wh:  (8,8)
    o.append(torch.sigmoid(torch.matmul(X,m[pre+'wi_'+bd])))   # input layer

    for l in range(layer):                                     # hidden layer      
        o.append(torch.sigmoid(torch.matmul(o[-1],m[pre+'w_'+bd][l])+m[pre+'b_'+bd][l]))

    out = torch.sigmoid(torch.matmul(o[-1],m[pre+'wo_'+bd]) + m[pre+'bo_'+bd])  # output layer
    # print(out.shape)
    return  out.squeeze(dim=[2,3]) 

class ReaxFF_nn_force(nn.Module):
  ''' Force Learning '''
  name = "ReaxFF_nn"
  implemented_properties = ["energy", "forces"]
  def __init__(self,dataset={},
               batch=200,
               sample='uniform',
               libfile='ffield.json',
               vdwcut=10.0,
               messages=1,
               hbshort=6.75,hblong=7.5,
               mf_layer=None,be_layer=None,
               be_universal=None,mf_universal=None,
               cons=['val','vale','valang','vale','lp3','cutoff','hbtol'],# 'acut''val','valboc',
               opt=None,# optword='nocoul',
               bdopt=None,mfopt=None,beopt=None,
               eaopt=[],
               nomb=False,              # this option is used when deal with metal system
               autograd=True):
      super(ReaxFF_nn_force, self).__init__()
      self.dataset      = dataset 
      self.batch_size   = batch
      self.sample       = sample        # uniform or random
      self.opt          = opt
      self.bdopt        = bdopt
      self.mfopt        = mfopt
      self.beopt        = beopt
      self.eaopt        = eaopt
      self.cons         = cons
      self.mf_layer     = mf_layer
      self.be_layer     = be_layer
      self.mf_universal = mf_universal
      self.be_universal = be_universal
      # self.atoms      = atoms
      # self.cell       = atoms.get_cell()
      # self.atom_name  = self.atoms.get_chemical_symbols()
      # self.natom      = len(self.atom_name)
      # self.spec       = []
      self.hbshort      = hbshort
      self.hblong       = hblong
      self.vdwcut       = vdwcut

      self.m_,self.rcut,self.rcuta,re  = self.read_ffield(libfile)
      if self.m_ is not None:
         self.nn        = True          # whether use neural network
      self.set_p()
      self.get_data()
      self.stack_tensor()

      self.results      = {}
      self.EnergyFunction = 0
      self.autograd     = autograd
      self.nomb         = nomb # without angle, torsion and hbond manybody term
      self.messages     = messages 
      self.safety_value = 0.000000001
      self.GPa          = 1.60217662*1.0e2
      self.set_memory()

      # self.params = nn.Parameter(torch.rand(3, 3), requires_grad=True)
      # self.Qe= qeq(p=self.p,atoms=self.atoms)

  def forward(self):
      for st in self.strcs:
          self.get_bond_energy(st)   # get bond energy for every structure
          self.get_atomic_energy(st)
          self.get_threebody_energy(st)
          self.get_fourbody_energy(st)
        #   self.get_vdw_energy(st)
        #   self.get_hb_energy(st)
        #   self.get_total_energy(st)
      return self.E,self.force
  
  def get_atomic_energy(self,st):
      ''' compute atomic energy of structure (st): elone, eover,eunder'''
      # st_ = st.split('-')[0]
      self.Elone[st]  = torch.zeros_like(self.Delta[st])
      self.Eover[st]  = torch.zeros_like(self.Delta[st])
      self.Eunder[st] = torch.zeros_like(self.Delta[st])
      self.Nlp[st]    = torch.zeros_like(self.Delta[st])

      for sp in self.spec:
          delta    = self.Delta[st][:,self.s[st][sp]]
          delta_pi = self.Delta_pi[st][:,self.s[st][sp]]
          so       = self.SO[st][:,self.s[st][sp]]

          delta_lp,nlp,Elone     = self.get_elone(sp,delta) 
          delta_lpcorr,Eover = self.get_eover(sp,delta,delta_lp,delta_pi,so) 
          Eunder             = self.get_eunder(sp,delta_lpcorr,delta_pi) 
          
          self.Nlp[st][:,self.s[st][sp]]     = nlp
          self.Elone[st][:,self.s[st][sp]]   = Elone
          self.Eover[st][:,self.s[st][sp]]   = Eover
          self.Eunder[st][:,self.s[st][sp]]  = Eunder

      self.elone[st]  = torch.sum(self.Elone[st],1)
      self.eover[st]  = torch.sum(self.Eover[st],1)
      self.eunder[st] = torch.sum(self.Eunder[st],1)

      self.eatomic[st] = torch.tensor(0.0)
      for sp in self.spec:
          if self.ns[st][sp]>0:
             self.eatomic[st] += self.p['atomic_'+sp]*self.ns[st][sp]
      self.zpe[st]    = self.eatomic[st] + self.estruc[st]
  
  def get_eover(self,sp,delta,delta_lp,delta_pi,so):
      delta_lpcorr = delta - self.p['val_'+sp] - torch.divide(delta_lp,
                     1.0+self.p['ovun3']*torch.exp(self.p['ovun4']*delta_pi))
      otrm1              = DIV_IF(1.0,delta_lpcorr + self.p['val_'+sp])
      otrm2              = torch.sigmoid(-self.p['ovun2_'+sp]*delta_lpcorr)
      Eover              = so*otrm1*delta_lpcorr*otrm2
      return delta_lpcorr,Eover 
  
  def get_eunder(self,sp,delta_lpcorr,delta_pi):
      expeu1            = torch.exp(self.p['ovun6']*delta_lpcorr)
      eu1               = torch.sigmoid(self.p['ovun2_'+sp]*delta_lpcorr)
      expeu3            = torch.exp(self.p['ovun8']*delta_pi)
      eu2               = torch.divide(1.0,1.0+self.p['ovun7']*expeu3)
      Eunder            = -self.p['ovun5_'+sp]*(1.0-expeu1)*eu1*eu2                          # must positive
      return Eunder 
  
  def get_elone(self,sp,delta):
      Nlp            = 0.5*(self.p['vale_'+sp] - self.p['val_'+sp])
      delta_e        = 0.5*(delta - self.p['vale_'+sp])
      De             = -torch.relu(-torch.ceil(delta_e)) 
      nlp            = -De + torch.exp(-self.p['lp1']*4.0*torch.square(1.0+delta_e-De))

      Delta_lp       = Nlp - nlp           
      Delta_lp       = torch.relu(Delta_lp+1) -1

      explp          = 1.0+torch.exp(-75.0*Delta_lp) # -self.p['lp3']
      Elone          = self.p['lp2_'+sp]*Delta_lp/explp
      return Delta_lp,nlp,Elone
                                        
  def get_bond_energy(self,st):
      vr         = fvr(self.x[st])
      vrf        = torch.matmul(vr,self.rcell[st])
      vrf        = torch.where(vrf-0.5>0,vrf-1.0,vrf)
      vrf        = torch.where(vrf+0.5<0,vrf+1.0,vrf) 
      vr         = torch.matmul(vrf,self.cell[st])
      self.r[st] = torch.sqrt(torch.sum(vr*vr,dim=3)) # +0.0000000001
      
      self.get_bondorder_uc(st)
      self.message_passing(st)
      self.get_final_state(st)
      
      self.ebd[st] = torch.zeros_like(self.bosi[st])
      ebd  = []
      bosi = self.bosi[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]
      bopi = self.bopi[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]
      bopp = self.bopp[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]

      for bd in self.bonds:
          nbd_ = self.nbd[st][bd]
          if nbd_==0:
             continue
          b_  = self.b[st][bd]
          bosi_ = bosi[:,b_[0]:b_[1]]
          bopi_ = bopi[:,b_[0]:b_[1]]
          bopp_ = bopp[:,b_[0]:b_[1]]

          esi = fnn('fe',bd,[bosi_,bopi_,bopp_],
                    self.m,layer=self.be_layer[1])
          ebd.append(-self.p['Desi_'+bd]*esi)

      self.ebd[st][:,self.bdid[st][:,0],self.bdid[st][:,1]] = torch.cat(ebd,dim=1)
      self.ebond[st]= torch.sum(self.ebd[st],dim=[1,2])
  
  def get_bondorder_uc(self,st):
      bop_si,bop_pi,bop_pp = [],[],[]
      # print(self.r[st])
      r = self.r[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]
      self.bop_si[st] = torch.zeros_like(self.r[st])
      self.bop_pi[st] = torch.zeros_like(self.r[st])
      self.bop_pp[st] = torch.zeros_like(self.r[st])

      for bd in self.bonds:
          nbd_ = self.nbd[st][bd]
          b_   = self.b[st][bd]
          if nbd_==0:
             continue
          rbd = r[:,b_[0]:b_[1]]
          bodiv1 = torch.div(rbd,self.p['rosi_'+bd])
          bopow1 = torch.pow(bodiv1,self.p['bo2_'+bd])
          eterm1 = (1.0+self.botol)*torch.exp(torch.mul(self.p['bo1_'+bd],bopow1)) 

          bodiv2 = torch.div(rbd,self.p['ropi_'+bd])
          bopow2 = torch.pow(bodiv2,self.p['bo4_'+bd])
          eterm2 = torch.exp(torch.mul(self.p['bo3_'+bd],bopow2))

          bodiv3 = torch.div(rbd,self.p['ropp_'+bd])
          bopow3 = torch.pow(bodiv3,self.p['bo6_'+bd])
          eterm3 = torch.exp(torch.mul(self.p['bo5_'+bd],bopow3))
          
          bop_si.append(taper(eterm1,rmin=self.botol,rmax=2.0*self.botol)*(eterm1-self.botol)) # consist with GULP
          bop_pi.append(taper(eterm2,rmin=self.botol,rmax=2.0*self.botol)*eterm2)
          bop_pp.append(taper(eterm3,rmin=self.botol,rmax=2.0*self.botol)*eterm3)
      
      self.bop_si[st][:,self.bdid[st][:,0],self.bdid[st][:,1]] = self.bop_si[st][:,self.bdid[st][:,1],self.bdid[st][:,0]] = torch.cat(bop_si,dim=1)
      self.bop_pi[st][:,self.bdid[st][:,0],self.bdid[st][:,1]] = self.bop_pi[st][:,self.bdid[st][:,1],self.bdid[st][:,0]] = torch.cat(bop_pi,dim=1)
      self.bop_pp[st][:,self.bdid[st][:,0],self.bdid[st][:,1]] = self.bop_pp[st][:,self.bdid[st][:,1],self.bdid[st][:,0]] = torch.cat(bop_pp,dim=1)
      self.bop[st]    = self.bop_si[st] + self.bop_pi[st] + self.bop_pp[st]
      # print(self.bop[st].size)
      self.Deltap[st] = torch.sum(self.bop[st],2)
      self.D_si[st]   = torch.sum(self.bop_si[st],2)
      self.D_pi[st]   = torch.sum(self.bop_pi[st],2)
      self.D_pp[st]   = torch.sum(self.bop_pp[st],2)

  def message_passing(self,st):
      self.H[st]    = [self.bop[st]]                     # 
      self.Hsi[st]  = [self.bop_si[st]]                  #
      self.Hpi[st]  = [self.bop_pi[st]]                  #
      self.Hpp[st]  = [self.bop_pp[st]]                  # 
      self.D[st]    = [self.Deltap[st]]    
      
      for t in range(1,self.messages+1):
          Di   = torch.unsqueeze(self.D[st][t-1],1)*self.eye[st]
          Dj   = torch.unsqueeze(self.D[st][t-1],2)*self.eye[st]

          Dbi  = Di  - self.H[st][t-1] 
          Dbj  = Dj  - self.H[st][t-1]

          Dbi_ = Dbi[:,self.bdid[st][:,0],self.bdid[st][:,1]]
          Dbj_ = Dbj[:,self.bdid[st][:,0],self.bdid[st][:,1]]
          H    = self.H[st][t-1][:,self.bdid[st][:,0],self.bdid[st][:,1]]
          Hsi  = self.Hsi[st][t-1][:,self.bdid[st][:,0],self.bdid[st][:,1]]
          Hpi  = self.Hpi[st][t-1][:,self.bdid[st][:,0],self.bdid[st][:,1]]
          Hpp  = self.Hpp[st][t-1][:,self.bdid[st][:,0],self.bdid[st][:,1]]

          bo,bosi,bopi,bopp = self.get_bondorder(st,t,Dbi_,H,Dbj_,Hsi,Hpi,Hpp)
          
          self.H[st].append(bo)                      # get the hidden state H[t]
          self.Hsi[st].append(bosi)
          self.Hpi[st].append(bopi)
          self.Hpp[st].append(bopp)

          Delta = torch.sum(bo,2)
          self.D[st].append(Delta)                  # degree matrix

  def get_final_state(self,st):     
      self.Delta[st]  = self.D[st][-1]
      self.bo0[st]    = self.H[st][-1]                 # fetch the final state 
      self.bosi[st]   = self.Hsi[st][-1]
      self.bopi[st]   = self.Hpi[st][-1]
      self.bopp[st]   = self.Hpp[st][-1]

      self.bo[st]     = torch.relu(self.bo0[st] - self.atol)

      bso             = []
      bo0             = self.bo0[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]

      for bd in self.bonds:
          if self.nbd[st][bd]==0:
             continue
          b_   = self.b[st][bd]
          bo0_ = bo0[:,b_[0]:b_[1]]
          bso.append(self.p['ovun1_'+bd]*self.p['Desi_'+bd]*bo0_)
      
      bso_             = torch.zeros_like(self.bo0[st])
      bso_[:,self.bdid[st][:,0],self.bdid[st][:,1]]  = torch.cat(bso,1)
      bso_[:,self.bdid[st][:,1],self.bdid[st][:,0]]  = torch.cat(bso,1)

      self.SO[st]      = torch.sum(bso_,2)  
      self.Delta_pi[st]= torch.sum(self.bopi[st]+self.bopp[st],2) 

      self.fbot[st]   = taper(self.bo0[st],rmin=self.atol,rmax=2.0*self.atol) 
      self.fhb[st]    = taper(self.bo0[st],rmin=self.hbtol,rmax=2.0*self.hbtol) 

  def get_bondorder(self,st,t,Dbi,H,Dbj,Hsi,Hpi,Hpp):
      ''' compute bond-order according the message function'''
      flabel  = 'fm'
      bosi = torch.zeros_like(self.r[st])
      bopi = torch.zeros_like(self.r[st])
      bopp = torch.zeros_like(self.r[st])

      bosi_ = []
      bopi_ = []
      bopp_ = []
      bso_  = []
      for bd in self.bonds:
          nbd_ = self.nbd[st][bd]
          if nbd_==0:
             continue
          b_   = self.b[st][bd]
          Di   = Dbi[:,b_[0]:b_[1]]
          Dj   = Dbj[:,b_[0]:b_[1]]

          h    = H[:,b_[0]:b_[1]]
          hsi  = Hsi[:,b_[0]:b_[1]]
          hpi  = Hpi[:,b_[0]:b_[1]]
          hpp  = Hpp[:,b_[0]:b_[1]]
          b    = bd.split('-')
 
          Fi   = fmessage(flabel,b[0],nbd_,[Di,h,Dj],self.m,
                          batch=self.batch[st],layer=self.mf_layer[1])
          Fj   = fmessage(flabel,b[1],nbd_,[Dj,h,Di],self.m,
                          batch=self.batch[st],layer=self.mf_layer[1])
          F    = Fi*Fj
          Fsi,Fpi,Fpp = torch.unbind(F,axis=2)

          bosi_.append(hsi*Fsi)
          bopi_.append(hpi*Fpi)
          bopp_.append(hpp*Fpp)

      bosi[:,self.bdid[st][:,0],self.bdid[st][:,1]] = bosi[:,self.bdid[st][:,1],self.bdid[st][:,0]] = torch.cat(bosi_,1)
      bopi[:,self.bdid[st][:,0],self.bdid[st][:,1]] = bopi[:,self.bdid[st][:,1],self.bdid[st][:,0]] = torch.cat(bopi_,1)
      bopp[:,self.bdid[st][:,0],self.bdid[st][:,1]] = bopp[:,self.bdid[st][:,1],self.bdid[st][:,0]] = torch.cat(bopp_,1)
      bo   = bosi+bopi+bopp
      return bo,bosi,bopi,bopp

  def get_threebody_energy(self,st):
      ''' compute three-body term interaction '''
      PBOpow        = -torch.pow(self.bo[st],8)        # original: self.BO0 
      PBOexp        =  torch.exp(PBOpow)
      self.Pbo[st]  =  torch.prod(PBOexp,2)     # BO Product

      if self.nang[st]==0:
         self.eang[st] = torch.zeros(self.batch[st]) 
         self.epen[st] = torch.zeros(self.batch[st]) 
         self.tconj[st]= torch.zeros(self.batch[st]) 
      else:
         Eang  = []
         Epen  = []
         Etcon = []
         for ang in self.angs:
             sp  = ang.split('-')[1]
             if self.na[st][ang]>0:
                ai        = np.squeeze(self.ang_i[st][self.a[st][ang][0]:self.a[st][ang][1]])
                aj        = np.squeeze(self.ang_j[st][self.a[st][ang][0]:self.a[st][ang][1]])
                ak        = np.squeeze(self.ang_k[st][self.a[st][ang][0]:self.a[st][ang][1]])
                boij      = self.bo[st][:,ai,aj]
                bojk      = self.bo[st][:,aj,ak]
                fij       = self.fbot[st][:,ai,aj]
                fjk       = self.fbot[st][:,aj,ak]
                delta     = self.Delta[st][:,aj]
                delta_i   = self.Delta[st][:,ai]
                delta_k   = self.Delta[st][:,ak]
                sbo       = self.Delta_pi[st][:,aj]
                pbo       = self.Pbo[st][:,aj]
                nlp       = self.Nlp[st][:,aj]
                theta     = self.theta[st][:,self.a[st][ang][0]:self.a[st][ang][1]]
                Ea,fijk,delta_ang  = self.get_eangle(sp,ang,boij,bojk,fij,fjk,theta,delta,sbo,pbo,nlp)
                Ep        = self.get_epenalty(ang,delta,boij,bojk,fijk)
                Et        = self.get_three_conj(ang,delta_ang,delta_i,delta_k,boij,bojk,fijk) 
                Eang.append(Ea)
                Epen.append(Ep)
                Etcon.append(Et)
                
         self.Eang[st] = torch.cat(Eang,dim=1)
         self.Epen[st] = torch.cat(Epen,dim=1)
         self.Etcon[st]= torch.cat(Etcon,dim=1)
         self.eang[st] = torch.sum(self.Eang[st],1)
         self.epen[st] = torch.sum(self.Epen[st],1)
         self.etcon[st]= torch.sum(self.Etcon[st],1)
 
  def get_eangle(self,sp,ang,boij,bojk,fij,fjk,theta,delta,sbo,pbo,nlp):
      delta_ang      = delta - self.p['valang_'+sp]
      # delta        = delta - self.p['val_'+sp]
      fijk           = fij*fjk

      theta0         = self.get_theta0(ang,delta_ang,sbo,pbo,nlp)
      thet           = theta0 - theta
      thet2          = torch.square(thet)

      expang         = torch.exp(-self.p['val2_'+ang]*thet2)
      f_7            = self.f7(sp,ang,boij,bojk)
      f_8            = self.f8(sp,ang,delta_ang)
      Eang           = fijk*f_7*f_8*(self.p['val1_'+ang]-self.p['val1_'+ang]*expang) 
      return Eang,fijk,delta_ang

  def get_theta0(self,ang,delta_ang,sbo,pbo,nlp):
      Sbo   = sbo - (1.0-pbo)*(delta_ang+self.p['val8']*nlp)    
      
      ok    = torch.logical_and(torch.less_equal(Sbo,1.0),torch.greater(Sbo,0.0))
      S1    = torch.where(ok,Sbo,torch.zeros_like(Sbo))    #  0< sbo < 1                  
      Sbo1  = torch.where(ok,torch.pow(S1,self.p['val9']),torch.zeros_like(S1)) 

      ok    = torch.logical_and(torch.less(Sbo,2.0),torch.greater(Sbo,1.0))
      S2    = torch.where(ok,Sbo,torch.zeros_like(Sbo))                     
      F2    = torch.where(ok,torch.ones_like(S2),torch.zeros_like(S2))                                    #  1< sbo <2
     
      S2    = 2.0*F2-S2  
      Sbo12 = torch.where(ok,2.0-torch.pow(S2,self.p['val9']),torch.zeros_like(Sbo))  #  1< sbo <2
                                                                                                 #     sbo >2
      Sbo2  = torch.where(torch.greater_equal(Sbo,2.0),
                          torch.ones_like(Sbo),torch.zeros_like(Sbo))

      Sbo3   = Sbo1 + Sbo12 + 2.0*Sbo2
      theta0_ = 180.0 - self.p['theta0_'+ang]*(1.0-torch.exp(-self.p['val10']*(2.0-Sbo3)))
      theta0 = theta0_/57.29577951
      return theta0

  def f7(self,sp,ang,boij,bojk): 
      Fboi  = torch.where(torch.greater(boij,0.0),
                          torch.ones_like(boij),torch.zeros_like(boij))   
      Fbori = 1.0 - Fboi                                                                         # prevent NAN error
      expij = torch.exp(-self.p['val3_'+sp]*torch.pow(boij+Fbori,self.p['val4_'+ang])*Fboi)

      Fbok  = torch.where(torch.greater(bojk,0.0),
                          torch.ones_like(bojk),torch.zeros_like(bojk))   
      Fbork = 1.0 - Fbok 
      expjk = torch.exp(-self.p['val3_'+sp]*torch.pow(bojk+Fbork,self.p['val4_'+ang])*Fbok)
      fi = 1.0 - expij
      fk = 1.0 - expjk
      F  = fi*fk
      return F 

  def f8(self,sp,ang,delta_ang):
      exp6 = torch.exp( self.p['val6']*delta_ang)
      exp7 = torch.exp(-self.p['val7_'+ang]*delta_ang)
      F    = self.p['val5_'+sp] - (self.p['val5_'+sp]-1.0)*torch.divide(2.0+exp6,1.0+exp6+exp7)
      return F

  def get_epenalty(self,ang,delta,boij,bojk,fijk):
      f_9  = self.f9(delta)
      expi = torch.exp(-self.p['pen2']*torch.square(boij-2.0))
      expk = torch.exp(-self.p['pen2']*torch.square(bojk-2.0))
      Ep   = self.p['pen1_'+ang]*f_9*expi*expk*fijk
      return Ep
  
  def f9(self,Delta):
      exp3 = torch.exp(-self.p['pen3']*Delta)
      exp4 = torch.exp( self.p['pen4']*Delta)
      F = torch.divide(2.0+exp3,1.0+exp3+exp4)
      return F

  def get_three_conj(self,ang,delta_ang,delta_i,delta_k,boij,bojk,fijk):
      delta_coa  = delta_ang # self.D_ang[st] + valang - valboc
      expcoa1    = torch.exp(self.p['coa2']*delta_coa)

    #   delta_i    = tf.gather_nd(self.Delta[st],self.ang_i[st])
    #   delta_k    = tf.gather_nd(self.Delta[st],self.ang_k[st])

      texp0 = torch.divide(self.p['coa1_'+ang],1.0 + expcoa1)  
      texp1 = torch.exp(-self.p['coa3']*torch.square(delta_i-boij))
      texp2 = torch.exp(-self.p['coa3']*torch.square(delta_k-bojk))
      texp3 = torch.exp(-self.p['coa4']*torch.square(boij-1.5))
      texp4 = torch.exp(-self.p['coa4']*torch.square(bojk-1.5))
      Etc   = texp0*texp1*texp2*texp3*texp4*fijk 
      return Etc

  def get_fourbody_energy(self,st):
      if self.optword.find('notor')>=0 or self.ntor[st]==0:
         self.etor[st] = torch.zeros([self.batch[st]])
         self.efcon[st]= torch.zeros([self.batch[st]])
      else:
         self.get_etorsion(st,tor)
         self.get_four_conj(st,cot1)

         self.etor[st] = tf.reduce_sum(input_tensor=self.Etor[st],axis=0,name='etor_%s' %mol)
         self.efcon[st]= tf.reduce_sum(input_tensor=self.Efcon[st],axis=0,name='efcon_%s' %mol)

  def get_etorsion(self,mol,tor1,V1,V2,V3):
      self.BOtij[st]  = tf.gather_nd(self.bo[st],self.tij[st])
      self.BOtjk[st]  = tf.gather_nd(self.bo[st],self.tjk[st])
      self.BOtkl[st]  = tf.gather_nd(self.bo[st],self.tkl[st])
      fij              = tf.gather_nd(self.fbot[st],self.tij[st])
      fjk              = tf.gather_nd(self.fbot[st],self.tjk[st])
      fkl              = tf.gather_nd(self.fbot[st],self.tkl[st])
      self.fijkl[st]  = fij*fjk*fkl

      Dj    = tf.gather_nd(self.Dang[st],self.tor_j[st])
      Dk    = tf.gather_nd(self.Dang[st],self.tor_k[st])

      self.f_10[st]   = self.f10(mol)
      self.f_11[st]   = self.f11(mol,Dj,Dk)

      self.BOpjk[st]  = tf.gather_nd(self.bopi[st],self.tjk[st]) 
      #   different from reaxff manual
      self.expv2[st]  = tf.exp(tor1*tf.square(2.0-self.BOpjk[st]-self.f_11[st])) 

      self.cos3w[st]  = tf.cos(3.0*self.w[st])
      v1 = 0.5*V1*(1.0+self.cos_w[st])   
      v2 = 0.5*V2*self.expv2[st]*(1.0-self.cos2w[st])
      v3 = 0.5*V3*(1.0+self.cos3w[st])
      self.Etor[st]=self.fijkl[st]*self.f_10[st]*self.s_ijk[st]*self.s_jkl[st]*(v1+v2+v3)

  def f10(self,mol):
      with tf.compat.v1.name_scope('f10_%s' %mol):
           exp1 = 1.0 - tf.exp(-self.p['tor2']*self.BOtij[st])
           exp2 = 1.0 - tf.exp(-self.p['tor2']*self.BOtjk[st])
           exp3 = 1.0 - tf.exp(-self.p['tor2']*self.BOtkl[st])
      return exp1*exp2*exp3

  def f11(self,mol,Dj,Dk):
      delt = Dj+Dk
      self.f11exp3[st] = tf.exp(-self.p['tor3']*delt)
      self.f11exp4[st] = tf.exp( self.p['tor4']*delt)
      f_11 = tf.math.divide(2.0+self.f11exp3[st],1.0+self.f11exp3[st]+self.f11exp4[st])
      return f_11

  def get_four_conj(self,mol,cot1):
      exptol= tf.exp(-self.p['cot2']*tf.square(self.atol - 1.5))
      expij = tf.exp(-self.p['cot2']*tf.square(self.BOtij[st]-1.5))-exptol
      expjk = tf.exp(-self.p['cot2']*tf.square(self.BOtjk[st]-1.5))-exptol 
      expkl = tf.exp(-self.p['cot2']*tf.square(self.BOtkl[st]-1.5))-exptol

      self.f_12[st] = expij*expjk*expkl
      prod = 1.0+(tf.square(tf.cos(self.w[st]))-1.0)*self.s_ijk[st]*self.s_jkl[st]
      self.Efcon[st] = self.fijkl[st]*self.f_12[st]*cot1*prod  

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


  def get_rcbo(self):
      ''' get cut-offs for individual bond '''
      self.rc_bo = {}
      for bd in self.bonds:
          b= bd.split('-')
          ofd=bd if b[0]!=b[1] else b[0]

          log_ = np.log((self.botol/(1.0+self.botol)))
          rr = log_/self.p_['bo1_'+bd] 
          self.rc_bo[bd]=self.p_['rosi_'+ofd]*np.power(log_/self.p_['bo1_'+bd],1.0/self.p_['bo2_'+bd])

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

  def check_hb(self):
      if 'H' in self.spec:
         for sp1 in self.spec:
             if sp1 != 'H':
                for sp2 in self.spec:
                    if sp2 != 'H':
                       hb = sp1+'-H-'+sp2
                       if hb not in self.Hbs:
                          self.Hbs.append(hb) # 'rohb','Dehb','hb1','hb2'
                          self.p_['rohb_'+hb] = 1.9
                          self.p_['Dehb_'+hb] = 0.0
                          self.p_['hb1_'+hb]  = 2.0
                          self.p_['hb2_'+hb]  = 19.0


  def check_offd(self):
      p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
      for key in p_offd:
          for sp in self.spec:
              self.p_[key+'_'+sp+'-'+sp]  = self.p_[key+'_'+sp]  

      for bd in self.bonds:             # check offd parameters
          b= bd.split('-')
          if 'rvdw_'+bd not in self.p_:
             for key in p_offd:        # set offd parameters according combine rules
                 if self.p_[key+'_'+b[0]]>0.0 and self.p_[key+'_'+b[1]]>0.0:
                    self.p_[key+'_'+bd] = np.sqrt(self.p_[key+'_'+b[0]]*self.p_[key+'_'+b[1]])
                 else:
                    self.p_[key+'_'+bd] = -1.0

      for bd in self.bonds:             # check minus ropi ropp parameters
          if self.p_['ropi_'+bd]<0.0:
             self.p_['ropi_'+bd] = 0.3*self.p_['rosi_'+bd]
             self.p_['bo3_'+bd]  = -50.0
             self.p_['bo4_'+bd]  = 0.0
          if self.p_['ropp_'+bd]<0.0:
             self.p_['ropp_'+bd] = 0.2*self.p_['rosi_'+bd]
             self.p_['bo5_'+bd]  = -50.0
             self.p_['bo6_'+bd]  = 0.0

  def set_p(self):
      ''' setting up parameters '''
      self.unit   = 4.3364432032e-2
      self.punit  = ['Desi','Depi','Depp','lp2','ovun5','val1',
                     'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb']
      ##  All Parameters
      self.p_bond = ['Desi','Depi','Depp','be1','bo5','bo6','ovun1',
                'be2','bo3','bo4','bo1','bo2',
                'Devdw','rvdw','alfa','rosi','ropi','ropp',
                'corr13','ovcorr']
      self.p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
      self.p_g  = ['boc1','boc2','coa2','ovun6','lp1','lp3',
                   'ovun7','ovun8','val6','tor2',
                   'val8','val9','val10',
                   'tor3','tor4','cot2','coa4','ovun4',               
                   'ovun3','val8','coa3','pen2','pen3','pen4',
                   'vdw1'] 
      self.p_spec = ['valang','valboc','val','vale',
                     'lp2','ovun5','val3','val5', # ,'boc3','boc4','boc5'
                     'ovun2','atomic',
                     'mass','chi','mu'] # 'gamma','gammaw','Devdw','rvdw','alfa'
      self.p_ang  = ['theta0','val1','val2','coa1','val7','val4','pen1'] 
      self.p_hb   = ['rohb','Dehb','hb1','hb2']
      self.p_tor  = ['V1','V2','V3','tor1','cot1']  
      if self.opt is None:
         self.opt = []
         for key in self.p_g:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_spec:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_bond:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_ang:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_tor:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_hb:
             if key not in self.cons:
                self.opt.append(key)

      
      self.botol        = 0.01*self.p_['cutoff']
      self.atol         = self.p_['acut']   # atol
      self.hbtol        = self.p_['hbtol']  # hbtol
      
      self.check_offd()
      self.check_hb()
      self.tors = self.check_tors(self.p_tor)
      self.get_rcbo()
      
      self.p            = {}   # training parameter 

      for key in self.p_g:
          unit_ = self.unit if key in self.punit else 1.0
          grad = True if key in self.opt else False
          self.p[key] = nn.Parameter(torch.tensor(self.p_[key]*unit_), 
                                     requires_grad=grad)
      for key in self.p_spec:
          unit_ = self.unit if key in self.punit else 1.0
          for sp in self.spec:
              key_ = key+'_'+sp
              grad = True if key in self.opt or key_ in self.opt else False
              self.p[key_] = nn.Parameter(torch.tensor(self.p_[key_]*unit_), 
                                         requires_grad=grad)
      
      for key in self.p_bond:
          unit_ = self.unit if key in self.punit else 1.0
          for bd in self.bonds:
              key_ = key+'_'+bd
              grad = True if key in self.opt or key_ in self.opt else False
              self.p[key_] = nn.Parameter(torch.tensor(self.p_[key+'_'+bd]*unit_), 
                                                requires_grad=grad)
      
      for key in self.p_ang:
          unit_ = self.unit if key in self.punit else 1.0
          for a in self.angs:
              key_ = key + '_' + a
              grad = True if key in self.opt or key_ in self.opt else False
              self.p[key_] = nn.Parameter(torch.tensor(self.p_[key_]*unit_),
                                          requires_grad=grad)

      for key in self.p_tor:
          unit_ = self.unit if key in self.punit else 1.0
          for t in self.tors:
              key_ = key + '_' + t
              grad = True if key in self.opt or key_ in self.opt else False
              self.p[key_] = nn.Parameter(torch.tensor(self.p_[key_]*unit_),
                                          requires_grad=grad)

      for key in self.p_hb:
          unit_ = self.unit if key in self.punit else 1.0
          for h in self.hbs:
              key_ = key + '_' + h
              grad = True if key in self.opt or key_ in self.opt else False
              self.p[key_] = nn.Parameter(torch.tensor(self.p_[key_]*unit_),
                                          requires_grad=grad)
   
      if self.nn:
         self.set_m()

  def init_bonds(self):
      self.bonds,self.offd,self.angs,self.torp,self.hbs = [],[],[],[],[]
      self.spec = []
      for key in self.p_:
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
             self.hbs.append(k[1])
          elif k[0]=='val':
             self.spec.append(k[1])
      self.torp = self.checkTors(self.torp)

  def checkTors(self,torp):
      tors_ = torp
      for tor in tors_:
          [t1,t2,t3,t4] = tor.split('-')
          tor1 = t1+'-'+t3+'-'+t2+'-'+t4
          tor2 = t4+'-'+t3+'-'+t2+'-'+t1
          tor3 = t4+'-'+t2+'-'+t3+'-'+t1

          if tor1 in torp and tor1!=tor:
             # print('-  dict %s is repeated, deleteting ...' %tor1)
             torp.remove(tor1)
          elif tor2 in self.torp and tor2!=tor:
             # print('-  dict %s is repeated, deleteting ...' %tor2)
             torp.remove(tor2)
          elif tor3 in self.torp and tor3!=tor:
             # print('-  dict %s is repeated, deleteting ...' %tor3)
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
                    self.p[key+'_'+tor] = self.p[key+'_'+tor1] # consistent with lammps
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
      
  def stack_tensor(self):
      self.x     = {}
      self.rcell = {}
      self.cell  = {}
      self.eye   = {}
      for st in self.strcs:
          self.x[st]     = torch.tensor(self.data[st].x,requires_grad=True)
          self.cell[st]  = torch.tensor(np.expand_dims(self.data[st].cell,axis=1))
          self.rcell[st] = torch.tensor(np.expand_dims(self.data[st].rcell,axis=1))
          self.eye[st]   = torch.tensor(np.expand_dims(1.0 - np.eye(self.natom[st]),axis=0))
    #   for key in self.p_spec:
    #       # unit_ = self.unit if key in self.punit else 1.0
    #       self.P[key] = np.zeros([self.natom],dtype=np.float64)
    #       self.P[key] = torch.tensor(self.P[key])

    #   for key in ['boc3','boc4','boc5','gamma','gammaw']:
    #       self.P[key] = np.zeros([self.natom,self.natom],dtype=np.float64)
    #       for i in range(self.natom):
    #           for j in range(self.natom):
    #               self.P[key][i][j] = np.sqrt(self.p[key+'_'+self.atom_name[i]]*self.p[key+'_'+self.atom_name[j]],
    #                                           dtype=np.float64)
    #       self.P[key] = torch.tensor(self.P[key])

    #   self.rcbo = np.zeros([self.natom,self.natom],dtype=np.float64)
    #   self.r_cut = np.zeros([self.natom,self.natom],dtype=np.float64)
    #   self.r_cuta = np.zeros([self.natom,self.natom],dtype=np.float64)

    #   for i in range(self.natom):
    #       for j in range(self.natom):
    #           bd = self.atom_name[i] + '-' + self.atom_name[j]
    #           if not bd in self.bonds:
    #              bd = self.atom_name[j] + '-' + self.atom_name[i]
    #           self.rcbo[i][j] = min(self.rcut[bd],self.rc_bo[bd])   #  ###### TODO #####

    #           if i!=j:
    #              self.r_cut[i][j]  = self.rcut[bd]  
    #              self.r_cuta[i][j] = self.rcuta[bd] 
    #           # if i<j:  self.nbe0[bd] += 1

    #   for key in self.p_bond:
    #       unit_ = self.unit if key in self.punit else 1.0
    #       self.P[key] = np.zeros([self.natom,self.natom],dtype=np.float64)
    #       for i in range(self.natom):
    #           for j in range(self.natom):
    #               bd = self.atom_name[i] + '-' + self.atom_name[j]
    #               if bd not in self.bonds:
    #                  bd = self.atom_name[j] + '-' + self.atom_name[i]
    #               self.P[key][i][j] = self.p[key+'_'+bd]*unit_
    #       self.P[key] = torch.tensor(self.P[key])

    #   self.rcbo_tensor = torch.from_numpy(self.rcbo)
    #   self.d1  = torch.tensor(np.triu(np.ones([self.natom,self.natom],dtype=np.float64),k=0))
    #   self.d2  = torch.tensor(np.triu(np.ones([self.natom,self.natom],dtype=np.float64),k=1))
    #   self.eye = torch.tensor(1.0 - np.eye(self.natom,dtype=np.float64))

  def get_data(self): 
      self.nframe      = 0
      strucs           = {}
      self.max_e       = {}
      self.cell        = {}
      self.strcs       = []
      self.batch       = {}
      self.eself,self.evdw_,self.ecoul_ = {},{},{}

      for st in self.dataset: 
          nindex = []
          for key in strucs:
              if self.dataset[key]==self.dataset[st]:
                 nindex.extend(strucs[key].indexs)
          data_ = reax_force_data(structure=st,
                                 traj=self.dataset[st],
                               vdwcut=self.vdwcut,
                                 rcut=self.rcut,
                                rcuta=self.rcuta,
                              hbshort=self.hbshort,
                               hblong=self.hblong,
                                batch=self.batch_size,
                       variable_batch=True,
                               sample=self.sample,
                       p=self.p_,spec=self.spec,bonds=self.bonds,
                  angs=self.angs,tors=self.tors,
                                  hbs=self.hbs,
                               nindex=nindex)

          if data_.status:
             self.strcs.append(st)
             strucs[st]      = data_
             self.batch[st]  = strucs[st].batch
             self.nframe    += self.batch[st]
             print('-  max energy of %s: %f.' %(st,strucs[st].max_e))
             self.max_e[st]  = strucs[st].max_e
             # self.evdw_[st]= strucs[st].evdw
             # self.ecoul_[st] = strucs[st].ecoul  
             self.eself[st]   = strucs[st].eself     
             self.cell[st]   = strucs[st].cell
          else:
             print('-  data status of %s:' %st,data_.status)
      self.nstrc  = len(strucs)
      self.generate_data(strucs)
      # self.memory(molecules=strucs)
      print('-  generating dataset ...')
      return strucs

  def generate_data(self,strucs):
      ''' get data '''
      self.dft_energy                  = {}
      self.blist,self.bdid             = {},{}
      self.dilink,self.djlink          = {},{}
      self.nbd,self.b,self.a,self.t    = {},{},{},{}
      self.ang_i,self.ang_j,self.ang_k = {},{},{}
      self.abij,self.abjk              = {},{}
      self.tij,self.tjk,self.tkl       = {},{},{}
      self.tor_j,self.tor_k            = {},{}
      self.tor_i,self.tor_l            = {},{}
      self.atom_name                   = {}
      self.natom                       = {}
      self.nang                        = {}
      self.ntor                        = {}
      self.ns                          = {}
      self.s                           = {s:[] for s in self.spec}
      self.nv                          = {}
      self.na                          = {}
      self.nt                          = {}
      self.nh                          = {}
      self.v                           = {}
      self.h                           = {}
      self.hij                         = {}
      self.s_ijk                       = {}
      self.s_jkl                       = {}
      self.w                           = {}
      self.data                        = {}
      self.theta                       = {}
      self.estruc                      = {}
      for s in strucs:
          self.natom[s]    = strucs[s].natom
          self.blist[s]    = strucs[s].blist
          self.dilink[s]   = strucs[s].dilink
          self.djlink[s]   = strucs[s].djlink

          self.nang[s]     = strucs[s].nang
          self.ang_j[s]    = np.expand_dims(strucs[s].ang_j,axis=1)
          self.ang_i[s]    = np.expand_dims(strucs[s].ang_i,axis=1)
          self.ang_k[s]    = np.expand_dims(strucs[s].ang_k,axis=1)
          #   self.abij[s] = strucs[s].abij
          #   self.abjk[s] = strucs[s].abjk

          #   self.tij[s]  = strucs[s].tij
          #   self.tjk[s]  = strucs[s].tjk
          #   self.tkl[s]  = strucs[s].tkl
          self.ntor[s]     = strucs[s].ntor
          self.tor_i[s]    = np.expand_dims(strucs[s].tor_i,axis=1)
          self.tor_j[s]    = np.expand_dims(strucs[s].tor_j,axis=1)
          self.tor_k[s]    = np.expand_dims(strucs[s].tor_k,axis=1)
          self.tor_l[s]    = np.expand_dims(strucs[s].tor_l,axis=1)

          self.nbd[s]      = strucs[s].nbd
          self.na[s]       = strucs[s].na
          self.nt[s]       = strucs[s].nt
          # self.nv[s]     = strucs[s].nv
          self.b[s]        = strucs[s].B
          self.a[s]        = strucs[s].A
          self.t[s]        = strucs[s].T
          # self.v[s]      = strucs[s].V
          # self.nh[s]     = strucs[s].nh
          # self.h[s]      = strucs[s].H
          # self.hij[s]    = strucs[s].hij
          self.bdid[s]     = strucs[s].bond  # bond index like pair (i,j).
          self.atom_name[s]= strucs[s].atom_name
          
          self.s[s]        = {sp:[] for sp in self.spec}
          for i,sp in enumerate(self.atom_name[s]):
              self.s[s][sp].append(i)
          self.ns[s]       = {sp:len(self.s[s][sp]) for sp in self.spec}

          self.data[s]     = Dataset(dft_energy=strucs[s].energy_dft,
                                     x=strucs[s].x,
                                     cell=strucs[s].cell,
                                     rcell=strucs[s].rcell,
                                     forces=strucs[s].forces,
                                     theta=strucs[s].theta,
                                     s_ijk=strucs[s].s_ijk,
                                     s_jkl=strucs[s].s_jkl,
                                     w=strucs[s].w)

          self.dft_energy[s] = torch.tensor(self.data[s].dft_energy)
          if self.nang[s]>0:
             self.theta[s] = torch.tensor(self.data[s].theta)

          if self.ntor[s]>0:
             self.s_ijk[s] = torch.tensor(self.data[s].s_ijk)
             self.s_jkl[s] = torch.tensor(self.data[s].s_jkl)
             self.w[s]     = torch.tensor(self.data[s].w)

        #   if self.nhb[s]>0:
        #      self.rhb[s]   = self.data[s].rhb
        #      self.frhb[s]  = self.data[s].frhb
        #      self.hbthe[s] = self.data[s].hbthe

          self.estruc[s] =  nn.Parameter(torch.tensor(0.0),requires_grad=True) 

  def set_m(self):
      self.m = set_matrix(self.m_,self.spec,self.bonds,
                          self.mfopt,self.beopt,self.bdopt,1,
                          (6,0),(6,0),0,0,
                          self.mf_layer,self.mf_layer_,self.MessageFunction_,self.MessageFunction,
                          self.be_layer,self.be_layer_,1,1,
                          (9,0),(9,0),1,1,
                          None,self.be_universal,self.mf_universal,None)
      
  def read_ffield(self,libfile):
      if libfile.endswith('.json'):
         lf                  = open(libfile,'r')
         j                   = js.load(lf)
         self.p_             = j['p']
         self.m_             = j['m']
         self.MolEnergy_     = j['MolEnergy']
         self.messages       = j['messages']
         self.BOFunction     = j['BOFunction']
         self.EnergyFunction_ = self.EnergyFunction = j['EnergyFunction']
         self.MessageFunction_=self.MessageFunction= j['MessageFunction']
         self.VdwFunction    = j['VdwFunction']
         self.mf_layer_      = j['mf_layer']
         self.be_layer_      = j['be_layer']
         rcut                = j['rcut']
         rcuta               = j['rcutBond']
         re                  = j['rEquilibrium']
         lf.close()
         self.init_bonds()
         self.emol = 0.0
      else:
         (self.p_,zpe_,self.spec,self.bonds,self.offd,self.angs,
          self.torp,self.hbs) = read_ffield(libfile=libfile,zpe=False)
         self.m_        = None
         self.mf_layer_ = None
         self.be_layer_ = None
         self.emol      = 0.0
         rcut           = None
         rcuta          = None
         re             = None
         self.vdwnn     = False
         self.EnergyFunction_ = 0
         self.MessageFunction_= 0
         self.VdwFunction     = 0
         self.p_['acut']   = 0.0001
         self.p_['hbtol']  = 0.0001
      if self.mf_layer is None:
         self.mf_layer = self.mf_layer_
      if self.be_layer is None:
         self.be_layer = self.be_layer_
      if self.m_ is None:
         self.nn=False
         
      # for sp in self.atom_name:
      #     if sp not in self.spec:
      #        self.spec.append(sp)
      return self.m_,rcut,rcuta,re
  
  def set_memory(self):
      self.r               = {}
      self.E               = {}
      self.force           = {}
      self.ebd,self.ebond   = {},{}
      self.bop,self.bop_si,self.bop_pi,self.bop_pp = {},{},{},{}
      self.bo,self.bo0,self.bosi,self.bopi,self.bopp = {},{},{},{},{}

      self.Deltap,self.Delta = {},{}
      self.D_si,self.D_pi,self.D_pp = {},{},{}
      self.D,self.H,self.Hsi,self.Hpi,self.Hpp = {},{},{},{},{}

      self.SO,self.Delta_pi   = {},{}
      self.fbot,self.fhb = {},{}

      self.Elone,self.Eover,self.Eunder = {},{},{}
      self.elone,self.eover,self.eunder = {},{},{}
      self.eatomic,self.zpe             = {},{}

      self.Eang,self.Epen,self.Etcon    = {},{},{}
      self.eang,self.epen,self.etcon    = {},{},{}

      self.Pbo,self.Nlp = {},{}


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

