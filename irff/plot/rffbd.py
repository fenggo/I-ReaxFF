#!/usr/bin/env python
from __future__ import print_function
from os.path import isfile
import json as js
from ase import Atoms
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase.visualize import view
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
import matplotlib.colors as col
import numpy as np
import argh
import argparse


p_name = ['boc1','boc2','coa2','trip4','trip3','kc2','ovun6','trip2',
         'ovun7','ovun8','trip1','swa','swb','n.u.','val6','lp1',
         'val9','val10','n.u.','pen2','pen3','pen4','n.u.','tor2',
         'tor3','tor4','n.u.','cot2','vdw1','cutoff','coa4','ovun4',
         'ovun3','val8','acut','hbtol','n.u.','Eo','coa3']
         
line_spec = []
line_spec.append(['rosi','val','mass','rvdw','Devdw','gamma','ropi','vale'])
line_spec.append(['alfa','gammaw','valang','ovun5','n.u.','chi','mu','atomic'])
line_spec.append(['ropp','lp2','n.u.','boc4','boc3','boc5','n.u.','n.u.'])
line_spec.append(['ovun2','val3','n.u.','valboc','val5','n.u.','n.u.','n.u.'])

line_bond = []
line_bond.append(['Desi','Depi','Depp','be1','bo5','corr13','bo6','ovun1'])
line_bond.append(['be2','bo3','bo4','n.u.','bo1','bo2','ovcorr','n.u.'])

line_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
line_ang  = ['theta0','val1','val2','coa1','val7','pen1','val4']
line_tor  = ['V1','V2','V3','tor1','cot1','n.u.','n.u.']
line_hb   = ['rohb','Dehb','hb1','hb2']


def read_lib(p={},zpe=False,libfile='irnn.lib',
             p_name=p_name,line_spec=line_spec,
             line_bond=line_bond,line_offd=line_offd,
             line_ang=line_ang,line_tor=line_tor,line_hb=line_hb):
    print('-  initial variable read from: %s.' %libfile)
    if isfile(libfile):
       flib = open(libfile,'r')
       lines= flib.readlines()
       flib.close()
       npar = int(lines[1].split()[0])
       
       if npar>len(p_name):
          print('error: npar >39')
          exit()

       zpe_= {}
       if zpe:
          lin = lines[0].split()
          for i in range(int(len(lin)/2-1)):
              k = lin[i*2+1]
              zpe_[k] = float(lin[i*2])

       for i in range(npar):
           pn = p_name[i]
           p[pn] = float(lines[2+i].split()[0]) 

       # ---------   parameters for species   ---------
       nofc   = 1                 #  number of command line
       nsl    = len(line_spec)
       nsc    = nsl
       npar   = npar + 1
       nspec  = int(lines[nofc+npar].split()[0])
       spec   = []   
       for i in range(nspec):
           spec.append(lines[nofc+npar+nsc+i*nsl].split()[0]) # read in species name in first line
           for il,line in enumerate(line_spec):
               ls = 1 if il == 0 else 0
               for ip,pn in enumerate(line):
                   p[pn+'_'+spec[i]] = np.float(lines[nofc+npar+nsc+i*nsl+il].split()[ls+ip])


       # ---------  parameters for bonds   ---------
       bonds = []
       nbl = len(line_bond)
       nbc = nbl
       nbond = int(lines[nofc+npar+nsc+nspec*nsl].split()[0])
       for i in range(nbond):
           b1= int(lines[nofc+npar+nsc+nspec*nsl+nbc+i*nbl].split()[0])
           b2= int(lines[nofc+npar+nsc+nspec*nsl+nbc+i*nbl].split()[1])
           bond = spec[b1-1] + '-' +spec[b2-1]
           bonds.append(bond)
           for il,line in enumerate(line_bond):
               ls = 2 if il == 0 else 0
               for ip,pn in enumerate(line):
                   p[pn+'_'+bond] = np.float(lines[nofc+npar+nsc+nspec*nsl+nbc+i*nbl+il].split()[ls+ip])


       # ---------   parameters for off-diagonal bonds   ---------
       offd  = []
       nol   = 1
       noc   = nol
       noffd = int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl].split()[0])
       
       for i in range(noffd):
           b1=int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+i].split()[0])
           b2=int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+i].split()[1])
           bond = spec[b1-1] + '-' +spec[b2-1]
           offd.append(bond)
           for ip,pn in enumerate(line_offd):
               p[pn+'_'+bond] = np.float(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+i].split()[2+ip])
           

       # ---------   parameters for angles   ---------
       angs = []
       nal  = 1
       nac  = nal
       nang = int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd].split()[0])
       for i in range(nang):
           l = lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+i].split()
           b1,b2,b3  = int(l[0]),int(l[1]),int(l[2])
           ang = spec[b1-1] + '-' +spec[b2-1] + '-' +spec[b3-1]
           angr= spec[b3-1] + '-' +spec[b2-1] + '-' +spec[b1-1]
           if (not ang in angs) and (not angr in angs):
              angs.append(ang)
              for ip,pn in enumerate(line_ang):
                  p[pn+'_'+ang] = np.float(l[3+ip])


       # ---------   parameters for torsions   --------- 
       ntl  = 1 
       ntc  = ntl
       ntor = int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+nang].split()[0])
       tors = []
       for i in range(ntor):
           l = lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+nang+ntc+i].split()
           b1,b2,b3,b4 = int(l[0]),int(l[1]),int(l[2]),int(l[3])
           e2,e3 = spec[b2-1],spec[b3-1]
           e1 = 'X' if b1==0  else spec[b1-1]
           e4 = 'X' if b4==0  else spec[b4-1]
            
           tor  = e1+'-'+e2 +'-'+e3 +'-'+e4
           tor1 = e4+'-'+e2 +'-'+e3 +'-'+e1
           torr = e4+'-'+e3 +'-'+e2 +'-'+e1
           torr1= e1+'-'+e3 +'-'+e2 +'-'+e4
           if (not tor in tors) and (not torr in tors) and\
              (not tor1 in tors) and (not torr1 in tors):
              tors.append(tor)
              for ip,pn in enumerate(line_tor):
                  p[pn+'_'+tor] = np.float(l[4+ip])

       # ---------   parameters for HBs   ---------
       hbs = []
       nhl = 1
       nhc = nhl
       nhb = int(lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+nang+ntc+ntor].split()[0])
       for i in range(nhb):
           l = lines[nofc+npar+nsc+nspec*nsl+nbc+nbond*nbl+noc+noffd+nac+nang+ntc+ntor+nhc+i].split()
           hb1,hb2,hb3  = int(l[0]),int(l[1]),int(l[2])
           hb = spec[hb1-1] + '-' +spec[hb2-1] + '-' +spec[hb3-1]
           hbs.append(hb)
           for ip,pn in enumerate(line_hb):
               p[pn+'_'+hb] = np.float(l[3+ip])
    else:
       print('* Warning: lib file is not found, initial parameters generated randomly!')
       p = None
    return p,zpe_,spec,bonds,offd,angs,tors,hbs


def taper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r>rmax,1.0,0.0) # r > rmax then 1 else 0
    if r>rmax:
       return 1.0
    elif r<rmin:
       return 0.0
    else:
       rterm = 1.0/(rmin-rmax)**3.0
       trm   = rmin + 2.0*r - 3.0*rmax
       rd    = rmin - r 
       return rterm*rd*rd*trm
    
    
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


class RFFBD(object):
  def __init__(self,atoms=None,
               libfile='ffield',
               rcut=None,rcuta=None,
               vdwcut=10.0,
               nn=False,
               hbshort=6.75,hblong=7.5,
               label="IRFF", **kwargs):
      # Calculator.__init__(self,label=label, **kwargs)
      self.atoms        = atoms
      self.cell         = atoms.get_cell()
      self.rcell        = np.linalg.inv(self.cell)
      # print('-  getting chemical symbols ...')
      self.atom_name    = self.atoms.get_chemical_symbols()
      self.natom        = len(self.atom_name)
      self.spec         = []
 
      # print('-  getting species ...')
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

      self.botol     = 0.01*self.p['cutoff']
      self.atol      = self.p['acut']
      self.hbtol     = self.p['hbtol']
      self.hbshort   = hbshort
      self.hblong    = hblong
      # print('-  setting rcut ...')
      self.set_rcut(rcut,rcuta)
      self.vdwcut    = vdwcut
      self.get_rcbo()
      print('-  setting parameters ...')
      self.set_p(m,self.bo_layer)


  def get_ebond(self):
      self.neighbor,self.r = self.get_neighbor(self.atoms.positions)
      self.ebond = self.get_bondorder()


  # def get_neighbor(self,positions):
  #     neighbor = [[] for i in range(self.natom)]
  #     self.r = [[] for i in range(self.natom)]
  #     for i in range(self.natom-1):
  #         print('-  building neighbors for atom {0} ... '.format(i),end='\r')
  #         for j in range(i+1,self.natom):
  #             bd = self.atom_name[i] + '-' + self.atom_name[j]
  #             vr =positions[j]-positions[i]
  #             vrf   = np.dot(vr,self.rcell)
  #             vrf   = np.where(vrf-0.5>0,vrf-1.0,vrf)
  #             vrf   = np.where(vrf+0.5<0,vrf+1.0,vrf)  
  #             vr    = np.dot(vrf,self.cell)
  #             r     = np.sqrt(np.sum(vr*vr))
                
  #             # print('-  neighbor: ',i,j,r)
  #             if r<self.rcut[bd]:
  #                neighbor[i].append(j)
  #                neighbor[j].append(i)
  #                self.r[i].append(r)
  #                self.r[j].append(r)
  #     # print(neighbor)
  #     return neighbor


  def get_neighbor(self,positions):
      R,neighbor = [],[]
      for i in range(self.natom):
          print('-  building neighbors for atom {0} ... '.format(i),end='\r')
          vr = positions - positions[i]
          vrf   = np.dot(vr,self.rcell)
          vrf   = np.where(vrf-0.5>0,vrf-1.0,vrf)
          vrf   = np.where(vrf+0.5<0,vrf+1.0,vrf)  
          vr    = np.dot(vrf,self.cell)

          r     = np.sqrt(np.sum(vr*vr,axis=1))
          id_   = np.where(np.logical_and(r<self.max_r,r>0.0001))

          nei_,r_ = [],[]
          for n,j in enumerate(id_[0]):
              bd = self.atom_name[i] + '-' + self.atom_name[j]
              if r[j]<self.rcut[bd]:
                 nei_.append(j)
                 r_.append(r[j])

          neighbor.append(nei_)
          R.append(r_)
      return neighbor,R


  def set_rcut(self,rcut,rcuta): 
      if rcut==None: ## bond order term cutoff
         # in princeple, the cutoff should 
         # cut the first nearest neithbors
         self.rcut = {'C-C':2.8,'C-H':2.2,'C-N':2.8,'C-O':2.8,
                      'N-N':2.8,'N-O':2.7,'N-H':2.2,
                      'O-O':2.7,'O-H':2.2,
                      'H-H':2.2,
                      'others':2.8}
      else:
         self.rcut = rcut

      if rcuta==None: ## angle term cutoff
         # in princeple, the cutoff should 
         # cut the first nearest neithbors
         self.rcuta = {'C-C':1.95,'C-H':1.75,'C-N':1.95,'C-O':1.95,
                      'N-N':1.95,'N-O':1.95,'N-H':1.75,
                      'O-O':1.95,'O-H':1.75,
                      'H-H':1.35,
                      'others':1.95}
      else:
         self.rcuta = rcuta
    
      self.max_r = 0.0
      for bd in self.bonds:
          b = bd.split('-')
          bdr = b[1]+'-'+b[0]
          if bd in self.rcut:
             self.rcut[bdr] = self.rcut[bd]
          elif bdr in self.rcut:
             self.rcut[bd]  = self.rcut[bdr]
          else:
             self.rcut[bd]  = self.rcut['others']

          if bd in self.rcuta:
             self.rcuta[bdr] = self.rcuta[bd]
          elif bdr in self.rcuta:
             self.rcuta[bd]  = self.rcuta[bdr]
          else:
             self.rcuta[bd]  = self.rcuta['others']


  def get_rcbo(self):
      ''' get cut-offs for individual bond '''
      self.rc_bo = {}
      for bd in self.bonds:
          # print('-  setting rcbo for bond {0} ...'.format(bd),end='\n')
          b= bd.split('-')
          ofd=bd if b[0]!=b[1] else b[0]
          bdr = b[1]+'-'+b[0]

          log_ = np.log((self.botol/(1.0+self.botol)))
          rr = log_/self.p['bo1_'+bd] 
          self.rc_bo[bd]=self.p['rosi_'+ofd]*np.power(log_/self.p['bo1_'+bd],1.0/self.p['bo2_'+bd])
          if self.rcut[bd]> self.rc_bo[bd]:
             self.rcut[bd]  = self.rc_bo[bd]
             self.rcut[bdr] = self.rc_bo[bd]
             
          if self.max_r<self.rcut[bd]:
             self.max_r = self.rcut[bd]

                
  def get_bondorder_uc(self):
      bopsi,boppi,boppp = [],[],[]
      Deltap = []
      
      for i in range(self.natom):
          # print('-  computing BOp for atom {0} ... '.format(i),end='\r')
          bosi,bopi,bopp = [],[],[]
          D = 0.0
          for n,j in enumerate(self.neighbor[i]):
              r = self.r[i][n]
              bd = self.atom_name[i]+'-'+self.atom_name[j]
              if bd not in self.bonds:
                 bd = self.atom_name[j]+'-'+self.atom_name[i]

              bodiv1 = r/self.p['rosi_'+bd]
              bopow1 = np.power(bodiv1,self.p['bo2_'+bd])
              eterm1 = (1.0+self.botol)*np.exp(self.p['bo1_'+bd]*bopow1)
              bop_si = taper(eterm1,rmin=self.botol,rmax=2.0*self.botol)*(eterm1-self.botol)

              if self.p['ropi_'+bd]>0.0:
                 bodiv2 = r/self.p['ropi_'+bd]
                 bopow2 = np.power(bodiv2,self.p['bo4_'+bd])
                 eterm2 = np.exp(self.p['bo3_'+bd]*bopow2)
                 bop_pi = taper(eterm2,rmin=self.botol,rmax=2.0*self.botol)*eterm2
              else:
                 bop_pi = 0.0

              if self.p['ropp_'+bd]>0.0:
                 bodiv3 = r/self.p['ropp_'+bd]
                 bopow3 = np.power(bodiv3,self.p['bo6_'+bd])
                 eterm3 = np.exp(self.p['bo5_'+bd]*bopow3)
                 bop_pp = taper(eterm3,rmin=self.botol,rmax=2.0*self.botol)*eterm3
              else:
                 bop_pp = 0.0
              bop = bop_si+bop_pi+bop_pp
              D += bop
              bosi.append(bop_si)
              bopi.append(bop_pi)
              bopp.append(bop_pp)

          Deltap.append(D)
          bopsi.append(bosi)
          boppi.append(bopi)
          boppp.append(bopp)
      return bopsi,boppi,boppp,Deltap


  def f1(self,i,j,Deltap):
      atomi = self.atom_name[i]
      atomj = self.atom_name[j]
      Div = Deltap[i] - self.p['val_'+atomi] # replace val in f1 with valp, 
      Djv = Deltap[j] - self.p['val_'+atomj] # different from published ReaxFF model
      f_2 = self.f2(Div,Djv)
      f_3 = self.f3(Div,Djv)

      f_1 = 0.5*(np.divide(self.p['val_'+atomi]+f_2, self.p['val_'+atomi]+f_2+f_3)  + 
                 np.divide(self.p['val_'+atomj]+f_2, self.p['val_'+atomj]+f_2+f_3))
      return f_1 
    

  def f2(self,Div,Djv):
      dexpf2i  = np.exp(-self.p['boc1']*Div)
      dexpf2j  = np.exp(-self.p['boc1']*Djv)
      f_2     = dexpf2i  + dexpf2j
      return f_2      


  def f3(self,Div,Djv):
      dexpf3i   = np.exp(-self.p['boc2']*Div)
      dexpf3j   = np.exp(-self.p['boc2']*Djv)
      delta_exp = dexpf3i + dexpf3j
      f3log     = np.log(0.5*delta_exp )
      f_3       = (-1.0/self.p['boc2'])*f3log
      return f_3


  def f45(self,i,j,Deltap,bop):
      atomi = self.atom_name[i]
      atomj = self.atom_name[j]
      boc3 = np.sqrt(self.p['boc3_'+atomi]*self.p['boc3_'+atomj])
      boc4 = np.sqrt(self.p['boc4_'+atomi]*self.p['boc4_'+atomj])
      boc5 = np.sqrt(self.p['boc5_'+atomi]*self.p['boc5_'+atomj])
        
      Dboci = Deltap[i] - self.p['valboc_'+atomi] # + self.p['val_'+atomi]
      Dbocj = Deltap[j] - self.p['valboc_'+atomj]
      
      df4 = boc4*np.square(bop)-Dboci
      f4r = np.exp(-boc3*(df4)+boc5)

      df5 = boc4*np.square(bop)-Dbocj
      f5r = np.exp(-boc3*(df5)+boc5)

      f_4 = 1.0/(1.0+f4r)
      f_5 = 1.0/(1.0+f5r)
      return f_4,f_5


  def get_bondorder(self):
      bopsi,boppi,boppp,Deltap = self.get_bondorder_uc()
      Ebond = []
      for i in range(self.natom):
          eb_ = []
          print('-  computing bond energy for atom {0} ... '.format(i),end='\r')
          for n,j in enumerate(self.neighbor[i]):
              bd = self.atom_name[i]+'-'+self.atom_name[j]
              if bd not in self.bonds:
                 bd = self.atom_name[j]+'-'+self.atom_name[i]

              bop     = bopsi[i][n]+boppi[i][n]+boppp[i][n]
              f_1     = self.f1(i,j,Deltap)
              f_4,f_5 = self.f45(i,j,Deltap,bop)
              F       = f_1*f_1*f_4*f_5 
              bo      = bop*f_1*f_4*f_5   #-0.001        # consistent with GULP
       
              bopi    = boppi[i][n]*F
              bopp    = boppp[i][n]*F
              bosi    = bo - bopi - bopp
              bosi    = bosi if bosi>0.0 else 0.0
      
              powb  = np.power(bosi,self.p['be2_'+bd])
              expb  = np.exp(np.multiply(self.p['be1_'+bd],1.0 - powb))
            
              sieng = self.p['Desi_'+bd]*bosi*expb 
              pieng = np.multiply(self.p['Depi_'+bd],bopi)
              ppeng = np.multiply(self.p['Depp_'+bd],bopp)
              ebond = - sieng - pieng - ppeng
              eb_.append(ebond)
              
          Ebond.append(eb_)  
      print(' ',end='\n')
      return Ebond


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

      # self.rcbo = np.zeros([self.natom,self.natom],dtype=np.float32)

      for key in p_offd:
          for sp in self.spec:
              try:
                 self.p[key+'_'+sp+'-'+sp]  = self.p[key+'_'+sp]  
              except KeyError:
                 print('-  warning: key not in dict') 

      p_spec = ['valang','valboc','val','vale',
                'lp2','ovun5',                 # 'val3','val5','boc3','boc4','boc5'
                'ovun2','atomic',
                'mass','chi','mu']             # 'gamma','gammaw','Devdw','rvdw','alfa'


      for key in self.p:
          kpre = key.split('_')[0]
          unit_ = self.unit if kpre in self.punit else 1.0
          self.p[key] = self.p[key]*unit_


  def init_bonds(self):
      self.bonds,self.offd = [],[]
      for key in self.p:
          k = key.split('_')
          if k[0]=='bo1':
             self.bonds.append(k[1])
          elif k[0]=='rosi':
             kk = k[1].split('-')
             if len(kk)==2:
                self.offd.append(k[1])


  def close(self):
      self.P  = None
      self.m  = None


def LammpsHistory(traj='ps0.1.lammpstrj',frame=0,atomType =['C','H','O','N']):
    fl = open(traj,'r')
    lines = fl.readlines()
    nl    = len(lines)
    fl.close()
    natom = int(lines[3])
    print('-  number of frames',int(nl/natom+9))

    n         = 0
    block     = natom+9

    atomName  = [' ' for i in range(natom)]
    positions = np.zeros([natom,3])
    cell      = np.zeros([3,3])
    line      = lines[block*frame + 5].split()
    cell[0][0]= float(line[1])-float(line[0])
    line      = lines[block*frame + 6].split()
    cell[1][1]= float(line[1])-float(line[0])
    line      = lines[block*frame + 7].split()
    cell[2][2]= float(line[1])-float(line[0])

    for i in range(natom):
        n = block*frame + i + 9
        line = lines[n].split()
        id_  = int(line[0])-1
        atomName[id_]=atomType[int(line[1])-1]
        positions[id_][0] = float(line[2])
        positions[id_][1] = float(line[3])
        positions[id_][2] = float(line[4])
        
    atoms  = Atoms(atomName,positions,cell=cell,pbc=[True,True,True])
    # view(atoms)
    lines= None
    return atoms


def p3d(traj='ps0.1.lammpstrj',frame=0,
      atomType =['C','H','O','N'],
          color={'C':'g','H':'khaki','O':'r','N':'b'},
          size = {'C':80,'H':40,'O':76,'N':76}):
    atoms = LammpsHistory(traj=traj,frame=frame,atomType=atomType)
    # atoms = read('md.traj')
    ir = RFFBD(atoms=atoms,libfile='ffield',rcut=None,nn=False)
    ir.get_ebond()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
     
    # set figure information
    ax.set_title("Bond Energy")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    e = []
    for e_ in ir.ebond:
        for em in e_:
            e.append(em)
    # print(e)
    mine = min(e)
    cmap = cm.ScalarMappable(col.Normalize(mine,0.0), cm.rainbow)

    for i in range(ir.natom):
        print('-  ploting bonds for atom {0} ... '.format(i),end='\r')
        for n,j in enumerate(ir.neighbor[i]):
            bd = ir.atom_name[i] + '-' + ir.atom_name[j]
            r = np.sqrt(np.sum(np.square(atoms.positions[j]-atoms.positions[i])))
            if j>i:
               if r<ir.rcut[bd]: 
                  x = [atoms.positions[i][0],atoms.positions[j][0]]
                  y = [atoms.positions[i][1],atoms.positions[j][1]]
                  z = [atoms.positions[i][2],atoms.positions[j][2]]
                  ax.plot(x,y,z,c=cmap.to_rgba(ir.ebond[i][n]),linewidth=1)

    print(' ',end='\n')
    # for a in atoms:   
    #     ax.scatter(a.x, a.y, a.z, c=color[a.symbol],s=size[a.symbol],label=a.symbol)
        
    ca = np.linspace(mine,0,100)
    cmap.set_array(ca)
    plt.colorbar(cmap,label='Color Map(Unit: eV)')
    # plt.show()
    plt.savefig('bondEnergy3d.eps')
    plt.close()


def p(traj='ps0.1.lammpstrj',frame=0,
      atomType =['C','H','O','N'],
      color={'C':'g','H':'khaki','O':'r','N':'b'},
      size = {'C':80,'H':40,'O':76,'N':76},
      surface='xy',
      sz=40,sy=40,sx=40,
      thick=5):
    atoms = LammpsHistory(traj=traj,frame=frame,atomType=atomType)
    # atoms = read('md.traj')
    
    print('-  boxes of the model:')
    cell = atoms.get_cell()
    for i in range(3):
        print(cell[i])

    ir = RFFBD(atoms=atoms,libfile='ffield',rcut=None,nn=False)
    ir.get_ebond()

    plt.figure()
    plt.ylabel('Y')
    plt.xlabel('X')

    e = []
    for e_ in ir.ebond:
        for em in e_:
            e.append(em)
    # print(e)
    mine = min(e)
    cmap = cm.ScalarMappable(col.Normalize(mine,0.0), cm.rainbow)
    X,Y,C = [],[],[]
    for i in range(ir.natom):
        print('-  ploting bonds for atom {0} ... '.format(i),end='\r')
        for n,j in enumerate(ir.neighbor[i]):
            bd = ir.atom_name[i] + '-' + ir.atom_name[j]
            r = np.sqrt(np.sum(np.square(atoms.positions[j]-atoms.positions[i])))
            if r<ir.rcut[bd] and ir.ebond[i][n]<-0.2: 
               x = 0.5*(atoms.positions[i][0]+atoms.positions[j][0])
               y = 0.5*(atoms.positions[i][1]+atoms.positions[j][1])
               z = 0.5*(atoms.positions[i][2]+atoms.positions[j][2])
               if j>i:
                  if surface=='xy':
                     if z<sz+thick and z>sz-thick:
                        X.append(x)
                        Y.append(y)
                        C.append(cmap.to_rgba(ir.ebond[i][n]))
                  elif surface=='yz':
                     if z<sx+thick and z>sx-thick:
                        X.append(y)
                        Y.append(z)
                        C.append(cmap.to_rgba(ir.ebond[i][n]))
                  elif surface=='xz':
                     if z<sy+thick and z>sy-thick:
                        X.append(x)
                        Y.append(z)
                        C.append(cmap.to_rgba(ir.ebond[i][n]))
                    
    print(' ',end='\n')
    plt.scatter(X,Y,color=C,s=1)
    ca = np.linspace(mine,0,100)
    cmap.set_array(ca)

    # X,Y = np.meshgrid(X,Y)
    # plt.contourf(X,Y,E,8,alpha=0.75,cmap=plt.cm.rainbow)
    plt.colorbar(cmap,label='Color Map(Unit: eV)')
    # plt.show()
    plt.savefig('bondEnergySurface.eps')
    plt.close()



if __name__ == '__main__':
   ''' use commond like ./rffbd.py p --traj=p.lampstraj --frame=0 --surface=xz to run it
       p: plot the bond energy
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [p3d,p])
   argh.dispatch(parser)

