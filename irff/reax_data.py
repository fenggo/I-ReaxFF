import numpy as np
from os import system, getcwd, chdir,listdir
#from os.path import isfile
#from pyrsistent import v
#from .dft.cpmd import get_lattice
#from .dft.nwchem import get_nw_gradient,out_xyz
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter,Trajectory
from .md.gulp import write_gulp_in,get_reaxff_q
#from .reaxfflib import read_ffield,write_lib
from .qeq import qeq
import random
# import pickle
# np.set_printoptions(threshold=np.inf) 


def rtaper(r,rmin=6.75,rmax=7.5):
    r1    = np.where(r<rmin,1.0,0.0)  
    r2r   = np.where(np.logical_and(r>rmin,r<rmax),1.0,0.0)  

    rterm = 1.0/(rmax - rmin)**3
    rd    = rmax - r
    trm1  = (rmax + 2.0*r - 3.0*rmin)
    f     = rterm*rd*rd*trm1*r2r
    return  f+r1

def taper(r,rmin=0.001,rmax=0.002):
    r1  = np.where(r>rmax,1.0,0.0) # r > rmax then 1 else 0
    rr  = np.where((r>rmin) & (r<rmax),1.0,0.0) # r > rmax then 1 else 0

    rterm = 1.0/(rmax - rmin)**3
    rd = rmax - r
    trm1 = (rmax + 2.0*r - 3.0*rmin)
    f = rterm*rd*rd*trm1*rr
    return  f+r1

def get_data(structure='data',direc=None,
             vdwcut=10.0,rcut=None,rcuta=None,
             hbshort=6.5,hblong=7.5,
             atoms=None,
             batch=1000,sample='random',
             minib=100,variable_batch=False,
             p=None,spec=None,bonds=None,angs=None,
             tors=None,hbs=None,
             #screen=False,
             nindex=[]):
    t = direc.split('.')
    if t[-1]=='traj':
       dft='ase'
    data = reax_data(structure=structure,direc=direc,
                     sample=sample,
                     vdwcut=vdwcut,
                     rcut=rcut,rcuta=rcuta,
                     hbshort=hbshort,hblong=hblong,
                     atoms=atoms,
                     batch=batch,minib=minib,
                     variable_batch=variable_batch,
                     p=p,spec=spec,bonds=bonds,angs=angs,
                     tors=tors,hbs=hbs,
                     #screen=screen,
                     nindex=nindex)
    return data

class Dataset(object):
  '''Data set to feed the ReaxFF-nn computaional graph'''
  def __init__(self,dft_energy,forces,rbd,rv,qij,theta,s_ijk,s_jkl,w,rhb,frhb,hbthe):
      self.dft_energy = dft_energy
      self.forces     = forces
      self.rbd        = rbd.transpose()
      self.rv         = rv
      self.qij        = qij
      self.theta      = theta
      self.s_ijk      = s_ijk
      self.s_jkl      = s_jkl
      self.w          = w
      self.rhb        = rhb
      self.frhb       = frhb
      self.hbthe      = hbthe


class reax_data(object):
  """ Collecting datas for mathine learning for bond order potential ReaxFF or ReaxFF-nn
      Atribute:
      --------
         bond:  int bond table list such as [[1,2],[1,3],...]
         rbd:   all bonds in the bond listed in "bond" list
  """
  def __init__(self,structure='cl20mol',botol=0.001,direc=None,
               vdwcut=10.0,rcut=None,rcuta=None,
               hbshort=6.75,hblong=7.5,
               atoms=None,
               batch=1000,minib=100,sample='uniform',
               p=None,spec=None,bonds=None,angs=None,
               tors=None,hbs=None,
               traj=False,          
               variable_batch=False,
               nindex=[]):
      self.structure = structure
      self.energy_nw,self.energy_bop = [],[]
      self.table     = []
      self.atom_name = []
      self.min_e     = 0.0
      self.max_e     = 0.0
      self.botol     = botol
      self.vdwcut    = vdwcut
      self.hbshort   = hbshort
      self.hblong    = hblong
      self.p         = p
      self.spec      = spec
      self.bonds     = bonds
      self.angs      = angs
      self.tors      = tors
      self.hbs       = hbs
      self.r_cut     = rcut
      self.rcuta     = rcuta
      self.traj      = traj
      # self.screen  = screen             #   screen zero three-body and four-body interaction
      self.status    = True

      print('-  Getting informations from directory {:s} ...\n'.format(direc))
      cdir = getcwd()

      if atoms==None:
         images  = self.get_ase_energy(direc)
         trajonly= False
      else:
         images      = [atoms]
         self.nframe = 1
         trajonly    = True
      
      if variable_batch:
         if self.nframe>batch:
            n = self.nframe//batch
            self.batch = int(self.nframe/n)
         else:
            self.batch = self.nframe 
      else:
         self.batch = batch

      self.minib = minib    
      nni        = len(nindex)   # the number of frame alread be collected

      random.seed()
      pool = np.arange(self.nframe)
      pool = list(set(pool).difference(set(nindex)))

      if self.nframe>=self.batch+nni:
         if sample=='uniform':
            pool        = np.array(pool)
            ind_        = np.linspace(0,len(pool)-1,num=self.batch,dtype=np.int32)
            self.indexs = pool[ind_]
         else: # random
            indexs    = random.sample(pool,self.batch)

            indexs    = np.array(indexs)
            indices   = indexs.argsort()
            self.indexs = indexs[indices]

         self.get_ase_data(images,trajonly)
      else:
         print('-  data set of {:s} is not sufficient, repeat frames ...'.format(self.structure))
         self.nframe = len(pool)
         nb = int(self.batch/self.nframe+1)
         for i in range(nb):
               # print('-  in direc: %s' %self.structure)
               if self.nframe*(i+1)<self.batch:
                  if i==0:
                     self.indexs = pool
                  else:
                     self.indexs = np.append(self.indexs,pool)
               else:
                  left = self.batch - self.nframe*i   
                  if left>0:
                     self.indexs = np.append(self.indexs,pool[:left])

         self.get_ase_data(images,trajonly)
         self.nframe=self.batch
  
      self.set_parameters()
      self.min_e     = min(self.energy_nw)
      self.max_e     = max(self.energy_nw)

      # self.natom_images = self.natom                                    # all atoms include images
      r_  = max(self.r_cut.values())* 2.0
      R_  = np.max(np.sqrt(np.sum(self.cell*self.cell,axis=2)))    
      # self.image_mask = [0,0,0]
      if R_<r_:
         #self.image_mask[i] = 1
         raise RuntimeError('-  Error: cell length must lager than 2.0*r_cut, using supercell!')
             
      self.R,self.vr = self.compute_bond(self.x)
      image_rs = self.compute_image(self.vr)                            # vdw interaction images

      self.get_table()
      self.get_bonds(self.R)

      if self.structure.find('nomb')>=0:
         self.nhb  = 0
         self.nang = 0
         self.ntor = 0
         self.hb_i,self.hb_j,self.hb_k = [],[],[]
         self.abij,self.abjk = [],[]
         self.ang_i,self.ang_j,self.ang_k = [],[],[]
         self.tij,self.tjk,self.tkl = [],[],[]
         self.tor_i,self.tor_j,self.tor_k,self.tor_l = [],[],[],[]
         self.A,self.T,self.H,self.hij={},{},{},{}
         self.na,self.nt={},{}
      else:
         self.compute_angle(self.R,self.vr)
         self.compute_torsion(self.R,self.vr)
         self.compute_hbond(image_rs)
         
      self.compute_vdw(image_rs)

      # self.get_gulp_energy()
      self.get_charge()
      self.get_ecoul(image_rs)
      self.get_eself()
      
      if variable_batch:
         # print('-  grouping rvdw ...')
         self.group_rvdw()
         self.group_rhb()

      # for i,e in enumerate(self.energy_nw):  # new version zpe = old zpe + max_e
      #     self.energy_nw[i] = e - self.max_e  
      print('-  end of gathering datas from directory {:s} ...\n'.format(direc))

  def compute_bond(self,x):
      hfcell    = 0.5 
      u         = np.linalg.inv(self.cell)
      x         = np.array(x)                       #  project to the fractional coordinate
      xf        = np.matmul(x,u) 

      xj   = np.expand_dims(xf,axis=1)
      xi   = np.expand_dims(xf,axis=2)
      vr   =  xj - xi                          

      lm   = np.where(vr-hfcell>0)
      lp   = np.where(vr+hfcell<0)

      while (lm[0].size!=0 or lm[1].size!=0 or lm[2].size!=0 or
            lp[0].size!=0 or lp[1].size!=0 or lp[2].size!=0):
         vr = np.where(vr-hfcell>0,vr-1.0,vr)
         vr = np.where(vr+hfcell<0,vr+1.0,vr)     # apply pbc
         lm = np.where(vr-hfcell>0)
         lp = np.where(vr+hfcell<0)
      
      vr  = np.matmul(vr,np.expand_dims(self.cell,axis=1)) # convert to ordinary coordinate
      R   = np.sqrt(np.sum(vr*vr,axis=3),dtype=np.float32)
      return R,vr 

  def get_neighbors(self,atoms=None,R=None,rcut=None,rcuta=None):
      table  = [[] for i in range(self.natom)]
      atable = [[] for i in range(self.natom)]
      for i in range(self.natom):
          for j in range(self.natom):
              if j==i:
                 continue
              i_ = i #% self.natom
              j_ = j % self.natom

              pair  = atoms[i_]+'-'+atoms[j_]
              pairr = atoms[j_]+'-'+atoms[i_]
              r = R[i][j]
              if pair in rcut:
                 key = pair
              elif pairr in rcut:
                 key = pairr
              else:
                 raise RuntimeError('-  Error: rcut of bond {:s} not found. '.format(pair))
              # print(r,i,j,rcut[key])
              if r<rcut[key]:
                 table[i].append(j)
                 # table[j_].append(i)
                 
              if r<rcuta[key]:
                 atable[i].append(j)
                 # atable[j_].append(i)
      return table,atable

  def get_table(self):
      for nf in range(len(self.x)):
          table,atable = self.get_neighbors(atoms=self.atom_name,R=self.R[nf],
                                            rcut=self.r_cut,rcuta=self.rcuta)
          if nf==0:
             self.table  = table
             self.atable = atable
             print('-----------------------------------------------------------------\n')
             print('---    number of {:4d} atom in molecule {:18s} ---'.format(self.natom,self.structure))
             print('-----------------------------------------------------------------\n')
          else:
             print('-  compute table of batch {0}/{1} ...\r'.format(nf+1,self.batch),end='\r')
             for na,tab in enumerate(table):
                 for atom in tab:
                     if not atom in self.table[na]:
                        self.table[na].append(atom)
             for na,tab in enumerate(atable):
                 for atom in tab:
                     if not atom in self.atable[na]:
                        self.atable[na].append(atom)
      print('\n')

  def get_bonds(self,R):
      self._bond   = []
      self.max_nei = 0
      bond_        = {}
      self.B       = {}
      self.nbd     = {}
      for i in range(self.natom): 
          if len(self.table[i])>self.max_nei:
             self.max_nei = len(self.table[i])
          for n_j,j in enumerate(self.table[i]):   
              if j!=i:
                 #i_ = i % self.natom
                 j_ = j % self.natom
                 bn = self.atom_name[i]+'-'+self.atom_name[j_]
                 bnr= self.atom_name[j_]+'-'+self.atom_name[i]
                 if bn in self.bonds:
                    bd = bn
                    pair  = (i,j)
                    pairr = (j,i)
                 elif bnr in self.bonds and bnr!=bn:
                    bd = bnr
                    pair  = (j,i)
                    pairr = (i,j)
                 else:
                    raise RuntimeError('-  an error case encountered, {:s} not found in bondlist.'.format(bn))
                 # print('-  pair of atom %d & %d, %s' %(i,j,bn),pair)

                 if bd not in bond_:
                    bond_[bd] = []
                 if (not pair in bond_[bd]) and (not pairr in bond_[bd]):
                    bond_[bd].append(pair)

      for bd in self.bonds:
          st = len(self._bond)
          if bd in bond_:
             self._bond.extend(bond_[bd])
          ed = len(self._bond)
          self.B[bd]   = (st,ed-st)
          self.nbd[bd] = ed - st
          
      self.nbond  = len(self._bond)

      self.blist  = [] # np.zeros((self.natom,self.max_nei))
      for i,tab in enumerate(self.table):                       # construct bond table of every atom
          bl = [[0] for _ in range(self.max_nei)]
          for n,j in enumerate(tab):
              bd_ = (i,j)
              if bd_ not in self._bond:
                 bd_ = (j,i)
              nb = self._bond.index(bd_)                     
              bl[n][0] = nb+1
          self.blist.append(bl)
      self.blist = np.array(self.blist)
      #self.blist.expand_dims(axis=0)
      
      self.bond   = np.array(self._bond,dtype=np.int64)
      self.rbd    = R[:,self.bond[:,0],self.bond[:,1]]
      dilink      = np.expand_dims(self.bond[:,0],axis=1)
      djlink      = np.expand_dims(self.bond[:,1],axis=1)
      self.dilink,self.djlink = {},{}
      for bd in bond_:
          self.dilink[bd] = dilink[self.B[bd][0]:self.B[bd][0]+self.B[bd][1]]
          self.djlink[bd] = djlink[self.B[bd][0]:self.B[bd][0]+self.B[bd][1]]

  def compute_angle(self,R,vr):
      angles,self.A,self.ang_i,self.ang_j,self.ang_k = {},{},[],[],[]
      self.na = {}

      for i in range(self.natom):
          for n_j,j in enumerate(self.atable[i]):         
              if j >= self.natom or j==i or self.atom_name[j]=='H':
                 continue
              for k in self.atable[j]:
                  if k != i and k!=j:
                     ang = (i,j,k)
                     angr= (k,j,i)
                     an  = self.atom_name[i]+'-' + self.atom_name[j]+'-'+self.atom_name[k]
                     if an not in self.angs:
                        an  = self.atom_name[k]+'-' + self.atom_name[j]+'-'+self.atom_name[i]
                        ang = angr
                     if an not in angles:
                        angles[an] = [] 
                     if (not ang in angles[an]) and (not angr in angles[an]):
                        angles[an].append(ang)
                        

      for ang in self.angs:
          if ang in angles:
             st= len(self.ang_i)
             self.ang_i.extend(np.array(angles[ang])[:,0])
             ed= len(self.ang_i)
             self.ang_j.extend(np.array(angles[ang])[:,1])
             self.ang_k.extend(np.array(angles[ang])[:,2])
             self.A[ang] = (st,ed-st)
             self.na[ang] = ed - st
      self.nang = len(self.ang_i)
      print('-  number of angles: {:d} ...\n'.format(self.nang))
      
      self.abij,self.abjk = [],[]
      for i in range(self.nang):
          bij = (self.ang_i[i],self.ang_j[i])
          if bij not in self._bond:
             bij = (self.ang_j[i],self.ang_i[i])
          n = self._bond.index(bij)
          self.abij.append([n])

          bjk = (self.ang_j[i],self.ang_k[i])
          if bjk not in self._bond:
             bjk = (self.ang_k[i],self.ang_j[i])
          n = self._bond.index(bjk)
          self.abjk.append([n])
    
      Rij   = R[:,self.ang_i,self.ang_j]
      Rjk   = R[:,self.ang_j,self.ang_k]
      # Rik = R[:,self.ang_i,self.ang_k]
      vik   = vr[:,self.ang_i,self.ang_j] + vr[:,self.ang_j,self.ang_k]  
      Rik   = np.sqrt(np.sum(vik*vik,axis=2),dtype=np.float32)

      Rij2  = Rij*Rij
      Rjk2  = Rjk*Rjk
      Rik2  = Rik*Rik

      cos_theta = (Rij2+Rjk2-Rik2)/(2.0*Rij*Rjk)
      cos_theta = np.where(cos_theta>1.0,1.0,cos_theta)
      cos_theta = np.where(cos_theta<-1.0,-1.0,cos_theta)

      self.cos_theta = np.transpose(cos_theta,[1,0])
      self.theta     = np.arccos(self.cos_theta)

  def compute_torsion(self,R,vr):
      '''  compute torsion angles  '''
      self.tor_i,self.tor_j,self.tor_k,self.tor_l = [],[],[],[]
      torsion,self.T,self.nt = {},{},{}
      for i in range(self.natom): # atomic energy of i
          for n_j,j in enumerate(self.atable[i]):   
              if j>= self.natom or j==i or self.atom_name[j]=='H':
                 continue
              for k in self.atable[j]:
                  if k >= self.natom or k == i or k==j or self.atom_name[k]=='H':
                     continue
                  for l in self.atable[k]:
                      if l==k or l==j or l==i:
                         continue
                      
                      t1=self.atom_name[i]+'-'+self.atom_name[j]+'-'+self.atom_name[k]+'-'+self.atom_name[l]
                      t2=self.atom_name[l]+'-'+self.atom_name[k]+'-'+self.atom_name[j]+'-'+self.atom_name[i]
                      t3=self.atom_name[i]+'-'+self.atom_name[j]+'-'+self.atom_name[k]+'-X'
                      t4='X-'+self.atom_name[k]+'-'+self.atom_name[j]+'-'+self.atom_name[i]
                      t5='X-'+self.atom_name[j]+'-'+self.atom_name[k]+'-X'
                      t6='X-'+self.atom_name[k]+'-'+self.atom_name[j]+'-X'
                      if t1 in self.tors:
                         t = t1
                      elif t2 in self.tors:
                         t = t2
                      elif t3 in self.tors:
                         t = t3
                      elif t4 in self.tors:
                         t = t4
                      elif t5 in self.tors:
                         t = t5   
                      elif t6 in self.tors:
                         t = t6    
                      else:
                         # raise RuntimeError('-  torsion angle {:s} not in list!'.format(t1))
                         continue
                      
                      if t not in torsion:
                         torsion[t] = []
                      tor = (i,j,k,l)
                      torr= (l,k,j,i)
                      if (not tor in torsion[t]) and (not torr in torsion[t]):
                         torsion[t].append(tor)
      for t in self.tors:
          st = len(self.tor_i)
          if t in torsion:
             self.tor_i.extend(np.array(torsion[t])[:,0])
             self.tor_j.extend(np.array(torsion[t])[:,1])
             self.tor_k.extend(np.array(torsion[t])[:,2])
             self.tor_l.extend(np.array(torsion[t])[:,3])
          ed = len(self.tor_i)
          self.T[t] = (st,ed-st)
          self.nt[t] = ed - st

      self.ntor = len(self.tor_i)
      print('-  number of torsion angles: {:d} \n'.format(self.ntor))

      self.tij,self.tjk,self.tkl = [],[],[]
      for n in range(self.ntor):
          i = self.tor_i[n]
          j = self.tor_j[n]
          k = self.tor_k[n]
          l = self.tor_l[n]
          t = (i,j)
          if t not in self._bond:
             t = (j,i)
          self.tij.append([self._bond.index(t)])

          t = (j,k)
          if t not in self._bond:
             t = (k,j)
          self.tjk.append([self._bond.index(t)])

          t = (k,l)
          if t not in self._bond:
             t = (l,k)
          self.tkl.append([self._bond.index(t)])

      nb = int(self.batch/self.minib)
      yu = self.batch-nb*self.minib
      if yu>0: nb += 1

      for b in range(nb): 
          st = b*self.minib
          ed = (b+1)*self.minib
          if ed > self.batch:
             ed = self.batch
          Rij = R[st:ed,self.tor_i,self.tor_j]
          Rjk = R[st:ed,self.tor_j,self.tor_k]
          Rkl = R[st:ed,self.tor_k,self.tor_l]     

          vrjk = vr[st:ed,self.tor_j,self.tor_k,:]
          vrkl = vr[st:ed,self.tor_k,self.tor_l,:]
          vrjl = vrjk + vrkl                   # consist with GULP
          Rjl  = np.sqrt(np.sum(vrjl*vrjl,axis=2),dtype=np.float32)

          vrij = vr[st:ed,self.tor_i,self.tor_j,:]
          #vrjl = vr[st:ed,self.tor_j,self.tor_l,:]
          vril = vrij + vrjl                   # consist with GULP
          Ril  = np.sqrt(np.sum(vril*vril,axis=2),dtype=np.float32)

          vrik = vrij + vrjk
          Rik  = np.sqrt(np.sum(vrik*vrik,axis=2),dtype=np.float32)

          Rij2 = Rij*Rij
          Rjk2 = Rjk*Rjk
          Rkl2 = Rkl*Rkl
          Rjl2 = Rjl*Rjl
          Ril2 = Ril*Ril
          Rik2 = Rik*Rik

          c_ijk = (Rij2+Rjk2-Rik2)/(2.0*Rij*Rjk)
          c_ijk = np.where(c_ijk>1.0,1.0,c_ijk)
          c_ijk = np.where(c_ijk<-1.0,-1.0,c_ijk)

          ccijk = np.where(c_ijk>0.99999999,0.0,1.000)
          c2ijk = c_ijk*c_ijk
          #thet_ijk = np.arccos(c_ijk)

          c     = 1.0-c2ijk
          # print(np.where(np.isinf(c)))

          s_ijk = np.sqrt(np.where(c<0.0,0.0,c))
          strm  = np.transpose(s_ijk,[1,0])

          if b==0:
             self.s_ijk = strm
          else:
             self.s_ijk = np.concatenate((self.s_ijk,strm),axis=1)

          c_jkl = (Rjk2+Rkl2-Rjl2)/(2.0*Rjk*Rkl)
          c_jkl = np.where(c_jkl>1.0,1.0,c_jkl)
          c_jkl = np.where(c_jkl<-1.0,-1.0,c_jkl)

          ccjkl = np.where(c_jkl>0.99999999,0.0,1.0)
          c2jkl = c_jkl*c_jkl
          #thet_jkl = np.arccos(c_jkl)
          c = 1.0-c2jkl
          s_jkl = np.sqrt(np.where(c<0.0,0.0,c))
          strm  = np.transpose(s_jkl,[1,0])

          if b==0:
             self.s_jkl = strm
          else:
             self.s_jkl = np.concatenate((self.s_jkl,strm),axis=1)

          #c_ijl = (Rij2+Rjl2-Ril2)/(2.0*Rij*Rjl)

          c_kjl = (Rjk2+Rjl2-Rkl2)/(2.0*Rjk*Rjl)
          c_kjl = np.where(c_kjl>1.0,1.0,c_kjl)
          c_kjl = np.where(c_kjl<-1.0,-1.0,c_kjl)

          # cckjl = np.where(c_kjl>0.99999999,0.0,1.0)
          c2kjl = c_kjl*c_kjl
          c     = 1.0-c2kjl
          s_kjl = np.sqrt(np.where(c<0.0,0.0,c))

          fz = Rij2+Rjl2-Ril2-2.0*Rij*Rjl*c_ijk*c_kjl
          fm = Rij*Rjl*s_ijk*s_kjl

          fm = np.where(fm==0.0,1.0,fm)
          fac= np.where(fm==0.0,0.0,1.0)
          cos_w = 0.5*fz*fac/fm
          cos_w = cos_w*ccijk*ccjkl

          cos_w = np.where(cos_w>1.0,1.0,cos_w)   
          cos_w = np.where(cos_w<-1.0,-1.0,cos_w)
          ctm   = np.transpose(cos_w,[1,0])

          if b==0:
             self.cos_w = ctm
          else:
             self.cos_w = np.concatenate((self.cos_w,ctm),axis=1)

          # self.w= np.arccos(self.cos_w)
          wtm = np.arccos(ctm)
          if b==0:
             self.w = wtm
          else:
             self.w = np.concatenate((self.w,wtm),axis=1)

          # self.cos2w = np.cos(2.0*self.w)
          c2wtm = np.cos(2.0*wtm)
          # print('-- mini batch shape',c2wtm.shape[1])
          if b==0:
             self.cos2w = c2wtm
          else:
             self.cos2w = np.concatenate((self.cos2w,c2wtm),axis=1) 

  def compute_image(self,vr):
      vr_   = []
      cell_ = np.expand_dims(np.expand_dims(self.cell,axis=1),axis=1)
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell = cell_[:,:,:,0]*i + cell_[:,:,:,1]*j+cell_[:,:,:,2]*k
                  vr_.append(vr+cell)
      return vr_

  def compute_vdw(self,image_rs):
      vi,vj,vi_p,vj_p,self.vi,self.vj = [],[],[],[],[],[]
      # self.V = {}
      # vdws   = {}
      self.nv  = {}
      rv_      = {}
      for i in range(self.natom-1):
          for j in range(i+1,self.natom):
              vn = self.atom_name[i] + '-' + self.atom_name[j] 
              if vn not in self.bonds:
                 vn = self.atom_name[j] + '-' + self.atom_name[i] 
              vi_p.append(i)
              vj_p.append(j)

      vi_p = np.array(vi_p)
      vj_p = np.array(vj_p)

      for i in range(self.natom):
          for j in range(i,self.natom):
              vi.append(i)
              vj.append(j)

      vi = np.array(vi)
      vj = np.array(vj)

      for i,vr in enumerate(image_rs):
          if i<13:
             vr_ = vr[:,vi,vj,:]
          else:
             vr_ = vr[:,vi_p,vj_p,:]
          r_  = np.sqrt(np.sum(np.square(vr_),axis=2),dtype=np.float32)

          ind = np.where(np.logical_and(np.min(r_,axis=0)<=self.vdwcut, 
                                        np.max(r_,axis=0)>0.000001))
          ind = np.reshape(ind,[-1])
          if i<13:
             self.vi.extend(vi[ind])
             self.vj.extend(vj[ind])
          else:
             self.vi.extend(vi_p[ind])
             self.vj.extend(vj_p[ind])
          rv_ = r_[:,ind]

          if i==0:
             self.rv = rv_
          else:
             self.rv = np.append(self.rv,rv_,axis=1)

      self.rv = np.transpose(self.rv,[1,0])
      self.nvb= len(self.rv)

  def compute_hbond(self,image_rs):
      self.hb_i,self.hb_j,self.hb_k = [],[],[]
      hb_i,hb_j,hb_k = [],[],[]
      # self.H  = {}
      for i in range(self.natom): 
          if self.atom_name[i]!='H':
             for n_j,j in enumerate(self.atable[i]):   
                 if j<self.natom:
                    if self.atom_name[j]=='H' :
                       for k in range(self.natom):
                           if k!=j and self.atom_name[k]!='H':  # from prime cell
                              hb = str(i)+'-'+str(j)+'-'+str(k)
                              hb_i.append(i)
                              hb_j.append(j)
                              hb_k.append(k)

      hb_i = np.array(hb_i)
      hb_j = np.array(hb_j)
      hb_k = np.array(hb_k)

      if len(hb_i)>0 and len(hb_j)>0:
         vij  = self.vr[:,hb_i,hb_j,:]
         Rij2 = np.sum(np.square(vij),axis=2)

         for i,vr in enumerate(image_rs):
             vjk  = vr[:,hb_j,hb_k,:]
             vik  = vij + vjk 

             Rik2 = np.sum(np.square(vik),axis=2)
             Rik  = np.sqrt(Rik2)
            
             ind  = np.where(np.logical_and(np.min(Rik,axis=0)<=self.hblong, 
                                            np.max(Rik,axis=0)>0.000001))
             ind  = np.reshape(ind,[-1])

             self.hb_i.extend(hb_i[ind])
             self.hb_j.extend(hb_j[ind])
             self.hb_k.extend(hb_k[ind])

             Rij2_ = Rij2[:,ind] 
             Rij_  = np.sqrt(Rij2_)

             vjk_ = vjk[:,ind]

             Rjk2_= np.sum(np.square(vjk_),axis=2)
             Rjk_ = np.sqrt(Rjk2_)

             Rik_ = Rik[:,ind] 
             Rik2_= Rik2[:,ind]

             cos_theta = (Rij2_+Rjk2_-Rik2_)/(2.0*Rij_*Rjk_)
             hbthe_    = 0.5-0.5*cos_theta
             frhb_     = rtaper(Rik_,rmin=self.hbshort,rmax=self.hblong)

             if i==0:
                self.rhb   = Rjk_
                self.frhb  = frhb_
                self.hbthe = hbthe_
             else:
                self.rhb   = np.append(self.rhb,Rjk_,axis=1)
                self.frhb  = np.append(self.frhb,frhb_,axis=1)
                self.hbthe = np.append(self.hbthe,hbthe_,axis=1)

         self.rhb   = np.transpose(self.rhb,[1,0])
         self.frhb  = np.transpose(self.frhb,[1,0])
         self.hbthe = np.transpose(self.hbthe,[1,0])
         self.nhb   = len(self.hb_i)
      else:  
         # case for no hydrogen atom 
         self.rhb   = []
         self.frhb  = []
         self.hbthe = []
         self.nhb   = 0

  def group_rvdw(self):
      self.nv   = {}
      rv        = {}
      vi        = {}
      vj        = {}
      qij       = {}
      for vb in self.bonds:
          rv[vb]       = []
          qij[vb]      = []
          vi[vb]       = []
          vj[vb]       = []
          for i,vi_ in enumerate(self.vi):
              vj_ = self.vj[i]
              vn = self.atom_name[vi_]+'-'+self.atom_name[vj_]
              vn_= self.atom_name[vj_]+'-'+self.atom_name[vi_]
              if vn==vb or vn_==vb:
                 rv[vb].append(self.rv[i,:])  # changed here by [:,i]
                 qij[vb].append(self.q[:,vi_]*self.q[:,vj_]*14.39975840)
                 vi[vb].append([vi_])
                 vj[vb].append([vj_])
          self.nv[vb] = len(vi[vb])

      self.V      = {}
      self.rv     = []
      self.vi     = []
      self.vj     = []
      self.qij    = []
      st,ed       = 0,0
      for vb in self.bonds:
          if self.nv[vb]>0:
             st = ed
             ed = st+self.nv[vb]
             self.V[vb] = (st,self.nv[vb])
             self.rv.extend(rv[vb])
             self.vi.extend(vi[vb])
             self.vj.extend(vj[vb])
             self.qij.extend(qij[vb])

  def group_rhb(self):
      self.nh   = {}
      rhb       = {}
      hij       = {}
      hbthe     = {}
      frhb      = {}

      for hb in self.hbs:
          rhb[hb]    = []
          hij[hb]    = []
          hbthe[hb]  = []
          frhb[hb]   = []
          for i,hi in enumerate(self.hb_i):
              # hi = self.hb_i[i]
              hj = self.hb_j[i]
              hk = self.hb_k[i]
              hn = self.atom_name[hi]+'-'+self.atom_name[hj]+'-'+self.atom_name[hk]
              if hn==hb:
                 bd = self.atom_name[hi]+'-'+self.atom_name[hj]
                 bd_ = (hi,hj)
                 if bd not in self.bonds:
                    bd = self.atom_name[hj]+'-'+self.atom_name[hi]
                    bd_ = (hj,hi)
                 ibd = self._bond.index(bd_)
                 hij[hb].append([ibd])
                 rhb[hb].append(self.rhb[i,:])  
                 hbthe[hb].append(self.hbthe[i,:])
                 frhb[hb].append(self.frhb[i,:])
          self.nh[hb] = len(rhb[hb])

      self.H      = {}
      self.hij    = hij
      self.rhb    = []
      self.hbthe  = []
      self.frhb   = []
      st,ed       = 0,0
      for hb in self.hbs:
          if self.nh[hb]>0:
             st = ed
             ed = st+self.nh[hb]
             self.H[hb] = (st,self.nh[hb])
             # self.hij.extend(hij[hb])
             self.rhb.extend(rhb[hb])
             self.hbthe.extend(hbthe[hb])
             self.frhb.extend(frhb[hb])

  def get_gulp_energy(self):
      q,ecoul,eself,evdw = [],[],[],[]
      print('-  get charges from gulp ... \n')
      A = Atoms(symbols=self.atom_name,
                positions=self.x[0],
                cell=self.cell[0],
                pbc=(1, 1, 1))
      for nf in range(self.batch):
          #print('*  get charges of batch {0}/{1} ...\r'.format(nf,self.batch),end='\r')
          A.set_positions(self.x[nf])
          A.set_cell(self.cell[nf])
          write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
          system('gulp<inp-gulp>out')
          q_,ec_,es_,ev_=get_reaxff_q(self.natom,fo='out')
          q.append(q_)
          ecoul.append(ec_)
          eself.append(es_)
          evdw.append(ev_)
      self.q = np.array(q)
      self.ecoul = np.array(ecoul)
      self.eself = np.array(eself)
      self.evdw  = np.array(evdw)

  def get_charge(self):
      q,ecoul,eself,evdw = [],[],[],[]
      print('-  get charges by QEq ... \n')
      A = Atoms(symbols=self.atom_name,
          positions=self.x[0],
          cell=self.cell[0],
          pbc=(1, 1, 1))
      Qe= qeq(p=self.p,atoms=A)

      for nf in range(self.batch):
          # print('*  get charges of batch {0}/{1} ...\r'.format(nf,self.batch),end='\r')
          #  A = Atoms(symbols=self.atom_name,
          #            positions=self.x[nf],
          #            cell=self.cell[nf],
          #            pbc=(1, 1, 1))
          positions = self.x[nf]    # A.get_positions()
          cell      = self.cell[nf] # A.get_cell()
          Qe.calc(cell,positions)
          q.append(Qe.q[:-1])
      self.q = np.array(q)

  def get_ecoul(self,rs):
      gm     = np.sqrt(np.expand_dims(self.P['gamma'],axis=0)*np.expand_dims(self.P['gamma'],axis=1))
      gm     = np.expand_dims(gm,axis=0)
      gm3    = (1.0/gm)**3.0
      qij    = np.expand_dims(self.q,axis=1)*np.expand_dims(self.q,axis=2)
      qij    = qij*14.39975840
      self.qij = qij
      ecoul = 0.0

      for i,vr in enumerate(rs):
          r_  = np.sqrt(np.sum(np.square(vr),axis=3))
          if i<13:
             r = np.triu(r_,k=0)
          else:
             r = np.triu(r_,k=1)

          fv   = np.where(np.logical_and(r<=self.vdwcut,r>=0.0001),1.0,0.0)
          r3   = r**3.0
          tp   = self.tap_vdw(r,vdwcut=self.vdwcut)

          r3third  = (r3+gm3)**(1.0/3.0)
          ecoul_   = np.divide(fv*tp*qij,r3third)
          ecoul   += ecoul_ 

      self.ecoul = np.sum(ecoul,axis=(1,2))

  def get_eself(self):
      chi    = np.expand_dims(self.P['chi'],axis=0)
      mu     = np.expand_dims(self.P['mu'],axis=0)
      eself_ = self.q*(chi+self.q*mu)
      self.eself = np.sum(eself_,axis=1)

  def tap_vdw(self,r,vdwcut=10.0):
      tp = 1.0+np.divide(-35.0,vdwcut**4.0)*(r**4.0)+ \
           np.divide(84.0,vdwcut**5.0)*(r**5.0)+ \
           np.divide(-70.0,vdwcut**6.0)*(r**6.0)+ \
           np.divide(20.0,vdwcut**7.0)*(r**7.0)
      return tp

  def get_ase_energy(self,direc):
      images = Trajectory(direc)
      self.nframe = len(images)
      return images

  def get_ase_data(self,images,trajonly):
      ''' getting data in the ase traj '''
      x         = []
      cell      = []
      forces    = []
      energy_nw = []
      for i,ind_ in enumerate(self.indexs):
          imag = images[ind_]
          if i==0:
             # self.cell    = imag.get_cell()
             self.atom_name = imag.get_chemical_symbols()
             self.natom     = len(self.atom_name)

          if trajonly:
             e = 0.0
          else:
             e = imag.get_potential_energy()
          cell.append(imag.get_cell())
          x.append(imag.positions)
          try:
             force_ = imag.get_forces()
          except:
             force_ = None
             # print('-  ignoring the forces as are not available.')
          forces.append(force_)
          energy_nw.append(e)
      self.energy_nw = np.array(energy_nw)
      self.x         = np.array(x)
      self.cell      = np.array(cell)
      self.forces    = np.array(forces) # ,dtype=np.float32)
      # print(len(self.x),len(self.cell),len(self.forces))

  def set_parameters(self):
      self.P={}
      self.P['gamma'] = np.zeros([self.natom])
      self.P['chi']   = np.zeros([self.natom])
      self.P['mu']    = np.zeros([self.natom])
      for i,a in enumerate(self.atom_name):
          self.P['gamma'][i] = self.p['gamma_'+a]
          self.P['chi'][i] = self.p['chi_'+a]
          self.P['mu'][i] = self.p['mu_'+a]

