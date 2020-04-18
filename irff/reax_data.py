from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from os import system, getcwd, chdir,listdir
from os.path import isfile,exists,isdir
from .cpmd import get_lattice
from ase import Atoms
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter,Trajectory
from .gulp import write_gulp_in,get_reaxff_q
from .reaxfflib import read_lib,write_lib
from .qeq import qeq
import random
import pickle
np.set_printoptions(threshold=np.inf) 


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


def get_data(structure='data',direc=None,out=None,
             vdwcut=10.0,rcut=None,rcuta=None,
             hbshort=6.5,hblong=7.5,
             dft='nwchem',atoms=None,
             batch=1000,sample='random',
             minib=100,
             p=None,spec=None,bonds=None,
             sort=False,pkl=False,nindex=[]):
    if (not isfile(structure + '.pkl')) or (not pkl):
       t = direc.split('.')
       if t[-1]=='traj':
          dft='ase'
       data = reax_data(structure=structure,direc=direc,
               vdwcut=vdwcut,
               rcut=rcut,rcuta=rcuta,
               hbshort=hbshort,hblong=hblong,
               dft=dft,atoms=None,
               batch=batch,minib=minib,sort=sort,
               p=p,spec=spec,bonds=bonds,
               nindex=nindex)
       if not data is None:
          if pkl:
             f = open(structure+'.pkl', 'wb') # open file with write-mode 
             pickle.dump(data,f)
             f.close()
    else:
       f = open(structure+'.pkl', 'rb') 
       data = pickle.load(f)  
       f.close()
    return data


class reax_data(object):
  """ Collecting datas for mathine learning for bond order potential"""
  def __init__(self,structure='cl20mol',botol=0.001,direc=None,
               vdwcut=10.0,rcut=None,rcuta=None,
               hbshort=6.75,hblong=7.5,
               dft='ase',atoms=None,
               batch=1000,minib=100,sample='uniform',
               p=None,spec=None,bonds=None,
               sort=False,
               traj=False,
               nindex=[]):
      self.sort  = sort
      self.structure = structure

      if direc.find('.traj')>=0:
         self.dft= 'ase'
      else:
         self.dft= dft

      self.energy_nw,self.energy_bop = [],[]
      self.table = []
      self.atom_name = []
      self.min_e = 0.0
      self.max_e = 0.0
      self.botol = botol
      self.vdwcut = vdwcut
      self.hbshort= hbshort
      self.hblong= hblong
      self.p     = p
      self.spec  = spec
      self.bonds = bonds
      self.r_cut = rcut
      self.rcuta = rcuta
      self.traj  = traj
      self.cell  = np.array([(10, 0, 0), (0, 10, 0), (0, 0, 10)])
      self.status= True

      print('-  Getting informations from directory %s ...\n' %direc)
      
      cdir = getcwd()
      if direc is None:
         outs = listdir(cdir)
      else:
         if self.dft!='ase':
            chdir(direc)
         
      if self.dft=='nwchem':
         outs = listdir(direc)
         xs = self.get_nw_data(outs)
      elif self.dft=='cpmd':
         self.get_cpmd_frame()
      elif self.dft=='siesta':
         self.get_siesta_energy()
      elif self.dft=='ase':
         if atoms==None:
            images  = self.get_ase_energy(direc)
            trajonly= False
         else:
            images      = [atoms]
            self.nframe = 1
            trajonly    = True
      else:
         print('-  Not supported yet!')

      self.batch = batch  
      self.minib = minib    
      nni =  len(nindex)   

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

         if self.dft=='cpmd':
            xs = self.get_cpmd_data()
         elif self.dft=='siesta':
            xs = self.get_siesta_cart()
            qs = self.get_siesta_charges()
         elif self.dft=='ase':
            self.get_ase_data(images,trajonly)

         if self.dft!='ase':
            xs         = np.array(xs)
            self.x     = xs[self.indexs]
            self.q_dft = qs[self.indexs]

            energy_nw  = np.array(self.energy_nw)
            self.energy_nw = energy_nw[self.indexs]
            self.nframe=self.batch
      else:
         if self.dft=='ase':
            print('-  data set of %s is not sufficient, repeat frames ...' %self.structure)
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
         else:
            print('-  data set of %s is not sufficient, discarded ... ... ...' %structure)
            self.status = False
            if not direc is None:
               chdir(cdir)
            return None

      self.set_parameters()
      self.min_e     = min(self.energy_nw)
      self.max_e     = max(self.energy_nw)
      self.R,self.vr = self.compute_bond(self.x,self.natom,self.nframe)

      self.get_table()
      self.get_bonds()

      self.compute_angle(self.R,self.vr)
      self.compute_torsion(self.R,self.vr)

      image_rs = self.compute_image(self.vr)
      self.compute_hbond(image_rs)
      self.compute_vdw(image_rs)

      if not direc is None:
         if self.dft!='ase':
            chdir(cdir)

      # self.get_gulp_energy()
      self.get_charge()
      self.get_ecoul(image_rs)
      self.get_eself()

      # for i,e in enumerate(self.energy_nw):  # new version zpe = old zpe + max_e
      #     self.energy_nw[i] = e - self.max_e  
      print('-  end of gathering datas from directory %s ...\n' %direc)


  def set_parameters(self):
      self.P={}
      self.P['gamma'] = np.zeros([self.natom])
      self.P['chi']   = np.zeros([self.natom])
      self.P['mu']    = np.zeros([self.natom])
      for i,a in enumerate(self.atom_name):
          self.P['gamma'][i] = self.p['gamma_'+a]
          self.P['chi'][i] = self.p['chi_'+a]
          self.P['mu'][i] = self.p['mu_'+a]


  def sort_data(self):
      ''' for cpmd data '''
      system('mv ENERGIES ENERGIES.orig')
      system('mv TRAJECTORY TRAJECTORY.orig')

      ft = open('TRAJECTORY.orig','r')
      fto = open('TRAJECTORY','w')
      first = None
      for il,line in enumerate(ft.readlines()):
          if line.find('<<<<<<  NEW DATA  >>>>>>') >0:
             fto.close()
             fto = open('TRAJECTORY','w')
             first = None
          else:
             if first is None:
                first = line.split()[0]
             print(line[:-1],file=fto)
      fto.close()
      ft.close()

      fe = open('ENERGIES.orig','r')
      feo = open('ENERGIES','w')
      for il,line in enumerate(fe.readlines()):
          if len(line.split())==0:
             continue
          if line.split()[0] == first:
             if il>0:
                feo.close()
                feo = open('ENERGIES','w')
             print(line[:-1],file=feo)
          else:
             print(line[:-1],file=feo)
      feo.close()
      fe.close()


  def get_cpmd_frame(self):
      if self.sort:
         self.sort_data()
      fe = open('ENERGIES','r')
      lines = fe.readlines()
      fe.close()
      self.nframe = len(lines)


  def get_ase_energy(self,direc):
      images = Trajectory(direc)
      self.nframe = len(images)
      return images


  def get_ase_data(self,images,trajonly):
      ''' getting data in the ase traj '''
      self.x=[]
      self.energy_nw = []
      for i,ind_ in enumerate(self.indexs):
          imag = images[ind_]
          if i==0:
             self.cell      =imag.get_cell()
             self.atom_name = imag.get_chemical_symbols()
             self.natom     = len(self.atom_name)

          if trajonly:
             e = 0.0
          else:
             e = imag.get_potential_energy()

          self.x.append(imag.positions)
          self.energy_nw.append(e)
      self.energy_nw = np.array(self.energy_nw)
      self.x = np.array(self.x)


  def get_siesta_energy(self,label='siesta'):
      fe = open(label+'.MDE','r')
      lines = fe.readlines()
      fe.close()
      l1= lines[1].split()
      l = lines[-1].split()
      if len(l)==0:
         l = lines[-2].split()
      self.nframe = int(l[0]) - int(l1[0]) + 1

      self.energy_nw = [] # np.zeros([self.nframe],dtype=np.float32)
      for line in lines:
          l = line.split()
          if l[1] != 'Step':
             # f = int(l[0]) - 1
             self.energy_nw.append(float(l[2])) # unit eV
             

  def get_siesta_cart(self,label='siesta'):
      fin = open('in.fdf','r') 
      lines= fin.readlines()
      fin.close()           # getting informations from input file

      for i,line in enumerate(lines):
          l = line.split()
          if len(l)>0:
             if l[0] == 'NumberOfSpecies':
                ns = int(l[1])
             if l[0] == 'NumberOfAtoms':
                self.natom = int(l[1])
             if l[0]=='%block':
                if l[1]=='ChemicalSpeciesLabel':
                   spl = i+1
                if l[1]=='AtomicCoordinatesAndAtomicSpecies':
                   atml= i+1

      sp = []
      for isp in range(ns):
          l = lines[spl+isp].split() 
          sp.append(l[2])

      for na in range(self.natom):
           l = lines[atml+na].split() 
           self.atom_name.append(sp[int(l[3])-1])

      fe = open(label+'.MD_CAR','r')
      lines = fe.readlines()
      fe.close()
      nl = len(lines)
      if nl-(self.natom+7)*self.nframe!=0:
         fra = (nl-(self.natom+7)*self.nframe)/(self.natom+7)
         print('-  %d frames more than expected, error case ... ... ...' %fra)
         exit()
         
      lsp = lines[5].split()
      nsp = [int(l) for l in lsp]
      xs = []
      if self.traj:
         his = TrajectoryWriter(self.structure+'.traj',mode='w')
      for nf in range(self.nframe):
          block = self.natom + 7
          nl = block*nf
          la = lines[nl+2].split()
          lb = lines[nl+3].split()
          lc = lines[nl+4].split()

          a = [float(la[0]),float(la[1]),float(la[2])]
          b = [float(lb[0]),float(lb[1]),float(lb[2])]
          c = [float(lc[0]),float(lc[1]),float(lc[2])]
          x = []
          il= 0
          for i,s in enumerate(nsp):
              for ns in range(s):
                  l = lines[nl+7+il].split()
                  xd = [float(l[0]),float(l[1]),float(l[2])]

                  x1 = xd[0]*a[0]+xd[1]*b[0]+xd[2]*c[0]
                  x2 = xd[0]*a[1]+xd[1]*b[1]+xd[2]*c[1]
                  x3 = xd[0]*a[2]+xd[1]*b[2]+xd[2]*c[2]

                  x.append([x1,x2,x3])
                  il += 1
          xs.append(x)
          if self.traj:
             A = Atoms(self.atom_name,x,cell=[a,b,c],pbc=[True,True,True])
             his.write(atoms=A)

      self.cell=np.array([a,b,c])

      if self.traj: 
         his.close()
      return xs


  def get_siesta_charges(self,label='siesta'):
      fo = open(label+'.out','r') 
      lines= fo.readlines()
      fo.close()            # getting informations from .out file
      qs  = []

      spec_atoms = {}
      obs        = {}
      for s in self.spec:
          spec_atoms[s] = []


      for i,s in enumerate(self.atom_name):
          spec_atoms[s].append(i) 

      nsp = {}
      for s in spec_atoms:
      	  nsp[s] = len(spec_atoms[s])

      iframe = 0
      spec_  = []
      for i,line in enumerate(lines):
          if line.find('Begin MD step')>=0:
             frame = int(line.split()[4])-1
         
          if line.find('mulliken: Atomic and Orbital Populations:')>=0:
             # print('-  current frame %d, MD step %d...' %(iframe,frame))
             if iframe==0:
                cl    = 0
                end_  = True
                while end_:
                      cl   += 1
                      line_ = lines[i+cl]
                      if line_.find('mulliken: Qtot')>=0:
                         end_ = False

                      if line_.find('Species:')>=0:
                         sl = 0
                         spec_.append(line_.split()[1])
                         
                         qline_ = lines[i+cl+1]
                         if qline_.find('Atom  Qatom  Qorb')<0:
                            print('-  an error case ... ... ')

                         qline_ = lines[i+cl+2]
                         ql_    = qline_.split()
                         nob    = len(ql_)

                         o      = 0
                         spec_end = True
                         while spec_end:
                               o += 1
                               qline_ = lines[i+cl+2+o]
                               ql_    = qline_.split()
                               if len(ql_)== nob+2:
                                  obs[spec_[-1]] = o
                                  spec_end = False

                # print('\n Qorb: \n',obs)
                q_     = np.zeros([self.natom])
                cl    = 0
                end_  = True
                while end_:
                      cl   += 1
                      line_ = lines[i+cl]
                      if line_.find('mulliken: Qtot')>=0:
                         end_ = False

                      if line_.find('Species:')>=0:
                         sl = 0
                         s_ = line_.split()[1]
                         # print('\n-  charges of species: %s \n' %s_)
                         
                         for i_ in range(nsp[s_]):
                             qline_ = lines[i+cl+2+(i_+1)*obs[s_]]
                             ql_    = qline_.split()

                             ai     = int(ql_[0])-1
                             q_[ai] = float(ql_[1])-self.p['vale_'+s_]

                         cl += 2+i_*obs[s_]
                qs.append(q_)
             else:
                q_    = np.zeros([self.natom])
                cl    = 0
                end_  = True
                while end_:
                      cl   += 1
                      line_ = lines[i+cl]
                      if line_.find('mulliken: Qtot')>=0:
                         end_ = False

                      if line_.find('Species:')>=0:
                         sl = 0
                         s_ = line_.split()[1]

                         for i_ in range(nsp[s_]):
                             qline_ = lines[i+cl+2+(i_+1)*obs[s_]]
                             ql_    = qline_.split()
                             
                             ai     = int(ql_[0])-1
                             q_[ai] = float(ql_[1])-self.p['vale_'+s_]

                         cl += 2+i_*obs[s_]
                qs.append(q_)

             iframe += 1
      return np.array(qs)


  def get_cpmd_data(self):
      fe = open('ENERGIES','r')
      lines = fe.readlines()
      fe.close()
      nl = len(lines)
      xs = []
      self.energy_nw = []
      if isfile('inp-nve'):
         inp='inp-nve'
      elif isfile('inp-nvt'):
         inp='inp-nvt'
      else: 
         print('-  error: inp-nvt or inp-nve file not found!')
         exit() 

      cell = get_lattice(inp=inp)  # surpose the cell is orthogonal
      self.cell = np.array(cell)
      
      for nf in range(nl):
          l = lines[nf].split()
          self.energy_nw.append(float(l[3])* 27.211396) ### a.u. to eV

      fg = open('GEOMETRY.xyz','r')
      lines = fg.readlines()
      fg.close()
      self.natom = len(lines) - 2
      for na in range(2,self.natom+2):
          self.atom_name.append(lines[na].split()[0])

      ft = open('TRAJECTORY','r')
      lines = ft.readlines()
      ft.close()

      if self.traj:
         his = TrajectoryWriter(self.structure+'.traj',mode='w')

      ii = 0
      for nf in range(nl):
          x = []
           
          for na in range(self.natom):
              l = lines[nf*self.natom+na].split()
              x.append([float(l[1])*0.52917721067,float(l[2])*0.52917721067,float(l[3])*0.52917721067])
          xs.append(x)

          if self.traj:
             if nf in self.indexs:
                A = Atoms(self.atom_name,x,cell=self.cell,pbc=[True,True,True])
                his.write(atoms=A)

      if self.traj: 
         his.close()
      return xs


  def get_nw_data(self,outs):
      i,ii = 0,0
      xs = []
      for out in outs:
          if out.find('.out')>=0:
             print('-  Getting molecular informations and nwchem datas from %s' %out)
             natm,atoms,X = out_xyz(out_file=out)

             xyz = out[:-4] + '.xyz'
             e, gradient = get_nw_gradient(out)

             if e is None or gradient is None:
                continue
             xs.append(X)
             self.energy_nw.append(e)
             ii += 1
      self.nframe = ii
      self.natom = natm
      self.atom_name = atoms
      return xs


  def get_neighbors(self,natm=None,atoms=None,R=None,rcut=None,rcuta=None):
      table  = [[] for i in range(natm)]
      atable = [[] for i in range(natm)]
      for i in range(0,natm-1):
          for j in range(i+1,natm):
              pair = atoms[i]+'-'+atoms[j]
              pairr = atoms[j]+'-'+atoms[i]
              r = R[i][j]
              if pair in rcut:
                 key = pair
              elif pairr in rcut:
                 key = pairr
              else:
                 print('-  an error case ... ... ...')
              if r<rcut[key]:
                 table[i].append(j)
                 table[j].append(i)
              if r<rcuta[key]:
                 atable[i].append(j)
                 atable[j].append(i)
      return natm,atoms,table,atable


  def get_table(self):
      for nf,X in enumerate(self.x):
          natm,atoms,table,atable = self.get_neighbors(natm=self.natom,
               atoms=self.atom_name,R=self.R[nf],rcut=self.r_cut,rcuta=self.rcuta)

          if nf==0:
             self.table  = table
             self.atable = atable
             print('-----------------------------------------------------------------\n')
             print('---    number of %4s atom in molecule %18s ---' %(natm,self.structure))
             print('-----------------------------------------------------------------\n')
          else:
             print('-  compute table of batch {0}/{1} ...\r'.format(nf,self.batch),end='\r')
             for na,tab in enumerate(table):
                 for atom in tab:
                     if not atom in self.table[na]:
                        self.table[na].append(atom)
             for na,tab in enumerate(atable):
                 for atom in tab:
                     if not atom in self.atable[na]:
                        self.atable[na].append(atom)
      print('\n')


  def compute_bond(self,x,natom,nframe):
      hfcell = 0.5 
      u    = np.linalg.inv(self.cell)
      x    = np.array(x)   #  project to the fractional coordinate
      xf   = np.dot(x,u) 

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
      
      vr_ = np.dot(vr,self.cell) # convert to ordinary coordinate
      R   = np.sqrt(np.sum(vr_*vr_,axis=3),dtype=np.float32)
      return R,vr_


  def get_bonds(self):
      self.bond = []
      self.max_nei = 0

      bondpair = []
      for i in range(self.natom): 
          if len(self.table[i])>self.max_nei:
             self.max_nei = len(self.table[i])
          for n_j,j in enumerate(self.table[i]):   
              if j!=i:
                 bn = self.atom_name[i]+'-'+self.atom_name[j]
                 bnr= self.atom_name[j]+'-'+self.atom_name[i]
                 if bn in self.bonds:
                    pair = [i,j]
                    pairr = [j,i]
                 elif bnr in self.bonds and bnr!=bn:
                    pair = [j,i]
                    pairr= [i,j]
                 else:
                    print('-  an error case encountered, %s not found in bondlist.' %bn)
                    exit()
                 # print('-  pair of atom %d & %d, %s' %(i,j,bn),pair)
                 if (not pair in self.bond) and (not pairr in self.bond):
                    self.bond.append(pair)
                    # print('-  adding pair of atom %d & %d, name %s' %(i,j,bn),pair)
      self.bond   = np.array(self.bond,dtype=np.int64)
      self.nbond  = len(self.bond)
      self.rbd    = self.R[:,self.bond[:,0],self.bond[:,1]]


  def compute_angle(self,R,vr):
      ang_ind,self.ang_i,self.ang_j,self.ang_k = {},[],[],[]
      for i in range(self.natom): # atomic energy of i
          for n_j,j in enumerate(self.atable[i]):   
              if j==i:
                 continue
              for n_k,k in enumerate(self.atable[j]):
                  if k != i and k!=j:
                     ang = str(i)+'-'+str(j)+'-'+str(k)
                     angr= str(k)+'-'+str(j)+'-'+str(i)
                     if (not ang in ang_ind) and (not angr in ang_ind):
                        ang_ind[ang] = True
                        ang_ind[angr] = True
                        self.ang_i.append(i)
                        self.ang_j.append(j)
                        self.ang_k.append(k)

      self.nang = len(self.ang_i)
      print('-  number of angles: %d ...\n' %self.nang)

      Rij = R[:,self.ang_i,self.ang_j]
      Rjk = R[:,self.ang_j,self.ang_k]
      # Rik = R[:,self.ang_i,self.ang_k]
      vik = vr[:,self.ang_i,self.ang_j] + vr[:,self.ang_j,self.ang_k]  
      Rik = np.sqrt(np.sum(vik*vik,axis=2),dtype=np.float32)

      Rij2= Rij*Rij
      Rjk2= Rjk*Rjk
      Rik2= Rik*Rik

      cos_theta = (Rij2+Rjk2-Rik2)/(2.0*Rij*Rjk)
      cos_theta = np.where(cos_theta>1.0,1.0,cos_theta)
      cos_theta = np.where(cos_theta<-1.0,-1.0,cos_theta)

      self.cos_theta = np.transpose(cos_theta,[1,0])
      self.theta     = np.arccos(self.cos_theta)


  def compute_torsion(self,R,vr):
      # print('-  compute torsion angles ...\n')
      tor_ind,self.tor_i,self.tor_j,self.tor_k,self.tor_l = {},[],[],[],[]
      for i in range(self.natom): # atomic energy of i
          for n_j,j in enumerate(self.atable[i]):   
              if j==i:
                 continue
              for n_k,k in enumerate(self.atable[j]):
                  if k == i or k==j:
                     continue
                  for n_l,l in enumerate(self.atable[k]):
                      if l==k or l==j or l==i:
                         continue
                      tor = str(i)+'-'+str(j)+'-'+str(k)+'-'+str(l)
                      torr= str(l)+'-'+str(k)+'-'+str(j)+'-'+str(i)
                      if (not tor in tor_ind) and (not torr in tor_ind):
                         tor_ind[tor] = True
                         tor_ind[torr] = True
                         self.tor_i.append(i)
                         self.tor_j.append(j)
                         self.tor_k.append(k)
                         self.tor_l.append(l)

      self.ntor = len(self.tor_i)
      print('-  number of torsion angles: %d \n' %self.ntor)
      nb = int(self.batch/self.minib)
      yu = self.batch-nb*self.minib
      if yu>0: nb += 1

      for b in range(nb): 
          st = b*self.minib
          ed = (b+1)*self.minib
          if ed>self.batch:
             ed=self.batch
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
          thet_ijk = np.arccos(c_ijk)

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
          thet_jkl = np.arccos(c_jkl)
          c = 1.0-c2jkl
          s_jkl = np.sqrt(np.where(c<0.0,0.0,c))
          strm  = np.transpose(s_jkl,[1,0])

          if b==0:
             self.s_jkl = strm
          else:
             self.s_jkl = np.concatenate((self.s_jkl,strm),axis=1)

          c_ijl = (Rij2+Rjl2-Ril2)/(2.0*Rij*Rjl)

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
      vr_ = []
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell = self.cell[0]*i + self.cell[1]*j+self.cell[2]*k
                  vr_.append(vr+cell)
      return vr_


  def compute_vdw(self,image_rs):
      vi,vj,vi_p,vj_p,self.vi,self.vj = [],[],[],[],[],[]
      for i in range(self.natom-1):
          for j in range(i+1,self.natom):
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


  def compute_hbond(self,image_rs):
      self.hb_i,self.hb_j,self.hb_k = [],[],[]
      hb_i,hb_j,hb_k = [],[],[]

      for i in range(self.natom): 
          if self.atom_name[i]!='H':
             for n_j,j in enumerate(self.atable[i]):   
                 if self.atom_name[j]=='H':
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


  def get_gulp_energy(self):
      q,ecoul,eself,evdw = [],[],[],[]
      print('-  get charges from gulp ... \n')
      for nf in range(self.batch):
          #print('*  get charges of batch {0}/{1} ...\r'.format(nf,self.batch),end='\r')
          A = Atoms(symbols=self.atom_name,
                    positions=self.x[nf],
                    cell=self.cell,
                    pbc=(1, 1, 1))
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
          cell=self.cell,
          pbc=(1, 1, 1))
      Qe= qeq(p=self.p,atoms=A)

      for nf in range(self.batch):
          # print('*  get charges of batch {0}/{1} ...\r'.format(nf,self.batch),end='\r')
          A = Atoms(symbols=self.atom_name,
                    positions=self.x[nf],
                    cell=self.cell,
                    pbc=(1, 1, 1))
          positions = A.get_positions()
          cell      = A.get_cell()
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

