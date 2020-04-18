from __future__ import print_function
from os import system, getcwd, chdir,listdir
from os.path import isfile,exists,isdir
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.siesta import Siesta
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Ry
from ..setRcut import setRcut
import datetime
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ..molecule import press_mol


def amp_data(lab='amp',
             dft='siesta',
             dic='/home/gfeng/siesta/train/ethane',
             batch=800,sort=False):
    ''' prepare data for AMP '''
    md = MDtoData(dft=dft,direc=dic,
                  batch=batch,sort=sort)

    if len(md.energy_nw)<batch:
       batch = len(md.energy_nw)

    box = [md.cell[0][0],md.cell[1][1],md.cell[2][2]]
    his = TrajectoryWriter(lab+'.traj',mode='w')

    for i in range(batch):
        md.energy_nw[i]
        x = np.mod(md.x[i],box)  
        A = Atoms(md.atom_name,x,cell=md.cell,pbc=[True,True,True])

        e = float(md.energy_nw[i]) 
        calc = SinglePointCalculator(A,energy=e,
                                     free_energy=float(md.max_e),
                                     forces=md.forces[i])

        A.set_calculator(calc)
        his.write(atoms=A)
        del A
    his.close()


def get_lattice(inp='inp-nve'):
    ''' get cpmd lattice constance '''
    finp = open(inp,'r')
    il = 0
    cell = []
    readlatv = False
    readlat  = False
    for line in finp.readlines():
        l = line.split()
        if line.find('CELL')>=0 and  line.find('VECTORS')>=0:
           readlatv = True
        elif line.find('CELL')>=0 and  line.find('VECTORS')<0:
           if line.find('RESTART')<0:
              readlat  = True
        if readlatv and il < 4:
           if not il==0:
              cell.append( [float(l[0]),float(l[1]),float(l[2])])
           il += 1
        if readlat and il < 2:
           if not il==0:
              cell = [[float(l[0]),0.0,0.0],
                      [0.0,float(l[1])*float(l[0]),0.0],
                      [0.0,0.0,float(l[2])*float(l[0])]]
           il += 1
    finp.close()
    return cell


class MDtoData(object):
  """ Collecting datas for mathine learning for bond order potential"""
  def __init__(self,structure='molecule',botol=0.001,direc=None,
               vdwcut=10.0,rcut=None,rcuta=None,
               dft='ase',atoms=None,
               batch=1000,minib=100,
               p=None,spec=['C','H','O','N','F','Al'],
               sort=False,
               checkMol=False,
               traj=False,
               nindex=[]):
      self.sort      = sort
      self.checkMol  = checkMol
      self.structure = structure
      self.dft       = dft
      self.energy_nw,self.energy_bop = [],[]
      self.table = []
      self.atom_name = []
      self.min_e = 0.0
      self.max_e = 0.0
      self.vdwcut = vdwcut

      if p is None:
         self.p  = {'vale_C':4,'vale_H':1,'vale_O':6,'vale_N':5,'vale_Al':3,'vale_F':7}
      else:
         self.p  = p

      self.spec  = spec
      
      self.rcut  = rcut
      self.rcuta = rcuta
      self.traj  = traj
      self.cell  = np.array([(10, 0, 0), (0, 10, 0), (0, 0, 10)])
      self.status= True
      self.initialize()

      self.rcut,self.rcuta,re = setRcut(self.bonds)
      
      for bd in self.bonds:
          b = bd.split('-')
          bdr = b[1]+'-'+b[0]
          if not bdr in self.rcut:
             self.rcut[bdr]  = self.rcut[bd]
          elif not bd in self.rcut:
             self.rcut[bd]   = self.rcut[bdr]
          if not bdr in self.rcuta:
             self.rcuta[bdr] = self.rcuta[bd] 
          elif not bd in self.rcuta:
             self.rcuta[bd]  = self.rcuta[bdr]

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
         indexs      = random.sample(pool,self.batch)
         indexs      = np.array(indexs)
         indices     = indexs.argsort()
         self.indexs = indexs[indices]
         forces,presses = None, None

         if self.dft=='cpmd':
            xs = self.get_cpmd_data()
         elif self.dft=='siesta':
            xs,cells = self.get_siesta_cart()
            forces,presses,qs = self.get_siesta_forces()
         elif self.dft=='ase':
            self.get_ase_data(images,trajonly)

         if self.dft!='ase':
            xs             = np.array(xs)
            self.x         = xs[self.indexs]

            cells          = np.array(cells)
            self.cells     = cells[self.indexs]

            forces         = np.array(forces)
            self.forces    = forces[self.indexs]
            self.presses   = presses[self.indexs]
            self.qs        = qs[self.indexs]

            energy_nw      = np.array(self.energy_nw)
            self.energy_nw = energy_nw[self.indexs]
            self.nframe=self.batch
      else:
         if self.dft=='ase':
            self.nframe = len(pool)
            nb = int(self.batch/self.nframe+1)

            for i in range(nb):
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
            self.indexs = np.arange(self.nframe-nni)
            if self.dft=='cpmd':
               xs,forces = self.get_cpmd_data(indexs)
            elif self.dft=='siesta':
               xs,cells    = self.get_siesta_cart()
               forces,presses,qs = self.get_siesta_forces()
            self.x = np.array(xs)
            self.cells = np.array(cells)
            forces = np.array(forces)
            self.energy_nw = np.array(self.energy_nw)
            if len(forces)>= len(self.indexs):
               self.forces    = forces[self.indexs]
            else:
               self.forces = None
            if len(forces)>= len(self.indexs):
               self.qs        = qs[self.indexs]
            else:
               self.qs = None
            if len(presses)>0:
               self.presses   = presses[self.indexs]

      # print(self.indexs)
      self.atom_name = np.array(self.atom_name)
      self.init_names()

      self.min_e = min(self.energy_nw)
      self.max_e = max(self.energy_nw)
      R,vr       = self.compute_bond(self.x,self.natom,self.nframe)
 
      self.vrs   = self.compute_image(vr)

      if not direc is None:
         if self.dft!='ase':
            chdir(cdir)


  def initialize(self):
      self.bonds = []
      for sp1 in self.spec:
          for sp2 in self.spec:
              bd  =  sp1+'-'+sp2
              bdr = sp2+'-'+sp1
              if (not bd in self.bonds) and (not bdr in self.bonds):
                 self.bonds.append(bd)


  def init_names(self):
      self.bond_name= []
      for an1 in self.atom_name:
          for an2 in self.atom_name:
                self.bond_name.append(an1+'-'+an2)
      # print(len(self.atom_name),self.atom_name)
      self.bond_name = np.reshape(self.bond_name,[self.natom,self.natom])


  def get_traj(self):
      self.uc= np.linalg.inv(self.cell)
      images = []
      his    = TrajectoryWriter(self.structure+'.traj',mode='w')
      indexs = np.array(self.indexs)
      ind    = indexs.argsort()
      batch_ = self.batch if self.nframe>self.batch else self.nframe

      for i in range(batch_):
          ii  = ind[i]
          xf  = np.dot(self.x[ii],self.uc) 
          xf  = np.mod(xf,1.0)  
          self.x[ii] = np.dot(xf,self.cell)
          if self.qs is None:
             c = None
          else:
             c = self.qs[i]
          A   = Atoms(self.atom_name,self.x[ii],
                      charges=c,
                      cell=self.cells[ii],pbc=[True,True,True])
          if self.checkMol:
             A =  press_mol(A)

          calc = SinglePointCalculator(A,energy=float(self.energy_nw[ii]))
          A.set_calculator(calc)
          his.write(atoms=A)
          images.append(A)
      his.close()
      return images


  def get_images(self):
      self.uc= np.linalg.inv(self.cell)
      images = []
      indexs = np.array(self.indexs)
      ind    = indexs.argsort()
      batch_ = self.batch if self.nframe>self.batch else self.nframe
      
      for i in range(batch_):
          ii  = ind[i]
          xf  = np.dot(self.x[ii],self.uc) 
          xf  = np.mod(xf,1.0)  
          self.x[ii] = np.dot(xf,self.cell)

          A    = Atoms(self.atom_name,self.x[ii],
                        charges=self.qs[i],
                        cell=self.cells[ii],pbc=[True,True,True])
          calc = SinglePointCalculator(A,energy=float(self.energy_nw[ii]))
          A.set_calculator(calc)
          images.append(A)
      return images


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


  def get_cpmd_frame(self):
      if self.sort:
         self.sort_data()
      fe = open('ENERGIES','r')
      lines = fe.readlines()
      fe.close()
      self.nframe = len(lines)


  def get_cpmd_data(self):
      fe = open('ENERGIES','r')
      lines = fe.readlines()
      fe.close()
      nl = len(lines)
      xs,gs = [],[]
      self.energy_nw = []
      if isfile('inp-nve'):
         inp='inp-nve'
      elif isfile('inp-nvt'):
         inp='inp-nvt'
      else: 
         print('*  error: inp-nvt or inp-nve file not found!')
         exit() 

      cell      = get_lattice(inp=inp)  # surpose the cell is orthogonal
      self.cell = np.array(cell)

      for nf in range(nl):
          l = lines[nf].split()
          if len(l)>0:
             self.energy_nw.append(float(l[3])*27.211396)  # to eV

      fg = open('GEOMETRY.xyz','r')
      lines = fg.readlines()
      fg.close()
      self.natom = len(lines) - 2
      for na in range(2,self.natom+2):
          self.atom_name.append(lines[na].split()[0])
       
      if isfile('FTRAJECTORY'):
         ft = open('FTRAJECTORY','r')
      else:
         print('*  warning: forces not found ...')
         ft = open('TRAJECTORY','r')
      lines = ft.readlines()
      ft.close()

      ii = 0
      au2ev = 27.211396/0.52917721067
      for nf in range(nl):
          x = []
          g = []
          for na in range(self.natom):
              l = lines[nf*self.natom+na].split()
              x.append([float(l[1])*0.52917721067,float(l[2])*0.52917721067,float(l[3])*0.52917721067])
              if len(l)==10:
                 # g.append([float(l[7])/0.52917721067,float(l[8])/0.52917721067,float(l[9])/0.52917721067])
                 g.append([float(l[7])*au2ev,float(l[8])*au2ev,float(l[9])*au2ev])
          xs.append(x)
          if len(l)==10:
             gs.append(g)
      gs = np.array(gs)
      return xs,gs


  def sort_data(self):
      ''' for cpmd data '''
      system('mv ENERGIES ENERGIES.orig')
      system('mv TRAJECTORY TRAJECTORY.orig')
      system('mv FTRAJECTORY FTRAJECTORY.orig')

      ft = open('FTRAJECTORY.orig','r')
      fto = open('FTRAJECTORY','w')
      first = None
      for il,line in enumerate(ft.readlines()):
          if line.find('<<<<<<  NEW DATA  >>>>>>') >0:
             fto.close()
             fto = open('FTRAJECTORY','w')
             first = None
          else:
             if first is None:
                first = line.split()[0]
             print(line[:-1],file=fto)
      fto.close()
      ft.close()

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
                # print(line[:-1])
                feo = open('ENERGIES','w')
             print(line[:-1],file=feo)
          else:
             print(line[:-1],file=feo)
      feo.close()
      fe.close()


  def get_nw_data(self,outs):
      i,ii = 0,0
      xs = []
      for out in outs:
          if out.find('.out')>=0:
             print('*  Getting molecular informations and nwchem datas from %s' %out)
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


  def get_siesta_energy(self,label='siesta'):
      if isfile(label+'.MDE'):
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
      else:
         self.energy_nw = [0.0]
         self.nframe = 1


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
                elif l[1]=='AtomicCoordinatesAndAtomicSpecies':
                   atml= i+1
                   zmat = False
                elif l[1]=='Zmatrix':
                   zmat = True
                   atml = i+1

      sp = []
      for isp in range(ns):
          l = lines[spl+isp].split() 
          sp.append(l[2])

      atoml = self.natom+2 if zmat else self.natom
      for na in range(atoml):
          l = lines[atml+na].split() 
          if zmat:
             if len(l)>2:
                self.atom_name.append(sp[int(l[0])-1])
          else:
             self.atom_name.append(sp[int(l[3])-1])

      if not isfile(label+'.MD_CAR'):
         return []
         
      fe = open(label+'.MD_CAR','r')
      lines = fe.readlines()
      fe.close()
      nl = len(lines)
      if nl-(self.natom+7)*self.nframe!=0:
         fra = (nl-(self.natom+7)*self.nframe)/(self.natom+7)
         print('-  warning: %d frames more than expected ... ... ...' %fra)
         
      lsp    = lines[5].split()
      nsp    = [int(l) for l in lsp]
      xs     = []
      cells  = []
     
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
                  for d in range(3):
                      if xd[d]>1.0:
                         xd[d] = xd[d] - 1.0
                      if xd[d]<0.0:
                         xd[d] = xd[d] + 1.0

                  x1 = xd[0]*a[0]+xd[1]*b[0]+xd[2]*c[0]
                  x2 = xd[0]*a[1]+xd[1]*b[1]+xd[2]*c[1]
                  x3 = xd[0]*a[2]+xd[1]*b[2]+xd[2]*c[2]

                  x.append([x1,x2,x3])
                  il += 1
          xs.append(x)
          cells.append(np.array([a,b,c]))

      self.cell=np.array([a,b,c])
      return xs,cells


  def get_siesta_forces(self,label='siesta'):
      fo = open(label+'.out','r') 
      lines= fo.readlines()
      fo.close()           # getting informations from input file
      forces  = []
      presses = []

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
          if line.find('Atomic forces (eV/Ang)')>=0:
             force   = []
             ln = lines[i+1]
             if ln.find('----------------------------------------')<0:
                for na in range(self.natom):
                   fl = lines[na+i+1] #.split()
                   if fl.find('siesta:')<0:
                       f1 = fl[6:18]
                       f2 = fl[18:30]
                       f3 = fl[30:42]
                       force.append([float(f1),float(f2),float(f3)])
             if len(force)>0:
                forces.append(force)
          elif line.find('Stress tensor (total)')>=0:
              press   = []
              for l in range(3):
                  pl = lines[l+i+1].split()
                  press.append([float(pl[0]),float(pl[1]),float(pl[2])])
              presses.append(press)
          elif line.find('mulliken: Atomic and Orbital Populations:')>=0:
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
                         # print('\n-  charges of species: %s \n' %s_)
                         
                         for i_ in range(nsp[s_]):
                             qline_ = lines[i+cl+2+(i_+1)*obs[s_]]
                             ql_    = qline_.split()

                             ai     = int(ql_[0])-1
                             q_[ai] = self.p['vale_'+s_] - float(ql_[1])  # original is oppose!

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
                # print('-  frame: %d' %iframe,q_)
                qs.append(q_)

             iframe += 1
      return np.array(forces),np.array(presses),np.array(qs)


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


  def compute_image(self,vr):
      vr_ = []
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell = self.cell[0]*i + self.cell[1]*j+self.cell[2]*k
                  vr_.append(vr+cell)
      return vr_
    

  def pdf(self,bins=0.01,rcut=4.0,pdf_plot=True):
      print('-  plotting bond pair distribution function ...')
      binsl,hists  = {},{}
      for bd in self.bonds:
          b = bd.split('-')
          bdr = b[0]+'-'+b[1]

          natom1 = np.sum(np.where(self.atom_name==b[0]))
          natom2 = np.sum(np.where(self.atom_name==b[1]))
          s=0.5 if b[0]==b[1] else 1.0
             
          ind_ = np.where(np.logical_or(self.bond_name==bd,self.bond_name==bdr))
          for i,vr in enumerate(self.vrs):
              vr2 = np.square(vr)
              R   = np.sqrt(np.sum(vr2,axis=3),dtype=np.float32)
              R_  = R[:,ind_[0],ind_[1]] 
              rbd_ = R_ if i==0 else np.concatenate([rbd_,R_],axis=1)

          rbd_= np.reshape(rbd_,[-1])
          id_ = np.where(rbd_>0.000001)
          rbd_= rbd_[id_]

          hist,bin_ = np.histogram(rbd_,range=(0.0,rcut),bins=100,density=True)
          bin_ = bin_[:-1]

          binsl[bd]  = bin_
          hists[bd] = hist

          if pdf_plot:
             plt.figure()   
             plt.ylabel('Pair Distribution Function')
             plt.xlabel(r'Radius $(\AA)$')
             plt.xlim(0,rcut)
             plt.ylim(0,np.max(hist)+0.01)
             # plt.hist(rbd_,bin_=32,density=1,alpha=0.01,label='%s' %bd)
             plt.plot(bin_,hist,alpha=0.01,label='%s' %bd)
             plt.legend(loc='best')
             plt.savefig('%s_bh.eps' %bd) 
             plt.close() 
      return binsl,hists
         

  def close(self):
      self.energy_nw,self.energy_bop = None,None
      self.table     = None
      self.atom_name = None
      self.x         = None
      self.forces    = None
      self.p         = None
      self.atom_name = None

      
if __name__ == '__main__':
   dic_list = {'nmmol':'/home/gfeng/cpmd/train/nmr/nm12'}
   TrainPrepare(dic_list)

