from unittest import result
from .irff_np import IRFF_NP
from ase.io import read,write
from .molecule import molecules
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
import matplotlib.pyplot as plt


def getAtomsToMove(i,j,j_,ToBeMove,neighbors,ring=False):
    ToBeMove.append(j_)
    for n in neighbors[j_]:
        if n!=i:
           if n not in ToBeMove:
              ToBeMove,ring = getAtomsToMove(i,j,n,ToBeMove,neighbors)
        elif j_!=j and n==i:
           ring = True
    return ToBeMove,ring


def get_group(i,j,atoms):
    positions = atoms.get_positions()
    center    = 0.5*(positions[i] + positions[j]) 
    vij = positions[j] - positions[i]
    rij = np.sqrt(np.sum(np.square(vij)))
    vij = vij/rij

    group_i,group_j = [],[]

    for i_,x in enumerate(positions):
        v    = x - center
        dot_ = np.dot(v,vij)
        # print(v,vij,dot_)
        if dot_<=0.0:
           group_i.append(i_)
        else:
           group_j.append(i_)
    return group_i,group_j


def getNeighbor(natom,r,rcut,bo,botol=0.0):
    neighbors = [[] for _ in range(natom)]
    for i in range(natom-1):
        for j in range(i+1,natom):
            # print(i,j,r[i][j],rcut[i][j])
            if r[i][j]<rcut[i][j] and bo[i][j]>=botol:
               # print(i,j)
               neighbors[i].append(j)
               neighbors[j].append(i)
    return neighbors


def getBonds(natom,r,rcut,bo,botol=0.0):
    bonds = [] 
    for i in range(natom-1):
        for j in range(i+1,natom):
            # print(r[i][j],rcut[i][j],bo[i][j],botol)
            if r[i][j]<rcut[i][j] and bo[i][j]>=botol:
               bonds.append((i,j))
    return bonds


def add_zmat_atom(fi,se,th,neighbors,zmat_id,zmat_index):
    for i in neighbors[fi]:
        if i not in zmat_id:
           zmat_id.append(i)
           zmat_index.append([fi,se,th])
           zmat_id,zmat_index=add_zmat_atom(i,fi,se,neighbors,zmat_id,zmat_index)
    return zmat_id,zmat_index


def get_zmat_variable(i,j,k,l,positions):
    vij  = positions[i]-positions[j]
    rij  = np.sqrt(np.sum(np.square(vij)))
    uij  = vij/rij

    vkj  = positions[k]-positions[j]
    rkj  = np.sqrt(np.sum(np.square(vkj)))
    ukj  = vkj/rkj

    cos_ = np.dot(uij,ukj)
    ang  = np.arccos(cos_)*180.0/3.141593

    ujk  = -ukj
    vlk  = positions[l]-positions[k]
    rlk  = np.sqrt(np.sum(np.square(vlk)))
    ulk  = vlk/rlk

    vi   = np.cross(uij,ukj)
    ri   = np.sqrt(np.sum(np.square(vi)))
    ui   = vi/ri

    vj   = np.cross(ulk,ujk)
    rj   = np.sqrt(np.sum(np.square(vj)))
    uj   = vj/rj
    
    cos_ = np.dot(ui,uj)
    vk   = np.cross(ui,uj)
    rk   = np.sqrt(np.sum(np.square(vk)))
    if rk==0.0:
       tor = 0.0
    else:
       uk   = vk/rk
       # print(ui,uj,cos_,vk,uk)
       s    = np.dot(uk,ujk)
       tor  = s*np.arccos(cos_)*180.0/3.141593
    return rij,ang,tor


def get_zmat_angle(i,j,k,positions):
    vij  = positions[i]-positions[j]
    rij  = np.sqrt(np.sum(np.square(vij)))

    vkj  = positions[k]-positions[j]
    rkj  = np.sqrt(np.sum(np.square(vkj)))

    uij  = vij/rij
    ukj  = vkj/rkj

    cos_ = np.dot(uij,ukj)
    if cos_>1.0: cos_=1.0
    if cos_<-1.0: cos_=-1.0

    ang  = np.arccos(cos_)*180.0/3.141593
    return rij,ang


def get_zmatrix(atoms,zmat_id,zmat_index):
    ''' get the zmatrix of a molecular'''
    zmatrix = []
    for i,iatom in enumerate(zmat_id):
        if zmat_index[i][0]==-1 and zmat_index[i][1]==-1 and zmat_index[i][2]==-1:
           zmatrix.append([0.0,0.0,0.0])
        elif zmat_index[i][0]!=-1 and zmat_index[i][1]==-1 and zmat_index[i][2]==-1:
           v = atoms.positions[zmat_id[i]]-atoms.positions[zmat_index[i][0]]
           r = np.sqrt(np.sum(np.square(v)))
           zmatrix.append([r,0.0,0.0])
        elif zmat_index[i][0]!=-1 and zmat_index[i][1]!=-1 and zmat_index[i][2]==-1:
           r,ang = get_zmat_angle(zmat_id[i],zmat_index[i][0],
                                   zmat_index[i][1],
                                   atoms.positions)
           zmatrix.append([r,ang,0.0])
        else:
           r,ang,tor = get_zmat_variable(zmat_id[i],zmat_index[i][0],
                                         zmat_index[i][1],zmat_index[i][2],
                                         atoms.positions)
           zmatrix.append([r,ang,tor])
    return zmatrix

 
def check_zmat(atoms=None,rmin=0.8,rmax=1.2,angmax=15.0,
               zmat_id=None,zmat_index=None,InitZmat=None):
    zmatrix = get_zmatrix(atoms,zmat_id,zmat_index)
    # print(zmatrix)
    Df_     = 0
    score   = []
    zvs     = []
    zvlo_   = []
    zvhi_   = []
    zvhi    = 0.0
    zv      = None
    for i in range(len(zmat_id)):
        zvlo    = 0.0
        if zmat_index[i][2]!=-1:                    # check torsion
           ang = InitZmat[i][2]-zmatrix[i][2]
           if ang>180.0:
              ang = ang - 360.0
           elif ang<-180.0:
              ang = ang + 360.0
           ang_ = abs(ang)
           if ang_>=1.5*angmax: 
              Df_ += 0.6
              zv   = (i,2)
              zvhi = ang
              zvs.append(zv)
              score.append(0.6)
              zvlo_.append(zvlo)
              zvhi_.append(zvhi)
           elif ang_>=angmax: 
              Df_ += 0.3

        if zmat_index[i][1]!=-1:                    # check angle
           ang  = InitZmat[i][1]-zmatrix[i][1]
           ang_ = abs(ang)
           if ang_>=1.5*angmax: 
              Df_ += 0.8
              zv   = (i,1)
              zvhi = ang
              zvs.append(zv)
              score.append(0.8)
              zvlo_.append(zvlo)
              zvhi_.append(zvhi)
           elif ang_>=angmax: 
              Df_ += 0.5
              zv   = (i,1)
              zvhi = ang
              zvs.append(zv)
              score.append(0.5)
              zvlo_.append(zvlo)
              zvhi_.append(zvhi)

        if zmat_index[i][0]!=-1: 
           r_   = zmatrix[i][0]/InitZmat[i][0]
           zvhi = InitZmat[i][0] - zmatrix[i][0]
           if r_<=0.9*rmin or r_>=1.1*rmax:
              Df_ += 1.0
              zv   = (i,0)
              zvs.append(zv)
              score.append(1.0)
              zvlo_.append(zvlo)
              zvhi_.append(zvhi)
           elif r_<=rmin or r_>=rmax:
              Df_ += 0.64
              zv   = (i,0)
              zvs.append(zv)
              score.append(0.64)
              zvlo_.append(zvlo)
              zvhi_.append(zvhi)
           elif r_<=1.1*rmin or r_>=0.9*rmax: 
              Df_ += 0.35
    if len(score)>=1:
       m = np.argmax(score)
       zv,zvlo,zvhi = zvs[m],zvlo_[m],zvhi_[m] 
    return Df_,zmatrix,zv,zvlo,zvhi


class AtomDance(object):
  def __init__(self,atoms=None,poscar=None,nn=True,ffield='ffield.json',
               rotAng=40.0,angmax=30.0,freeatoms=None,FirstAtom=None,
               rmin=0.4,rmax=1.25,botol=0.0):
      self.rmin          = rmin
      self.rmax          = rmax
      self.botol         = botol
      self.BondDistrubed = []
      self.rotAng        = rotAng
      self.angmax        = angmax
      self.FirstAtom     = FirstAtom
      if atoms is None:
         if poscar is None:
            atoms  = read('poscar.gen')
         else:
            atoms  = read(poscar)

      self.ir = IRFF_NP(atoms=atoms,
                        libfile=ffield,
                        rcut=None,
                        nn=nn)
      self.natom     = self.ir.natom
      self.atom_name = self.ir.atom_name
      spec           = self.ir.spec
      self.atoms     = self.ir.atoms
      self.mass      = atoms.get_masses()
      if freeatoms is None:
         self.freeatoms = [i for i in range(self.natom)]
      else: 
         self.freeatoms = freeatoms
      self.zmat_index= None
      self.InitZmat  = None
     
      label_dic      = {}
      for sp in self.atom_name:
          if sp in label_dic:
             label_dic[sp] += 1
          else:
             label_dic[sp]  = 1
      self.label = ''
      for sp in spec:
          if sp in label_dic:
             self.label += sp+str(label_dic[sp])

      self.ir.calculate_Delta(atoms)
      self.InitBonds = getBonds(self.natom,self.ir.r,self.rmax*self.ir.re,self.ir.bo0,
                                botol=self.botol)
      self.freebonds = self.InitBonds
      self.neighbors = getNeighbor(self.natom,self.ir.r,self.rmax*self.ir.re,self.ir.bo0,
                                   botol=self.botol)
      
      self.InitZmat = np.array(self.get_zmatrix(atoms))
      self.write_zmat(self.InitZmat)

  def get_zmatrix(self,atoms):
      ''' get the zmatrix of a molecular'''
      zmatrix = []
      if self.zmat_index is None:
         self.get_zmat_index(atoms)
      for i,iatom in enumerate(self.zmat_id):
          if self.zmat_index[i][0]==-1 and self.zmat_index[i][1]==-1 and self.zmat_index[i][2]==-1:
             zmatrix.append([0.0,0.0,0.0])
          elif self.zmat_index[i][0]!=-1 and self.zmat_index[i][1]==-1 and self.zmat_index[i][2]==-1:
             v = atoms.positions[self.zmat_id[i]]-atoms.positions[self.zmat_index[i][0]]
             r = np.sqrt(np.sum(np.square(v)))
             zmatrix.append([r,0.0,0.0])
          elif self.zmat_index[i][0]!=-1 and self.zmat_index[i][1]!=-1 and self.zmat_index[i][2]==-1:
             r,ang = get_zmat_angle(self.zmat_id[i],self.zmat_index[i][0],
                                         self.zmat_index[i][1],
                                         atoms.positions)
             zmatrix.append([r,ang,0.0])
          else:
             r,ang,tor = get_zmat_variable(self.zmat_id[i],self.zmat_index[i][0],
                                         self.zmat_index[i][1],self.zmat_index[i][2],
                                         atoms.positions)
             zmatrix.append([r,ang,tor])
      return zmatrix

  def get_zmat_index(self,atoms):
      self.zmat_index = []
      self.zmat_id    = []
      specs           = atoms.get_chemical_symbols()
      self.mols       = molecules(self.natom,specs,atoms.positions,
                                  cell=atoms.cell,
                                  table=self.neighbors)
      for m in self.mols:
          if len(m.mol_index)==1:
             self.zmat_index.append([-1,-1,-1])
             self.zmat_id.append(m.mol_index[0])
          else:
             for i in m.mol_index:
                 first = i
                 if len(self.neighbors[i])==1:
                    first = i
                    break 
             if not self.FirstAtom is None:
                if self.FirstAtom in m.mol_index:
                   first = self.FirstAtom
             self.zmat_index.append([-1,-1,-1])
             self.zmat_id.append(first)

             second = None
             if m.natom>=2:
                second= self.neighbors[first][0]
                self.zmat_index.append([first,-1,-1])
                self.zmat_id.append(second)

             third = None
             if m.natom>=3:
                for i in self.neighbors[second]:
                    if i != first:
                       if not third is None:
                          if len(self.neighbors[i])>len(self.neighbors[third]):
                             third = i
                       else:      
                          third = i

             if not third is None:
                self.zmat_index.append([second,first,-1])  
                self.zmat_id.append(third)
 
                self.zmat_id,self.zmat_index = add_zmat_atom(second,first,third,
                                     self.neighbors,self.zmat_id,self.zmat_index)
                self.zmat_id,self.zmat_index = add_zmat_atom(third,second,first,
                                     self.neighbors,self.zmat_id,self.zmat_index)

  def zmat_to_cartation(self,atoms,zmat):
      for i in range(len(self.zmat_id)):
          atomi = self.zmat_id[i]
          atomj = self.zmat_index[i][0]
          atomk = self.zmat_index[i][1]
          atoml = self.zmat_index[i][2]
          r     = zmat[i][0]
          ang   = zmat[i][1]
          tor   = zmat[i][2]
          if self.zmat_index[i][0]==-1 and self.zmat_index[i][1]==-1 and self.zmat_index[i][2]==-1:
             continue
          elif self.zmat_index[i][0]!=-1 and self.zmat_index[i][1]==-1 and self.zmat_index[i][2]==-1:
             atoms = self.stretch_atom(atomi,atomj,zmat[i][0],atoms)
          elif self.zmat_index[i][0]!=-1 and self.zmat_index[i][1]!=-1 and self.zmat_index[i][2]==-1:
             atoms = self.rotate_atom(atoms,atomi,atomj,atomk,atoml,r=r,ang=ang)
          else:
             atoms = self.rotate_atom(atoms,atomi,atomj,atomk,atoml,r=r,ang=ang,tor=tor)
      return atoms

  def stretch_atom(self,i,j,r,atoms):
      vij = atoms.positions[i]-atoms.positions[j]
      rij = np.sqrt(np.sum(np.square(vij)))
      moveDirection = vij/rij
      atoms.positions[i] = atoms.positions[j] + r*moveDirection  
      return atoms

  def rotate_atom(self,atoms,i,j,k,l,r=None,ang=None,tor=None):
      vij = atoms.positions[i] - atoms.positions[j] 
      vkj = atoms.positions[k] - atoms.positions[j]
      rkj = np.sqrt(np.sum(np.square(vkj)))
      rij = np.sqrt(np.sum(np.square(vij)))
      
      ux  = vkj/rkj
      if tor is None or tor == 0.0:
         uij = vij/rij
         rk  = np.dot(uij,ux)
         vy  = uij - rk*ux
         uy  = vy/np.sqrt(np.sum(np.square(vy)))
      else:
         vkl = atoms.positions[k] - atoms.positions[l] 
         rkl = np.sqrt(np.sum(np.square(vkl)))
         ukl = vkl/rkl
         rk  = np.dot(ukl,ux)
         vy  = ukl - rk*ux
         uy  = vy/np.sqrt(np.sum(np.square(vy))) 

      a   = ang*3.141593/180.0
      ox  = r*np.cos(a)*ux
      ro  = r*np.sin(a)
      oy  = ro*uy
      p   = ox + oy
      atoms.positions[i] = atoms.positions[j] + p

      if not tor is None:
         vij = p
         uz  = np.cross(ux,uy)
         o_  = atoms.positions[j] + ox
         a   = tor*3.141593/180.0
         p   = ro*np.cos(a)*uy + ro*np.sin(a)*uz
         atoms.positions[i] = o_ + p
      return atoms

  def get_rotate(self):
      groups = self.get_groups()
      for i_,bd in enumerate(self.freebonds):
          i,j = bd
          if groups[i] is None and groups[j] is None:
             continue
          elif groups[i] is None:
             group_ = groups[j]
             axis   = [i,j]
             o      = j
          elif groups[j] is None:
             group_ = groups[i]
             axis   = [j,i]
             o      = i
          else:
             if len(groups[i])>len(groups[j]):
                group_ = groups[i]
                axis   = [j,i]
                o      = i
             else:
                group_ = groups[j]
                axis   = [i,j]
                o      = j
          images = self.rotate(atms=group_,axis=axis,o=o,rang=self.rotAng,nbin=30,traj='md.traj')

  def get_groups(self):
      groups = [None for i in range(self.natom)]
      for bd in self.freebonds:
          i,j = bd
          group_j = []
          group_j,ring = getAtomsToMove(i,j,j,group_j,self.neighbors)
          if ring:
             group_j = None
          else:
             group_j.remove(j)
          groups[j] = group_j

          group_i = []
          group_i,ring = getAtomsToMove(j,i,i,group_i,self.neighbors)
          if ring:
             group_i = None
          else:
             group_i.remove(i)
          groups[i] = group_i
      return groups

  def get_freebond(self,freeatoms=None):
      if freeatoms is None:
         self.freeatoms = [i for i in range(self.natom)]
         self.freebonds = self.InitBonds
      else:
         self.freeatoms = freeatoms
         self.freebonds = []
         for bd in self.InitBonds:
             i,j = bd
             if i in freeatoms or j in freeatoms:
                self.freebonds.append(bd)

  def bond_momenta_bigest(self,atoms):
      ratio = []
      s     = []
      for bd in self.InitBonds:
          i,j = bd
          ratio_     = self.ir.r[i][j]/self.ir.re[i][j]
          s_         = ratio_ -1.0
          s.append(s_)
          ratio.append(abs(s_))

      m_  = np.argmax(ratio)
      i,j = self.InitBonds[m_]
      s_  = s[m_] 
      if s_>=0.0:
         sign = 1.0
      else:
         sign = -1.0
      atoms = self.set_bond_momenta(i,j,atoms,sign=sign)
      return atoms
      

  def bond_momenta(self,atoms):
      ratio = []
      self.ir.calculate_Delta(atoms)
      for bd in self.freebonds:
          i,j = bd
          if bd not in self.BondDistrubed:
             s_ = self.ir.r[i][j]/self.ir.re[i][j] -1.0
             if s_>=0.0:
                sign = 1.0
             else:
                sign = -1.0
             self.BondDistrubed.append(bd)
             atoms,groupi,groupj = self.set_bond_momenta(i,j,atoms,sign=sign)
             return atoms,bd,groupi,groupj
      return atoms,None,[],[]


  def check_momenta(self,atoms,freeatoms=None):
      v = atoms.get_velocities()

      try:
         f = atoms.get_forces()
         haveforce = True
      except:
         haveforce = False
      if haveforce:
         for i in freeatoms:
             for j in freeatoms:
                 if (i,j) in self.InitBonds:
                    vij = atoms.positions[j] - atoms.positions[i]
                    rij = np.sqrt(np.sum(np.square(vij)))
                    vij = vij/rij
 
                    vi   = np.dot(v[i],vij)
                    vj   = np.dot(v[j],vij)
 
                    fi   = np.dot(f[i],vij)
                    fj   = np.dot(f[j],vij)

                    if abs(vi)<0.003 and abs(fi)<0.5:
                       if vi>=0.0:
                          v[i] = v[i] + 0.05*vij
                       else:
                          v[i] = v[i] - 0.05*vij
      if not v is None:
         atoms.set_velocities(v)
      return atoms


  def set_bond_momenta(self,i,j,atoms,sign=1.0,add=False):
      ha      = int(0.5*self.natom)
      # x     = atoms.get_positions()
      v       = np.zeros([self.natom,3])

      group_j = []
      group_j,ring = getAtomsToMove(i,j,j,group_j,self.neighbors)
      jg      = len(group_j)

      group_i = []
      group_i,ring = getAtomsToMove(j,i,i,group_i,self.neighbors)
      ig      = len(group_i)

      if ring:
         group_i,group_j = get_group(i,j,atoms)

      vij   = self.ir.vr[j][i]/self.ir.r[i][j]
      massi = 0.0
      massj = 0.0

      for a in group_i:
          massi += self.mass[a] 
      for a in group_j:
          massj += self.mass[a] 

      vi  = 1.0/massi
      vj  = 1.0/massj

      for a in group_i:
          if add:
             v[a] = v[a] + sign*vi*vij
          else:
             v[a] = sign*vi*vij

      for a in group_j:
          if add:
             v[a] = v[a] - sign*vj*vij
          else:
             v[a] = -sign*vj*vij

      atoms.set_velocities(v)
      return atoms,group_i,group_j
      

  def zmat_relax(self,atoms,nbin=10,relax_step=None,
                 zmat_variable=None,zvlo=None,zvhi=None,
                 traj='zmat.traj',relaxlog = '',reset=False):
      atoms_ = atoms.copy()
      zmatrix  = np.array(self.get_zmatrix(atoms))
      initz    = np.array(self.InitZmat)         # target
      relax_step_ = nbin if relax_step is None else relax_step
      if not zmat_variable is None:
         i,j           = zmat_variable
         initz         = zmatrix.copy()
         initz[i][j]   = initz[i][j] + zvhi     # target
         zmatrix[i][j] = zmatrix[i][j] + zvlo   # starting point: current configration
         relaxlog += '                     ---------------------------\n'
         relaxlog += '                         %d-%d-%d-%d (%d,%d) \n' %(self.zmat_id[i],self.zmat_index[i][0],
                                                                   self.zmat_index[i][1],self.zmat_index[i][2],i,j)
         relaxlog += '                     ---------------------------\n'
         relaxlog += 'varied from: %8.4f to %8.4f,\n' %(zmatrix[i][j],initz[i][j])
      else:
         relaxlog += 'relax the structure to %d/%d of the initial value ...\n' %(relax_step_,nbin)
      dz_  = (initz - zmatrix)
      dz_     = np.where(dz_>180.0,dz_-360.0,dz_)
      dz      = np.where(dz_<-180.0,dz_+360.0,dz_)
      dz      = dz/nbin

      his     = TrajectoryWriter(traj,mode='w')
      images  = []
      nb_ = 0
      for i_ in range(relax_step_):
          zmat_ = zmatrix+dz*(i_+1)
          zmat_ = np.where(zmat_>180.0,zmat_-360.0,zmat_)           #  scale to a reasonalbale range
          zmat_ = np.where(zmat_==-180.0,zmat_+0.000001,zmat_)
          zmat_ = np.where(zmat_==180.0,zmat_-0.000001,zmat_)
          zmat_ = np.where(zmat_<-180.0,zmat_+360.0,zmat_)          #  scale to a reasonalbale range

          atoms = self.zmat_to_cartation(atoms,zmat_)
          self.ir.calculate(atoms)
          calc = SinglePointCalculator(atoms,energy=self.ir.E)
          atoms.set_calculator(calc)
          his.write(atoms=atoms)
          images.append(atoms)
          bonds   = getBonds(self.natom,self.ir.r,self.rmax*self.ir.re,self.ir.bo0)
          newbond,nbd = self.checkBond(bonds)
          if newbond: 
             iatom,jatom = nbd
             if nb_ == 0:
                nbd_ = nbd
                r_   = self.ir.r[iatom][jatom]
             else:
                if nbd==nbd_:
                   if self.ir.r[iatom][jatom]<r_:
                      relaxlog += 'stop at %d/%d  because new bond formed ...\n' %(i_,nbin)
                      break
                   r_ = self.ir.r[iatom][jatom]
                else:
                   relaxlog += 'stop at %d/%d  because new bond formed ...\n' %(i_,nbin)
                   break
             nb_ += 1
      his.close()
      if reset: images.append(atoms_)
      return images,relaxlog


  def get_optimal_zv(self,atoms,zmat_variable,optgen=None,nbin=100,maxiter=100,
                       plot_opt=False):
      i,j           = zmat_variable
      atoms_        = atoms.copy()
      if j == 0:
         zvlo,zvhi = -0.1,0.1
      else:
         zvlo,zvhi = -3.0,3.0

      bin_          = (zvhi-zvlo)/nbin
      zmatrix       = self.InitZmat # np.array(self.get_zmatrix(atoms_))
      initz         = zmatrix.copy()
      initz[i][j]   = zmatrix[i][j] + zvlo
      zmatrix[i][j] = zmatrix[i][j] + zvhi
      
      zmat_l        = initz
      zmat_r        = zmatrix
      dEdz          = 1.0
      E             = []
      dz            = (zmatrix-initz)/nbin
      optlog        = 'Getting the optimal zmatrix variable value ... \n'
      iter_         = 0

      while abs(dEdz)>0.001 and iter_<maxiter:
            dEdz_l = self.get_dz(atoms_,zmat_l,dz,bin_)
            dEdz_r = self.get_dz(atoms_,zmat_r,dz,bin_)
            
            dx     = (zmat_r - zmat_l)/3.0
            half_  = zmat_l+(zmat_r - zmat_l)/2
            dEdz   = self.get_dz(atoms_,half_,dz,bin_)
            E.append(self.ir.E)
            #print('-  Iter %3d: ' %iter_,'%6.4f %8.4f %8.4f %8.4f' %(half_[i][j],dEdz_l,dEdz,dEdz_r))
            if dEdz_l<=0.0 and dEdz_r<=0.0:
               half_ = zmat_r 
               zmat_r= zmat_r + dx
               # dEdz = 0.000001
            elif dEdz_l<=0 and dEdz_r>=0.0:
               if dEdz<=0:
                  zmat_l = half_
               elif dEdz>=0:
                  zmat_r = half_
            elif dEdz_l>=0 and dEdz_r<=0:
               dEdz = 0.000001          # no minima point in this reigen
               iter_= maxiter           # 
            elif dEdz_l>=0.0 and dEdz_r>0:
               half_ = zmat_l 
               zmat_l= zmat_l - dx
            else:
               dEdz = 0.000001          # no minima point in this reigen
               iter_= maxiter           # 
            iter_ += 1
      result = half_[i][j]
      if iter_<maxiter:
         optlog += '                     ---------------------------\n'
         optlog += '                         %d-%d-%d-%d (%d,%d) \n' %(self.zmat_id[i],
                          self.zmat_index[i][0],self.zmat_index[i][1],self.zmat_index[i][2],i,j)
         optlog += '                     ---------------------------\n'
         optlog += 'The optimal variable value of (%d,%d) is %f ...\n' %(i,j,half_[i][j])
         i_ = self.zmat_id[i]
         j_ = self.zmat_index[i][0]
         if j==0:
            if half_[i][j]>self.ir.re[i_][j_]*1.15 or half_[i][j]<self.ir.re[i_][j_]*0.8:
               half_[i][j] = self.ir.re[i_][j_]
               optlog += 'The optimal of (%d,%d) exceed limmit, reset to preset value ...\n' %(i,j)
         elif j==1:
            nn = len(self.neighbors[j_])
            if nn==3 or nn==4:
               if nn==3:
                  ang_ = 120.0
               elif nn==4:
                  ang_ = 109.0
               da_ = abs(half_[i][j]-ang_)
               if da_ >5.0 :
                  half_[i][j] = ang_
                  optlog += 'The optimal of (%d,%d) exceed limmit, reset to preset value ...\n' %(i,j)
         self.InitZmat = half_
         self.write_zmat(half_,zfile='optimal.zmat')
         atoms_ = self.zmat_to_cartation(atoms_,half_)
         if optgen is not None: atoms_.write(optgen)
         del atoms_
      else:
         optlog += 'The optimal variable search failed! \n'
         # print(zmat_variable,zvlo,zvhi)

      if plot_opt:
         plt.figure()   
         plt.ylabel('Energy (eV)')
         plt.xlabel('Step')
         plt.plot(E,alpha=0.8,
                  linestyle='-',marker='s',markerfacecolor='none',
                  markeredgewidth=1,markeredgecolor='r',markersize=4,
                  color='red',label='Energy')
         plt.savefig('ZVOpt.svg',transparent=True) 
         plt.close() 
      return optlog,result


  def get_dz(self,atoms,zmat,dz,bin_):
      atoms_ = atoms.copy()
      atoms_ = self.zmat_to_cartation(atoms_,zmat)
      self.ir.calculate(atoms_)
      e = self.ir.E

      atoms_ = atoms.copy()
      atoms_ = self.zmat_to_cartation(atoms_,zmat+dz)
      self.ir.calculate(atoms_)
      e_ = self.ir.E
      dEdz   = (e_-e)/bin_
      return dEdz
      
  def get_zmat_uncertainty(self,atoms):
      for i,jj in enumerate(self.zmat_index):
         if jj[0]>=0:
            iatom = self.zmat_id[i]
            jatom = jj[0]
            katom = jj[1]
            latom = jj[2]
            log,v = self.get_optimal_zv(atoms,(i,0))
            if v>self.InitZmat[i][0]*1.15:   
               # print('bond {:3d} {:2s} - {:3d} {:2s}:'.format(iatom,
               #       self.atom_name[iatom],jatom,self.atom_name[jatom]),v)
               return (i,0),self.InitZmat[i][0]*0.9,self.InitZmat[i][0]*1.2
            elif v<self.ir.re[iatom][jatom]*0.9:
               return (i,0),self.InitZmat[i][0]*0.8,self.InitZmat[i][0]*1.10
      return None,0.0,0.0

  def get_zmat_info(self,zmats):
      if zmats is None or len(zmats)==0:
         return None,0.0,0.0
      zmats = np.array(zmats)
      mean_zmat = np.mean(zmats,0)
      dz = mean_zmat - self.InitZmat
      dz_ = np.zeros(self.InitZmat.shape)
      for zmat in zmats:
          dz   = zmat - mean_zmat
          dz   = np.where(dz>180.0,dz-360.0,dz)
          dz   = np.where(dz<-180.0,dz+360.0,dz)
          dz_ += np.square(dz)
      dz_ = np.sqrt(dz_/self.natom)
      lo,hi = 0.0,0.0
      for i,z in enumerate(mean_zmat):
          for j in range(2):
              dr_ = self.InitZmat[i][j] - z[j] 
              dr  = abs(dr_)
              hi  = dr_
              if j==0:
                 if dz_[i][j]<=0.01 and dr>=0.1:
                    return (i,j),lo,hi
              elif j==1:
                 if dz_[i][j]<=1.0 and dr>=10.0:
                    return (i,j),lo,hi 
              elif j==2:
                 if dz_[i][j]<=1.0 and dr>=10.0:
                    return (i,j),lo,hi
      return None,0.0,0.0


  def get_zmats(self,mdtraj='md.traj'):
      images = Trajectory(mdtraj)
      mdzmat = []
      for atoms_ in images:
          zmatrix = self.get_zmatrix(atoms_)
          mdzmat.append(zmatrix)
      mdzmat = np.array(mdzmat)
      return mdzmat


  def check_bond(self,atoms=None,mdtraj=None,rmax=1.3):
      if atoms is None:
         atoms = self.ir.atoms
      if not rmax is None:
         self.rmax = rmax
      self.ir.calculate_Delta(atoms,updateP=True)

      bkbd       = None
      bB_        = 0
      bondBroken = False
      rmax_  = self.rmax - 0.015
      bonds      = getBonds(self.natom,self.ir.r,rmax_*self.ir.re,
                            self.ir.bo0,botol=self.botol*0.5)
      
      if len(bonds) >= len(self.InitBonds):
         for bd in self.InitBonds:
             bd_ = (bd[1],bd[0])
             if (bd not in bonds) and (bd_ not in bonds):
                bkbd = bd
                bondBroken = True
                break
      else:
         bondBroken = True
         for bd in self.InitBonds:
             bd_ = (bd[1],bd[0])
             if (bd not in bonds) and (bd_ not in bonds):
                bkbd = bd
                break
      if bondBroken:
         bB_ += 1

      bondBroken = False
      rmax_  = self.rmax  
      bonds      = getBonds(self.natom,self.ir.r,rmax_*self.ir.re,
                            self.ir.bo0,botol=self.botol)
      if len(bonds) >= len(self.InitBonds):
         for bd in self.InitBonds:
             bd_ = (bd[1],bd[0])
      else:
         bondBroken = True
         for bd in self.InitBonds:
             bd_ = (bd[1],bd[0])
      if bondBroken:
         bB_ += 1
      return bB_,bkbd


  def checkBond(self,bonds):
      newbond = False
      bd      = None
      for bd in bonds:
          bd_ = (bd[1],bd[0])
          if (bd not in self.InitBonds) and (bd_ not in self.InitBonds):
             newbond = True
             return newbond,bd 
      return newbond,bd


  def check(self,wcheck=2,i=0,atoms=None,rmin=None):
      if atoms is None:
         atoms = self.ir.atoms
      if not rmin is None:
         self.rmin = rmin

      self.ir.calculate_Delta(atoms,updateP=True)

      fc = open('check.log','w')
      if i%wcheck==0:
         atoms = self.checkLoneAtoms(atoms,fc)
      else:
         atoms = self.checkLoneAtom(atoms,fc)

      atoms = self.checkClosedAtom(atoms,fc)
      fc.close()
      return atoms


  def checkLoneAtom(self,atoms,fc):
      for i in range(self.natom):
          if self.ir.Delta[i]<=self.ir.atol:
             print('- find an lone atom',i,self.atom_name[i],file=fc)
             sid = np.argsort(self.ir.r[i])
             for j in sid:
                 if self.ir.r[i][j]>0.0001:
                    print('  move lone atom to nearest neighbor: %d' %j,file=fc)
                    vr = self.ir.vr[i][j]
                    u = vr/np.sqrt(np.sum(np.square(vr)))
                    atoms.positions[i] = atoms.positions[j] + u*0.64*self.ir.r_cuta[i][j]
                    break
             self.ir.calculate_Delta(atoms)
      return atoms


  def checkLoneAtoms(self,atoms,fc):
      for i in range(self.natom):
          if self.ir.Delta[i]<=self.ir.atol:
             print('- find an lone atom',i,self.atom_name[i],file=fc)
             mid = np.argmin(self.ir.ND)
             
             if mid == i:
                continue

             print('- find the most atractive atom:',mid,file=fc)
             print('\n- neighors of atom %d %s:' %(i,self.atom_name[i]),end='',file=fc)
             neighs = []
             for j,bo in enumerate(self.ir.bo0[mid]):
                 if bo>self.ir.botol:
                    neighs.append(j)         
                    print(j,self.atom_name[j],end='',file=fc)
             print(' ',file=fc)

             if len(neighs)==0:
                vr = self.ir.vr[mid][i]
                u = vr/np.sqrt(np.sum(np.square(vr)))
                atoms.positions[i] = atoms.positions[mid] + u*0.64*self.ir.r_cuta[i][mid]
             elif len(neighs)==1:
                j = neighs[0]
                vr = self.ir.vr[mid][j]
                u = vr/np.sqrt(np.sum(np.square(vr)))
                atoms.positions[i] = atoms.positions[mid] + u*0.64*self.ir.r_cuta[i][mid]
             elif len(neighs)==2:
                i_,j_ = neighs
                xj = atoms.positions[mid]
                xi = 0.5*(atoms.positions[i_]+atoms.positions[j_])
                vr = xj - xi
                u = vr/np.sqrt(np.sum(np.square(vr)))
                vij = atoms.positions[j_]-atoms.positions[i_]
                rij = np.sqrt(np.sum(np.square(vij)))
                r_  = np.dot(vij,u)
                if r_!=rij:
                   atoms.positions[i] = atoms.positions[mid] + u*0.64*self.ir.r_cuta[i][mid]
             elif len(neighs)==3:
                i_,j_,k_ = neighs
                vi = atoms.positions[i_] - atoms.positions[j_]
                vj = atoms.positions[i_] - atoms.positions[k_]
                # cross product
                vr = np.cross(vi,vj)
                c  = (atoms.positions[i_]+atoms.positions[j_]+atoms.positions[k_])/3
                v  = atoms.positions[mid] - c
                u = vr/np.sqrt(np.sum(np.square(vr)))
                # dot product
                dot = np.dot(v,u)
                if dot<=0:
                   u = -u
                atoms.positions[i] = atoms.positions[mid] + u*0.64*self.ir.r_cuta[i][mid]

             self.ir.calculate_Delta(atoms)
      return atoms

  def checkClosedAtom(self,atoms,fc):
      self.ir.calculate_Delta(atoms)
      neighbors = getNeighbor(self.natom,self.ir.r,self.ir.r_cuta,self.ir.bo0)
      for i in range(self.natom-1):
          for j in range(i+1,self.natom):
              if self.ir.r[i][j]<self.rmin*self.ir.r_cuta[i][j]:
                 print('- atoms %d and %d too closed' %(i,j),file=fc)

                 moveDirection = self.ir.vr[j][i]/self.ir.r[i][j]
                 moveD         = self.ir.r_cuta[i][j]*(self.rmin+0.01) - self.ir.r[i][j]
                 moveV         = moveD*moveDirection
                                                               
                 ToBeMove = []
                 ToBeMove,ring = getAtomsToMove(i,j,j,ToBeMove,neighbors)
                 print('  atoms to to be moved:',ToBeMove,file=fc)
                 for m in ToBeMove:
                     newPos = atoms.positions[m] + moveV
                     r = np.sqrt(np.sum(np.square(newPos-atoms.positions[i])))
                     if r>self.ir.r[i][m]:
                        atoms.positions[m] = newPos
                 self.ir.calculate_Delta(atoms)
                 neighbors = getNeighbor(self.natom,self.ir.r,self.ir.r_cuta,self.ir.bo0)
      return atoms

  def bend(self,ang=None,rang=20.0,nbin=10,scale=1.2,traj='md.traj'):
      i,j,k = ang
      axis = [i,k]
      images = self.rotate(atms=[i,k],axis=axis,o=j,rang=rang,nbin=nbin,traj=traj,scale=scale)
      return images


  def bend_axis(self,axis=None,group=None,rang=20,nbin=30,scale=1.2,traj=None):
      images = self.rotate(atms=group,axis=axis,o=axis[0],rang=rang,nbin=nbin,traj=traj,scale=scale)
      return images

  def swing_group(self,ang=None,group=None,rang=20.0,nbin=30,scale=1.2,
                  neighbors=None,traj=None):
      i,j,k = ang
      atoms = self.ir.atoms
      self.ir.calculate_Delta(atoms)

      vij = atoms.positions[i] - atoms.positions[j] 
      vjk = atoms.positions[k] - atoms.positions[j]
      r   = self.ir.r[j][k]
      ujk = vjk/r
      ui  = vij/self.ir.r[i][j]
      uk  = np.cross(ui,ujk)
      rk  = np.sqrt(np.sum(uk*uk))
      
      if rk<0.0000001:
         uk = np.array([1.0,0.0,0.0])
      else:
         uk  = uk/rk   

      if group is None:
         if neighbors is None:
            # self.ir.calculate_Delta(atoms)
            neighbors = getNeighbor(self.natom,self.ir.r,scale*self.ir.re,self.ir.bo0)
         if group is None:
            group      = []
            group,ring = getAtomsToMove(j,k,k,group,neighbors)
         else:
            ring       = False
         if ring:
            group = [k]

      images = self.rotate(atms=group,axis_vector=uk,o=j,rang=rang,
                           nbin=nbin,traj=traj,scale=scale)
      return images

  def rotate(self,atms=None,axis=None,axis_vector=None,o=None,rang=20.0,
             nbin=10,traj=None,scale=1.2):
      da = 2.0*rang/nbin
      atoms = self.ir.atoms
      self.ir.calculate_Delta(atoms)
      # neighbors = getNeighbor(self.natom,self.ir.r,scale*self.ir.re,self.ir.bo0)

      images = []
      if not traj is None: his = TrajectoryWriter(traj,mode='w')

      if axis_vector is None:
         i,j   = axis
         vaxis = atoms.positions[j] - atoms.positions[i] 
         uk    = vaxis/self.ir.r[i][j]
      else:
         uk    = axis_vector

      a_       =  -rang
      for i in range(nbin+1):
          atoms_ = atoms.copy()
          for atomk in atms:
              vo  = atoms.positions[atomk] - atoms.positions[o] 
              r_  = np.dot(vo,uk)

              o_  = atoms.positions[o] + r_*uk
              vi  = atoms.positions[atomk] - o_

              r   = np.sqrt(np.sum(np.square(vi)))
              ui  = vi/r
              uj  = np.cross(uk,ui)

              a   = a_*3.141593/180.0
              p   = r*np.cos(a)*ui + r*np.sin(a)*uj

              atoms_.positions[atomk] = o_ + p
              self.ir.calculate(atoms_)

              calc = SinglePointCalculator(atoms_,energy=self.ir.E)
              atoms_.set_calculator(calc)

          images.append(atoms_)
          if not traj is None: his.write(atoms=atoms_)
          a_ += da

      if not traj is None: his.close()
      return images

  def swing(self,ang,st=60.0,ed=180.0,nbin=50,scale=1.2,traj=None):
      da    = (ed - st)/nbin
      i,j,k = ang
      atoms = self.ir.atoms
      self.ir.calculate_Delta(atoms)
      # neighbors = getNeighbor(self.natom,self.ir.r,scale*self.ir.re,self.ir.bo0)
      images = []
      if not traj is None: his = TrajectoryWriter(traj,mode='w')

      vij = atoms.positions[i] - atoms.positions[j] 
      vjk = atoms.positions[k] - atoms.positions[j]
      r   = self.ir.r[j][k]
      ujk = vjk/r
      ui  = vij/self.ir.r[i][j]
      uk  = np.cross(ui,ujk)
      rk  = np.sqrt(np.sum(uk*uk))
      
      if rk<0.0000001:
         uk = np.array([1.0,0.0,0.0])
      else:
         uk  = uk/rk

      uj = np.cross(uk,ui)
      a_ = st

      for i in range(nbin+1):
          atoms_ = atoms.copy()
          a = a_*3.141593/180.0
          p = r*np.cos(a)*ui + r*np.sin(a)*uj
          atoms_.positions[k] = atoms_.positions[j]+p
          self.ir.calculate(atoms_)

          calc = SinglePointCalculator(atoms_,energy=self.ir.E)
          atoms_.set_calculator(calc)

          images.append(atoms_)
          if not traj is None: his.write(atoms=atoms_)
          a_ += da

      if not traj is None: his.close()
      return images

  def continous(self,atoms,dr=0.001,da=1.0,nbin=10,traj='md.traj'):
      ''' check the continous of the PES of the current configuration '''
      de = np.zeros((self.natom,3,2))
      # images = []
      zmatrix = np.array(get_zmatrix(atoms,self.zmat_id,self.zmat_index))
      self.ir.calculate(atoms)
      e       = self.ir.E
      for i_,i in enumerate(self.zmat_id):
          for j_,j in enumerate(self.zmat_index[i_]):
              if j!=-1:
                 if j_ == 0 :
                    d_ = dr 
                 else: 
                    d_ = da
                 for k_,k in enumerate([-1.0,1.0]):
                     d = d_*k
                     atoms_ = atoms.copy()
                     zmat_  = zmatrix.copy()
                     zmat_[i_][j_] = zmatrix[i_][j_] + d
                     # print(i_,j_,d,zmat_[i_][j_],zmatrix[i_][j_])
                     
                     atoms_ = self.zmat_to_cartation(atoms_,zmat_)
                     self.ir.calculate(atoms_)

                     e_ = self.ir.E
                     dE = e_-e
                     de[i_][j_][k_] = dE

                     calc = SinglePointCalculator(atoms_,energy=self.ir.E)
                     atoms_.set_calculator(calc)
                     # images.append(atoms_)
                     del atoms_
                     del zmat_
      zi = np.argmax(np.abs(de[:,0,0]))
      zj = np.argmax(np.abs(de[:,0,1]))
      if abs(de[zi,0,0]) >abs(de[zj,0,1]):
         zv = (zi,0,)
         l  = 0
      else:
         zv = (zj,0)
         l  = 1
      atomi = self.zmat_id[i]
      atomj = self.zmat_index[i][0]
      zvlo  = - 0.05*zmatrix[zv]
      zvhi  =   0.05*zmatrix[zv]
      # print(zvlo,zvhi)
      relaxlog  = 'MD step is littile than 3, checking the continous of the zmatmix,\n' 
      relaxlog += 'the dE/dr between atoms {:3d} {:3d} is {:6.4f}.\n'.format(atomi,atomj,
                                                                  float(de[zv[0]][zv[1]][l]))
      images,relaxlog = self.zmat_relax(atoms=atoms,zmat_variable=zv,nbin=nbin,
                                        zvlo=zvlo,zvhi=zvhi,traj=traj,
                                        relaxlog=relaxlog)
      return images,relaxlog

  def scan_pes(self,i,n,neighbor,atoms,images,r,nbin=[5],dr=[0.1]):
      if n>=len(neighbor):
         return None
      k = neighbor[n]

      if n>=len(nbin):
         nb_ = nbin[0]
      else:
         nb_ = nbin[n]

      if n>=len(dr):
         dr_ = dr[0]
      else:
         dr_ = dr[n]

      imag_ = self.stretch([i,k],atoms=atoms,nbin=nb_,
                           rst=r[k]-dr_,
                           red=r[k]+dr_,
                           neighbors=self.neighbors)
      for atoms_ in imag_:
          imag = self.scan_pes(i,n+1,neighbor,atoms_,images,r,nbin=nbin,dr=dr)
          if imag is None:
             images.append(atoms_)
      return images

  def pes(self,i,atoms,neighbor=None,nbin=[5],scandr=[0.2],traj=None):
      images = []
      r      = {}

      if neighbor is None:
         neighbor = self.neighbors[i]

      for j_ in neighbor:
          r[j_]   = self.ir.r[i][j_]  
      images = self.scan_pes(i,0,neighbor,atoms,images,r,nbin=nbin,dr=scandr)

      if not traj is None:
         his = TrajectoryWriter(traj,mode='w')
         for atoms in images:
             his.write(atoms=atoms)
         his.close()
       
      return images
  

  def stretch(self,pair,atoms=None,nbin=20,rst=0.7,red=1.3,scale=1.25,traj=None,
              ToBeMoved=None,neighbors=None):
      if atoms is None:
         atoms = self.ir.atoms

      if neighbors is None:
         self.ir.calculate_Delta(atoms)
         neighbors = getNeighbor(self.natom,self.ir.r,scale*self.ir.re,self.ir.bo0)
      images = []
 
      if not traj is None: his = TrajectoryWriter(traj,mode='w')
      #for pair in pairs:
      i,j = pair
      if ToBeMoved is None:
         ToBeMove = []
         ToBeMove,ring = getAtomsToMove(i,j,j,ToBeMove,neighbors)
      else:
         ToBeMove = ToBeMoved
         ring     = False
      
      if ring:
         # return None
         # ToBeMove = [j]
         group_i,group_j = get_group(i,j,atoms)
         ToBeMove = group_j

      bin_     = (red - rst)/nbin
      moveDirection = self.ir.vr[j][i]/self.ir.r[i][j]

      for n in range(nbin+1):
          atoms_ = atoms.copy()
          moveV  = atoms.positions[i] + moveDirection*(rst+bin_*n)-atoms.positions[j]
          # print(self.ir.re[i][j]*self.rmin+bin_*n)
          for m in ToBeMove:
              # sPos   = atoms.positions[i] + self.ir.re[i][m]*self.rmin*moveDirection
              newPos = atoms.positions[m] + moveV
              r = np.sqrt(np.sum(np.square(newPos-atoms.positions[i])))   
              atoms_.positions[m] = newPos

          self.ir.calculate(atoms_)
          i_= np.where(np.logical_and(self.ir.r<self.ir.re[i][j]*self.rmin-bin_,self.ir.r>0.0001))
          n = len(i_[0])

          try:
             assert n==0,'Atoms too closed!'
          except:
             print('Atoms too closed.')
             break
             
          calc = SinglePointCalculator(atoms_,energy=self.ir.E)
          atoms_.set_calculator(calc)
          images.append(atoms_)
          if not traj is None: his.write(atoms=atoms_)
      return images


  def write_zmat(self,zmatrix,zfile=None):
      if zfile is None:
         zfile = self.label+'.zmat'
      with open (zfile,'w') as f:
           for i,iatom in enumerate(self.zmat_id):
               i_ = self.zmat_id[i]
               print('[ \'%s\',%3d, %3d, %3d, %3d,' %(self.atom_name[i_],iatom,self.zmat_index[i][0],
                                              self.zmat_index[i][1],self.zmat_index[i][2]),
                     ' %7.4f,%8.4f,%9.4f ],' %(zmatrix[i][0],zmatrix[i][1],zmatrix[i][2]),
                     file=f)


  def close(self):
      self.ir        = None
      self.atom_name = None

