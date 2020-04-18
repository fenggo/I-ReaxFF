from __future__ import print_function
from os.path import isfile
from irff.irff_np import IRFF_NP
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np



def getAtomsToMove(i,j,ToBeMove,neighbors):
    ToBeMove.append(j)
    for n in neighbors[j]:
        if n!=i:
           if n not in ToBeMove:
              ToBeMove = getAtomsToMove(i,n,ToBeMove,neighbors)
    return ToBeMove
                       
                       
def getNeighbor(natom,r,rcut):
    neighbors = [[] for _ in range(natom)]
    for i in range(natom-1):
        for j in range(i+1,natom):
            if r[i][j]<rcut[i][j]:
               # print(i,j,r[i][j],rcut[i][j])
               neighbors[i].append(j)
               neighbors[j].append(i)      
    return neighbors


class AtomOP(object):
  def __init__(self,atoms=None,rtole=0.5):
      self.rtole  =  rtole
      if atoms is None:
         gen_   = 'md.traj' if isfile('md.traj') else 'poscar.gen'
         atoms  = read(gen_,index=-1)

      self.ir = IRFF_NP(atoms=atoms,
                        libfile='ffield.json',
                        rcut=None,
                        nn=True)

      self.natom     = self.ir.natom
      self.atom_name = self.ir.atom_name
      spec           = self.ir.spec
    
      label_dic      = {}
      for sp in self.atom_name:
          if sp in label_dic:
             label_dic[sp] += 1
          else:
             label_dic[sp]  = 1
      self.label = ''
      for sp in spec:
          self.label += sp+str(label_dic[sp])


  def check(self,wcheck=2,i=0,atoms=None,rtole=None):
      if atoms is None:
         atoms = self.ir.atoms
      if not rtole is None:
         self.rtole = rtole

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
      neighbors = getNeighbor(self.natom,self.ir.r,self.ir.r_cuta)
      for i in range(self.natom-1):
          for j in range(i+1,self.natom):
              if self.ir.r[i][j]<self.rtole*self.ir.r_cuta[i][j]:
                 print('- atoms %d and %d too closed' %(i,j),file=fc)

                 moveDirection = self.ir.vr[j][i]/self.ir.r[i][j]
                 moveD         = self.ir.r_cuta[i][j]*(self.rtole+0.01) - self.ir.r[i][j]
                 moveV         = moveD*moveDirection
                                                               
                 ToBeMove = []
                 ToBeMove = getAtomsToMove(i,j,ToBeMove,neighbors)
                 print('  atoms to to be moved:',ToBeMove,file=fc)
                 for m in ToBeMove:
                     newPos = atoms.positions[m] + moveV
                     r = np.sqrt(np.sum(np.square(newPos-atoms.positions[i])))
                     if r>self.ir.r[i][m]:
                        atoms.positions[m] = newPos
                 self.ir.calculate_Delta(atoms)
                 neighbors = getNeighbor(self.natom,self.ir.r,self.ir.r_cuta)
      return atoms


  def stretch(self,pairs,nbin=20,scale=1.2,wtraj=False):
      atoms = self.ir.atoms
      self.ir.calculate_Delta(atoms)
      neighbors = getNeighbor(self.natom,self.ir.r,scale*self.ir.re)
      images = []
      
      if wtraj: his = TrajectoryWriter('stretch.traj',mode='w')
      for pair in pairs:
          i,j = pair
          ToBeMove = []
          ToBeMove = getAtomsToMove(i,j,ToBeMove,neighbors)

          bin_     = (self.ir.r_cuta[i][j] - self.ir.re[i][j]*self.rtole)/nbin
          moveDirection = self.ir.vr[j][i]/self.ir.r[i][j]

          for n in range(nbin):
              atoms_ = atoms.copy()
              moveV  = atoms.positions[i] + moveDirection*(self.ir.re[i][j]*self.rtole+bin_*n)-atoms.positions[j]
              # print(self.ir.re[i][j]*self.rtole+bin_*n)
              for m in ToBeMove:
                  # sPos   = atoms.positions[i] + self.ir.re[i][m]*self.rtole*moveDirection
                  newPos = atoms.positions[m] + moveV
                  r = np.sqrt(np.sum(np.square(newPos-atoms.positions[i])))   
                  atoms_.positions[m] = newPos

              self.ir.calculate(atoms_)
              i_= np.where(np.logical_and(self.ir.r<self.ir.re[i][j]*self.rtole-bin_,self.ir.r>0.0001))
              n = len(i_[0])
              try:
                 assert n==0,'Atoms too closed!'
              except:
                 print('Atoms too closed.')
                 break
                 
              calc = SinglePointCalculator(atoms_,energy=self.ir.E)
              atoms_.set_calculator(calc)
              images.append(atoms_)
              if wtraj: his.write(atoms=atoms_)

      if wtraj: his.close()
      return images


  def close(self):
      self.ir        = None
      self.atom_name = None

