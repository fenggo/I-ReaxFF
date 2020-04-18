#!/usr/bin/env python
from __future__ import print_function
from .emdk import get_structure
import numpy as np
from ase.io import read,write
from ase import Atoms
from .getNeighbors import get_neighbors,find_mole


def get_mol(mol=None):
    strucs = ['tatb','tatbmol','datb','datbmol','n2co','rdx','rdxmol',
              'nitromethane','hmx-cl20','cl20-hmx','hmx','hmxmol','cl20','cl20mol',
              'ch4','methane','nh3','nmmol','nitroethane','ethane','ethylene','no2',
              'co2','c3h6n2o4','fox7','Si','dopamine1','dopamine','h2o']
    dimers = ['OO','CC','HH','NN',
              'CO','OC','HO','OH','ON','NO',
              'CH','HC','NH','HN','CN','NC' ]
    # elems = ['C','H','O','N']
    m = mol.split('-')

    if mol in strucs:
       get_structure(struc=mol,output='dftb',recover=True,center=True)
       A = read('card.gen')
    elif mol in dimers:
       if mol[0]=='H' and mol[1]=='H':
          r = 0.8
       elif mol[0]=='H' or mol[1]=='H':
          r = 1.0
       else:
          r = 1.2
       A= Atoms([mol[0],mol[1]],
                [[0.0,0.0,1.0],[r,0.0,1.0]],
                cell=[3.0,3.0,3.0])
    elif len(m)==3:                 # three elements
       if m[0]=='H' and m[1]=='H':
          r1 = 0.8
       elif m[0]=='H' or m[1]=='H':
          r1 = 1.0
       else:
          r1 = 1.2

       if m[1]=='H' and m[2]=='H':
          r2 = 0.8
       elif m[1]=='H' or m[2]=='H':
          r2 = 1.0
       else:
          r2 = 1.2
       A= Atoms([m[0],m[1],m[2]],
                [[r1,0.0,1.0],[0.0,0.0,1.0],[-0.5*r2,r2*1.732*0.5,1.0]],
                cell=[3.0,3.0,3.0])
    else:
       print('- error: %s not in database!' %mol)
    return A


def get_lattice(inp='inp-nve'):
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


def compress(A,comp=[1.0,1.0,1.0]):
    cell = A.get_cell()
    comp = np.reshape(np.array(comp),[3,1])
    cell = comp*cell
    A.set_cell(cell)
    return A


def select_mol(A=None,step=0,index=None,rcut=None,inbox=False):
    if A is None:
       A   = read('GEOMETRY.xyz')
       cel = get_lattice(inp='inp-nve')
       A.set_pbc([True,True,True])
       A.set_cell(cel)
    cel = A.get_cell()
    box = np.array([cel[0][0],cel[1][1],cel[2][2]])
    natm,atoms,X,table = get_neighbors(Atoms=A,r_cut=rcut,cell=box) #,exception=['O-O','H-H']

    M = molecules(natm,atoms,X,
                  cell=box,
                  table=table,
                  check=True,inbox=inbox)

    elems,x = [],[]
    ind     = []
    # flog    = open('log.txt','a')
    for mol in M:
        jiao = set(mol.mol_index) & set(index)
        if jiao:
           elems.extend(mol.atom_name)
           x.extend(mol.mol_x)
           ind.extend(mol.mol_index)
           # print(step,mol.mol_index,file=flog)
           # print(step,mol.atom_name,file=flog)
    # flog.close()
    A = Atoms(elems,x,cell=cel,charges=ind,tags=ind)
    return A


def packmol(strucs=[],supercell=[1,1,1],w=False,sizeiscell=True):
    mols = []
    for s in strucs:
        if s.find('.gen')>=0:
           A = read(s)
           atoms_label = A.get_chemical_symbols()
           pos = A.get_positions()
           cell= A.get_cell()
           mols.append(molecule(None,atoms_label,pos,
                                cell=cell,sizeiscell=sizeiscell))
        else:
           A = get_mol(s)
           atoms_label = A.get_chemical_symbols()
           pos = A.get_positions()
           mols.append(molecule(None,atoms_label,pos))

    cx,cy,cz    = 0.0,0.0,0.0
    cxo,cyo,czo = 0.0,0.0,0.0
    i,j,k       = 0,0,0
    n_mol       = 0
     
    cym = []
    czm = []

    n = len(strucs)-1
    x,elems = [],[]

    for i in range(supercell[0]):
        if n_mol>n: break 
        cx += mols[n_mol].size[0]
        cy  = 0.0
        cyo = 0.0
        for j in range(supercell[1]):
            if n_mol>n: break 
            cy += mols[n_mol].size[1]
            cz  = 0.0
            czo = 0.0
            for k in range(supercell[2]):
                if n_mol>n: break 
                cz += mols[n_mol].size[2]
              
                if czo + mols[n_mol].size[2]>cz:
                   cz = czo + mols[n_mol].size[2]
                if cyo + mols[n_mol].size[1]>cy:
                   cy = cyo + mols[n_mol].size[1]
                if cxo + mols[n_mol].size[0]>cx:
                   cx = cxo + mols[n_mol].size[0]

                mov =  np.array([cx,cy,cz]) - mols[n_mol].center - 0.5*mols[n_mol].size 
                mols[n_mol].move(mov)

                if n_mol==0:
                   x     = mols[n_mol].mol_x
                   elems = mols[n_mol].atom_name
                else:
                   x     = np.append(x,mols[n_mol].mol_x,axis=0)
                   elems = np.append(elems,mols[n_mol].atom_name)

                n_mol += 1

                czo = cz
            cyo = cy
            czm.append(cz)
        cym.append(cy)
        cxo = cx
 
    B = Atoms(elems, x)
    B.set_cell([[cx,0.0,0.0],
                [0.0,max(cym),0.0],
                [0.0,0.0,max(czm)]])
    B.set_pbc([True,True,True])

    if w:
       B.write('packed.gen')
    return B         


def press_mol(atoms,fac=1.0,inbox=False,check=True):
    cell = atoms.get_cell()
    natm,atoms,X,table = get_neighbors(Atoms=atoms,
                                r_cut=None,
                                cell=cell) #,exception=['O-O','H-H']
    
    M = molecules(natm,atoms,X,
                  cell=cell,
                  table=table,
                  check=check,
                  inbox=inbox,
                  sizeiscell=True)
    M,atoms  = enlarge(M,cell=cell,fac=fac,supercell=[1,1,1])
    return atoms


def SuperCell(m,cell=None,fac=1.0,supercell=None):
    mols,atoms = enlarge(m,cell=cell,fac=fac,supercell=supercell)
    return mols,atoms


def enlarge(m,cell=None,fac=1.0,supercell=None):
    a = cell[0]
    b = cell[1]
    c = cell[2]

    n_m = len(m)

    for mol in m:
        new_center = [x*fac for x in mol.center]
        mol.move(np.subtract(new_center,mol.center))

    natm = 0
    for mol in m:
        natm += mol.natm

    elems,x = [None for na in range(natm)],[None for na in range(natm)]
    for mol in m:
        for i,na in enumerate(mol.mol_index):
            elems[na] = mol.atom_name[i]
            x[na]     = mol.mol_x[i]

    if not supercell is None:           # build supercell
       for i in range(supercell[0]):
           for j in range(supercell[1]):
               for k in range(supercell[2]):
                   if k>0 or j>0 or i>0:
                      for n_mol in range(n_m):
                          ind = [ID+natm*k+ natm*supercell[2]*j+natm*supercell[2]*supercell[1]*i \
                                 for ID in m[n_mol].mol_index]
                          
                          cell_ = cell[0]*i + cell[1]*j + cell[2]*k
                          m_x   = m[n_mol].mol_x + cell_

                          m.append(molecule(ind,m[n_mol].atom_name,m_x))
                          elems.extend(m[n_mol].atom_name)
                          x.extend(m_x)
 
       a = [r*supercell[0]*fac for r in a]
       b = [r*supercell[1]*fac for r in b]
       c = [r*supercell[2]*fac for r in c]
 
    A = Atoms(elems, x)
    A.set_cell([a,b,c])
    A.set_pbc([True,True,True])
    return m,A


def moltoatoms(mols):
    elems = []
    x = []
    for m in mols:
        elems.extend(m.atom_name)
        x.extend(m.mol_x)

    A = Atoms(elems, x)
    A.set_cell(m.cell)
    A.set_pbc([True,True,True])
    return A


def Molecules(atoms,rcut=None,check=False):
    cell = atoms.get_cell()
    natm,atoms,X,table = get_neighbors(Atoms=atoms,
                                       r_cut=rcut,
                                       cell=cell) #,exception=['O-O','H-H']
    m = molecules(natm,atoms,X,
                  cell=cell,
                  table=table,
                  check=check,
                  inbox=False,
                  sizeiscell=True)
    return m


def molecules(natm,atoms,X,cell=None,table=None,
              check=False,
              inbox=False,
              sizeiscell=False):
    m = []
    for ID in range(natm):
        to_search = True
        for n_m in m:
            if ID in n_m.mol_index:
               to_search = False

        if to_search:
           mol_index = []
           mol_index = find_mole(ID,mol_index,table)

           atom_name,mol_x = [],[]
           for i in mol_index:
               atom_name.append(atoms[i])
               mol_x.append(X[i])

           m.append(molecule(mol_index,atom_name,mol_x,
                    check=check,inbox=inbox,
                    cell=cell,table=table,sizeiscell=sizeiscell))
    return m


class molecule(object):
  '''  molecule oprations  '''
  def __init__(self,mol_index,atom_name,mol_x,
               cell=[[10.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]],
               sizeiscell=False,
               table=None,
               check=False,
               inbox=False,
               rcut ={'Fe-C':2.0,'Fe-H':1.8,'Fe-N':2.0,'Fe-O':2.0,'Fe-Fe':2.2,
                       'C-C':1.8,'C-N':1.8,'C-O':1.8,'C-H':1.5,
                       'N-N':1.9,'N-H':1.8,'N-O':1.8,
                       'O-H':1.8,'O-O':1.7,
                       'H-H':1.3,
                       'Cl-C':2.0,'Cl-N':2.0,'Cl-O':2.0,'Cl-H':1.7,'Cl-Fe':2.2,
                       'F-C':1.85,'F-N':1.85,'F-O':1.85,'F-H':1.7,'F-Fe':1.8,'F-Cl':1.85,
                       'other':1.8}):
      self.inbox        = inbox
      self.rcut         = rcut.copy()
      self.mol_x        = np.array(mol_x)
      self.cell         = np.array(cell)
      self.sizeiscell   = sizeiscell

      if mol_index is None:
         self.mol_index = np.arange(len(atom_name))
      else:
         self.mol_index = mol_index

      self.natm         = len(self.mol_index)
      self.atom_name    = atom_name
      self.table        = []
      self.center       = np.sum(self.mol_x,axis=0)/self.natm

      for bd in rcut:    # check rcut
          if bd == 'other':
             continue
          b = bd.split('-')
          bdr = b[1]+'-'+b[0]
          if not bdr in self.rcut:
             self.rcut[bdr]  = self.rcut[bd]

      if (not table is None) and (not mol_index is None):
         for ind in self.mol_index:
             tab = table[ind]
             tab_ = []
             for a in tab:
                 if a in self.mol_index:
                    tab_.append(self.mol_index.index(a))
             self.table.append(tab_)

      self.get_radius()

      # check molecular conectivity
      if check: 
         self.check_mol()

      if not self.sizeiscell:
         self.get_box()

      if inbox:
         self.InBox()


  def InBox(self):
      cf  = np.dot(np.expand_dims(self.center,axis=0),self.u) 
      lm  = np.where(cf-1.0>0)
      lp  = np.where(cf<0)

      while (lm[0].size!=0 or lm[1].size!=0 or lp[0].size!=0 or lp[1].size!=0):
          cf = np.where(cf-1.0>0,cf-1.0,cf)
          cf = np.where(cf<0,cf+1.0,cf)     # apply pbc
          lm = np.where(cf-1.0>0)
          lp = np.where(cf<0)
      new_center = np.dot(cf,self.cell)
      self.move(new_center-self.center)


  def get_radius(self):
      self.u = np.linalg.inv(self.cell)
      x    = np.array(self.mol_x)   #  project to the fractional coordinate
      xf   = np.dot(x,self.u) 

      xj   = np.expand_dims(xf,axis=0)
      xi   = np.expand_dims(xf,axis=1)
      vr   = xj - xi

      if not self.cell is None:
         hfcell = 0.5 

         lm   = np.where(vr-hfcell>0)
         lp   = np.where(vr+hfcell<0)

         while (lm[0].size!=0 or lm[1].size!=0 or lm[2].size!=0 or
                 lp[0].size!=0 or lp[1].size!=0 or lp[2].size!=0):
              vr = np.where(vr-hfcell>0,vr-1.0,vr)
              vr = np.where(vr+hfcell<0,vr+1.0,vr)     # apply pbc
              lm = np.where(vr-hfcell>0)
              lp = np.where(vr+hfcell<0)

      # center = np.squeeze(np.dot([[0.5,0.5,0.5]],cell))
      self.vr= np.dot(vr,self.cell) # convert to ordinary coordinate
      vr2    = np.multiply(vr,vr)
      self.r = np.sqrt(np.sum(vr2,axis=2),dtype=np.float32)


  def get_box(self):
      x = self.mol_x[:,0]  
      y = self.mol_x[:,1]  
      z = self.mol_x[:,2]  

      self.box_min = np.array([np.min(x),np.min(y),np.min(z)])
      self.box_max = np.array([np.max(x),np.max(y),np.max(z)])

      if self.sizeiscell:
         self.size   = np.array([self.cell[0][0],self.cell[1][1],self.cell[2][2]])
      else:
         self.size   = self.box_max-self.box_min+3.0


  def move(self,mov): 
      mov        = np.reshape(mov,[1,3])
      self.mol_x = self.mol_x + mov


  def check_mol(self):
      for iatom,tab in enumerate(self.table):
          for jatom in tab:
              # print(iatom,jatom)
              bd   = self.atom_name[iatom]+'-'+self.atom_name[jatom]
              vr   = self.mol_x[jatom] - self.mol_x[iatom]
              r_   = np.sqrt(np.sum(vr*vr,axis=0))
              if r_>self.rcut[bd]: 
                 self.mol_x[jatom] = self.mol_x[iatom] + self.vr[iatom][jatom]


  def move_atom(self,i,j):
      vr  = self.mol_x[j] - self.mol_x[i]
      # vr  = np.where(vr-hfcell>=0,vr-cell,vr) # pbc
      # vr  = np.where(vr+hfcell<=0,vr+cell,vr) # pbc

      vr2 = np.multiply(vr,vr)
      r   = np.sqrt(np.sum(vr2),dtype=np.float32)
      
      self.mol_x[j] = np.where(vr-self.hfcell>=0,self.mol_x[j]-self.cell,self.mol_x[j])
      self.mol_x[j] = np.where(vr+self.hfcell<=0,self.mol_x[j]+self.cell,self.mol_x[j])
      self.moved[j] = True

      for k in self.table[j]:
          kk = self.mol_index.index(k)
          if not self.moved[kk]:
             self.move_atom(j,kk)



if __name__ == '__main__':
   get_structure(struc='rdx',output='dftb',recover=True,center=True,supercell=[1,1,1])
   A = read('card.gen')
   natm,atoms,X,table = get_neighbors(Atoms=A,r_cut=None,exception=['O-O','H-H'])
   m = molecules(natm,atoms,X,table)
   cell = A.get_cell()





