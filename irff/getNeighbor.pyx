# cython: language_level=3
from .RadiusCutOff import setRcut
import numpy as np


def find_mole(ID,mol_index,table):
    mol_index.append(ID)
    for nn in table[ID]: 
        if not nn in mol_index:
           mol_index = find_mole(nn,mol_index,table)
    return mol_index


def get_neighbors(filename=None,Atoms=None,r_cut=None,cell=None,exception=[]):
    cdef int i,j
    cdef float hfcell = 0.5
    if r_cut is None:
       rcut,r_cut,rcute = setRcut(None,None,None,None)

    atoms = Atoms.get_chemical_symbols()
    X     = Atoms.get_positions()
    
    cdef int natm  = len(atoms)
    table = [ [] for i in range(natm)]
    
    if cell is None:
       cell = Atoms.get_cell()

    u     = np.linalg.inv(cell)
    x     = np.array(X)         #  project to the fractional coordinate
    xf    = np.dot(x,u) 

    for i in range(natm-1):
        for j in range(i+1,natm):
            xj   = np.dot(x[j],u)
            xi   = np.dot(x[i],u)
            vr   = xj - xi

            for d in range(3):
                if vr[d]-hfcell>0:
                   vr[d] = vr[d]-1.0
                if vr[d]+hfcell<0:
                   vr[d] = vr[d]+1.0

            vr_ = np.dot(vr,cell) # convert to ordinary coordinate
            r   = np.sqrt(np.sum(vr_*vr_))
    
            pair = atoms[i]+'-'+atoms[j]
            pairr= atoms[j]+'-'+atoms[i]
            if pair in r_cut:
               rc = r_cut[pair]  
            elif pairr in r_cut:
               rc = r_cut[pairr] 
            else:
               print('-  warning: rcut not define for pair: %s, using default 1.8.' %pair)
               rc = r_cut['others']

            if r<rc and (not pair in exception):
               table[i].append(j)
               table[j].append(i)
            # print('-  atom pair %d-%d of %d ...\r' %(i,j,natm),end='\r')
    return natm,atoms,X,table

