from ..AtomDance import AtomDance,getNeighbor
# from ..molecule import moltoatoms
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np


def getBonds(natom,r,rcut):
    bonds = [] 
    for i in range(natom-1):
        for j in range(i+1,natom):
            if r[i][j]<rcut[i][j]:
               bonds.append((i,j))
    return bonds


def checkBond(bonds,InitBonds):
    nwbd = []
    bkbd = []
    for bd in bonds:
        bd_ = (bd[1],bd[0])
        if (bd not in InitBonds) and (bd_ not in InitBonds):
           nwbd.append(bd)

    for bd in InitBonds:
        bd_ = (bd[1],bd[0])
        if (bd not in bonds) and (bd_ not in bonds):
           bkbd.append(bd)

    return nwbd,bkbd


def newbond_score(bds,r,re=None,rmax=None,dr=0.1):
    nb = len(bds)
    if nb<=0:
       score = 0.0
       bd    = None
    else:
       scores = np.zeros([nb])
       for i_,bd in enumerate(bds):
           i,j = bd
           scores[i_] += 0.2
           if r[i][j]<(rmax-dr)*re[i][j]:
              scores[i_] += 0.2
           if r[i][j]<(rmax-2.0*dr)*re[i][j]:
              scores[i_] += 0.2
       m     = np.argmax(scores)
       score = scores[m]
       bd    = bds[m]
    return score,bd

def breakbond_score(bds,r,re=None,rmax=None,dr=0.1):
    nb = len(bds)
    if nb<=0:
       score = 0.0
       bd    = None
    else:
       scores = np.zeros([nb])
       for i_,bd in enumerate(bds):
           i,j = bd
           scores[i_] += 0.2
           if r[i][j]>(rmax+dr)*re[i][j]:
              scores[i_] += 0.2
           if r[i][j]>(rmax+2.0*dr)*re[i][j]:
              scores[i_] += 0.2
       m     = np.argmax(scores)
       score = scores[m]
       bd    = bds[m]
    return score,bd

def reaction_capture(traj='md.traj',rmin=0.5,rmax=1.2,angmax=50.0,dr=0.1):
    ''' capture a single chemical reaction '''
    bd,bd_ = None,None
    images = Trajectory(traj)
    ad     = AtomDance(atoms=images[0],rmax=rmax)
    natom  = len(images[0])
    re     = ad.ir.re
    initbonds = ad.InitBonds

    for i,atoms in enumerate(images):
        ad.ir.calculate_Delta(atoms)

        bonds        = getBonds(natom,ad.ir.r,rmax*re)
        bds,bds_     = checkBond(bonds,initbonds)
        nb_score,bd  = newbond_score(bds,ad.ir.r,re=re,rmax=rmax,dr=dr)
        bk_score,bd_ = breakbond_score(bds_,ad.ir.r,re=re,rmax=rmax,dr=dr)

        # print(i,bd,bd_,nb_score,bk_score)
        if nb_score>=0.4 or bk_score>=0.4:
           if i>=50:
              stframe = i-50
           else:
              stframe = 0
           edframe = i
           break
    
    if bd is None and bd_ is None:
       print('-  No reactions have been found!')
       return None
    # print(i,bd,bd_,nb_score,bk_score)
    ind_ = []
    for i,m in enumerate(ad.mols):
        if bd is not None:
           if bd[0] in m.mol_index or bd[1] in m.mol_index:
              for i_ in m.mol_index:
                  if i_ not in ind_:
                     ind_.append(i_)
        if bd_ is not None:
           if bd_[0] in m.mol_index or bd_[1] in m.mol_index:
              for i_ in m.mol_index:
                  if i_ not in ind_:
                     ind_.append(i_)    

    reaction = []
    reaxs    = TrajectoryWriter('reax.traj',mode='w')
    ad_      = None
    cell     = images[0].get_cell()

    for i,atoms in enumerate(images):
        if i>=stframe and i<=edframe:
           at = []
           for a in ind_:
               at.append(atoms[a])

           atoms_ = Atoms(at,cell=cell)
           if ad_ is None:
              ad_ = AtomDance(atoms=atoms_,rmax=rmax)
           
           ad_.ir.calculate(atoms_)
           calc = SinglePointCalculator(atoms_,energy=ad_.ir.E)
           atoms_.set_calculator(calc)

           reaction.append(atoms_)
           reaxs.write(atoms=atoms_)
    reaxs.close()
    return reaction


def zmatrix_matching(zmat_id,zmat_index,atoms):
    ''' finding the matching zmatrix ''' 
    return atoms
    



