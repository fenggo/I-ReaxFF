''' data preparation for a GaussianProcessRegressor
'''
import sys
import argh
import argparse
import numpy as np
import csv
from ase.io import read
from ase.io.trajectory import Trajectory
from irff.irff_np import IRFF_NP
from irff.molecule import Molecules,moltoatoms

# from sklearn.datasets import make_friedman2
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
# X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
# kernel = DotProduct() + WhiteKernel()
# gpr = GaussianProcessRegressor(kernel=kernel,
# ...         random_state=0).fit(X, y)
# gpr.score(X, y)
# 0.3680...
# gpr.predict(X[:2,:], return_std=True)


class DataFrame:
  def __init__(self,columns=None,dat=None):
      self.columns = columns
      if dat is not None:
         if len(dat.shape)==1:
            self.values = np.expand_dims(dat,axis=0)
         else:
            self.values   = dat
         self.shape   = self.values.shape
         self.par_dic = {col:i for i,col in enumerate(columns)}
      else:
         self.values = None

  def read_csv(self,fcsv,columns=None):
      self.values = np.loadtxt(fcsv, delimiter=',', skiprows=1) 
      if len(self.values.shape)==1:
         self.values = np.expand_dims(self.values,axis=0)
      if columns is None:
         with open(fcsv,'r') as f:
              line = f.readline()
              columns = line.split(',')
      self.columns = columns

  def append(self,new_row):
      new = [self.values.shape[0]]
      for col in self.columns:
          if not col:
             continue
          new.append(new_row[col])
      # print(new)
      new = np.array([new])
      self.values = np.concatenate((self.values, new), axis=0)
      # return self.values
      self.shape = self.values.shape

  def loc(self,i=None,key=None):
      if i is not None and key is None:
         return self.values[i]
      elif i is None and key is not None:
         return self.values[:,self.par_dic[key]]
      else:
         return self.values[i,self.par_dic[key]]

  def set_value(self,i,key,v):
      self.values[i,self.par_dic[key]] = v

  def sort_values(self):
      score = self.values[:,-1]
      ind   = np.argsort(-score)
      self.values = self.values[ind]

  def drop(self,i=-1):
      ''' drop the last row '''
      self.values = np.delete(self.values,i,axis=0)

  def save(self,fcsv):
      ''' save data to csv '''
      fcsv = open(fcsv,'w')
      csv_write = csv.writer(fcsv)
      csv_write.writerow(self.columns)
     
      for i,d_ in enumerate(self.values):
          d_[0] = i
          csv_write.writerow(d_)
      fcsv.close()


def read_csv(columns,fcsv):
    dat = np.loadtxt(fcsv, delimiter=',', skiprows=1) 
    df  = DataFrame(columns,dat)
    return df

def get_data(dataset={'nm-0': 'nm-0.traj'}, bonds=['C-C'],
             message_function=2,ffield='ffield.json'):
    D, Y = {}, {}
    R, Bp,B = {}, {}, {}

    for bd in bonds:
        D[bd] = []
        Y[bd] = []
        Bp[bd] = []
        B[bd] = []
        R[bd] = []
        
    for traj in dataset:
        images = Trajectory(dataset[traj])

        ir = IRFF_NP(atoms=images[0],
                     libfile=ffield,
                     nn=True, vdwnn=False)

        for i, atoms in enumerate(images):
            energy = atoms.get_potential_energy()
            ir.calculate_Delta(atoms)

            for ii in range(ir.natom-1):
                for jj in range(ii+1, ir.natom):
                    if ir.bop[ii][jj] > 0.0001:
                        bd = ir.atom_name[ii] + '-' + ir.atom_name[jj]
                        bdr = ir.atom_name[jj] + '-' + ir.atom_name[ii]

                        if bd in bonds:
                           if message_function==1:
                              D[bd].append([ir.D_si[0][ii]-ir.bop_si[ii][jj], 
                                            ir.D_pi[0][ii]-ir.bop_pi[ii][jj], 
                                            ir.D_pp[0][ii]-ir.bop_pp[ii][jj], 
                                            ir.bop[ii][jj], 
                                            ir.D_pp[0][jj]-ir.bop_pp[ii][jj], 
                                            ir.D_pi[0][jj]-ir.bop_pi[ii][jj], 
                                            ir.D_si[0][jj]-ir.bop_si[ii][jj] ])
                           else:
                              D[bd].append([ir.Deltap[ii]-ir.bop[ii][jj], ir.bop[ii][jj], 
                                            ir.Deltap[jj]-ir.bop[ii][jj]])
                           Bp[bd].append([ir.bop_si[ii][jj], ir.bop_pi[ii][jj], ir.bop_pp[ii][jj]])
                           B[bd].append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])
                           R[bd].append(ir.r[ii][jj])
                           Y[bd].append(ir.esi[ii][jj])
                        elif bdr in bonds:
                           if message_function==1:
                              D[bdr].append([ir.D_si[0][jj]-ir.bop_si[ii][jj], 
                                            ir.D_pi[0][jj]-ir.bop_pi[ii][jj], 
                                            ir.D_pp[0][jj]-ir.bop_pp[ii][jj], 
                                            ir.bop[ii][jj], 
                                            ir.D_pp[0][ii]-ir.bop_pp[ii][jj], 
                                            ir.D_pi[0][ii]-ir.bop_pi[ii][jj], 
                                            ir.D_si[0][ii]-ir.bop_si[ii][jj] ])
                           else:
                              D[bdr].append([ir.Deltap[jj]-ir.bop[ii][jj], ir.bop[ii][jj],
                                             ir.Deltap[ii]-ir.bop[ii][jj]])
                           Bp[bdr].append([ir.bop_si[ii][jj], ir.bop_pi[ii][jj], ir.bop_pp[ii][jj]])
                           B[bdr].append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])
                           R[bdr].append(ir.r[ii][jj])
                           Y[bdr].append(ir.esi[ii][jj])
        del ir
    return D, Bp,B, R, Y

def get_atoms_data(atoms=None, gen='poscar.gen', bonds=['C-C'],
                   message_function=2,ffield='ffield.json'):
    if atoms is None:
       atoms = read(gen)

    D, Y = {}, {}
    R, Bp,B = {}, {}, {}
    for bd in bonds:
        D[bd] = []
        Y[bd] = []
        Bp[bd] = []
        B[bd] = []
        R[bd] = []

    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 nn=True, vdwnn=False)
    ir.calculate_Delta(atoms)

    for ii in range(ir.natom-1):
        for jj in range(ii+1, ir.natom):
            if ir.bop[ii][jj] > 0.0001:
               bd = ir.atom_name[ii] + '-' + ir.atom_name[jj]
               bdr = ir.atom_name[jj] + '-' + ir.atom_name[ii]
               if bd in bonds:
                  if message_function==1:
                     D[bd].append([ir.D_si[0][ii]-ir.bop_si[ii][jj], 
                                ir.D_pi[0][ii]-ir.bop_pi[ii][jj], 
                                ir.D_pp[0][ii]-ir.bop_pp[ii][jj], 
                                ir.bop[ii][jj], 
                                ir.D_pp[0][jj]-ir.bop_pp[ii][jj], 
                                ir.D_pi[0][jj]-ir.bop_pi[ii][jj], 
                                ir.D_si[0][jj]-ir.bop_si[ii][jj] ])
                  else:
                     D[bd].append([ir.Deltap[ii]-ir.bop[ii][jj], ir.bop[ii][jj], 
                                   ir.Deltap[jj]-ir.bop[ii][jj]])
                  Bp[bd].append([ir.bop_si[ii][jj],ir.bop_pi[ii][jj],ir.bop_pp[ii][jj]])
                  B[bd].append([ir.bosi[ii][jj],ir.bopi[ii][jj],ir.bopp[ii][jj]])
                  R[bd].append(ir.r[ii][jj])
                  Y[bd].append(ir.esi[ii][jj])
               elif bdr in bonds:
                  if message_function==1:
                     D[bdr].append([ir.D_si[0][jj]-ir.bop_si[ii][jj], 
                                ir.D_pi[0][jj]-ir.bop_pi[ii][jj], 
                                ir.D_pp[0][jj]-ir.bop_pp[ii][jj], 
                                ir.bop[ii][jj], 
                                ir.D_pp[0][ii]-ir.bop_pp[ii][jj], 
                                ir.D_pi[0][ii]-ir.bop_pi[ii][jj], 
                                ir.D_si[0][ii]-ir.bop_si[ii][jj] ])
                  else:
                     D[bdr].append([ir.Deltap[jj]-ir.bop[ii][jj], ir.bop[ii][jj], 
                                    ir.Deltap[ii]-ir.bop[ii][jj]])
                  Bp[bdr].append([ir.bop_si[ii][jj], ir.bop_pi[ii][jj], ir.bop_pp[ii][jj]])
                  B[bdr].append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])
                  R[bdr].append(ir.r[ii][jj])
                  Y[bdr].append(ir.esi[ii][jj])
    del ir
    return D, Bp, B, R, Y

def get_md_data(images=None, traj='md.traj', bonds=['C-C'],
                message_function=2,ffield='ffield.json'):
    if images is None:
       images = Trajectory(traj)
    # mol_   = traj.split('.')[0]
    # mol    = mol_.split('-')[0]
    D, Y = {}, {}
    R, Bp,B = {}, {}, {}
    for bd in bonds:
        D[bd] = []
        Y[bd] = []
        Bp[bd] = []
        B[bd] = []
        R[bd] = []

    ir = IRFF_NP(atoms=images[0],
                 libfile=ffield,
                 nn=True, vdwnn=False)

    for i, atoms in enumerate(images):
        ir.calculate_Delta(atoms)

        for ii in range(ir.natom-1):
            for jj in range(ii+1, ir.natom):
                if ir.bop[ii][jj] > 0.0001:
                    bd = ir.atom_name[ii] + '-' + ir.atom_name[jj]
                    bdr = ir.atom_name[jj] + '-' + ir.atom_name[ii]
                    if bd in bonds:
                       if message_function==1:
                          D[bd].append([ir.D_si[0][ii]-ir.bop_si[ii][jj], 
                                    ir.D_pi[0][ii]-ir.bop_pi[ii][jj], 
                                    ir.D_pp[0][ii]-ir.bop_pp[ii][jj], 
                                    ir.bop[ii][jj], 
                                    ir.D_pp[0][jj]-ir.bop_pp[ii][jj], 
                                    ir.D_pi[0][jj]-ir.bop_pi[ii][jj], 
                                    ir.D_si[0][jj]-ir.bop_si[ii][jj] ])
                       else:
                          D[bd].append([ir.Deltap[ii]-ir.bop[ii][jj], ir.bop[ii][jj], 
                                        ir.Deltap[jj]-ir.bop[ii][jj]])
                       Bp[bd].append([ir.bop_si[ii][jj],ir.bop_pi[ii][jj],ir.bop_pp[ii][jj]])
                       B[bd].append([ir.bosi[ii][jj],ir.bopi[ii][jj],ir.bopp[ii][jj]])
                       R[bd].append(ir.r[ii][jj])
                       Y[bd].append(ir.esi[ii][jj])
                    elif bdr in bonds:
                        if message_function==1:
                           D[bdr].append([ir.D_si[0][jj]-ir.bop_si[ii][jj], 
                                        ir.D_pi[0][jj]-ir.bop_pi[ii][jj], 
                                        ir.D_pp[0][jj]-ir.bop_pp[ii][jj], 
                                        ir.bop[ii][jj], 
                                        ir.D_pp[0][ii]-ir.bop_pp[ii][jj], 
                                        ir.D_pi[0][ii]-ir.bop_pi[ii][jj], 
                                        ir.D_si[0][ii]-ir.bop_si[ii][jj] ])
                        else:
                           D[bdr].append([ir.Deltap[jj]-ir.bop[ii][jj], ir.bop[ii][jj], 
                                          ir.Deltap[ii]-ir.bop[ii][jj]])
                        Bp[bdr].append([ir.bop_si[ii][jj], ir.bop_pi[ii][jj], ir.bop_pp[ii][jj]])
                        B[bdr].append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])
                        R[bdr].append(ir.r[ii][jj])
                        Y[bdr].append(ir.esi[ii][jj])
    del ir
    return D, Bp, B, R, Y

def get_md_data_invariance(images=None, traj='md.traj', bonds=['C-C'],
                           rcut={"H-O":1.22,"H-H":1.2,"O-O":1.4,"others": 1.8},
                           message_function=2,ffield='ffield.json'):
    if images is None:
       images = Trajectory(traj)
    # mol_   = traj.split('.')[0]
    # mol    = mol_.split('-')[0]
    A = images[0]

    mols  = Molecules(A,rcut=rcut,check=True)
    nmol  = len(mols)
    print('\nnumber of molecules in trajectory: {:d}'.format(nmol))

    ir_total = IRFF_NP(atoms=A, libfile='ffield.json',nn=True)
    ir_total.calculate(A)
    print('\nTotal energy: \n',ir_total.E)

    ir    = [None for i in range(nmol)] 
    atoms = [None for i in range(nmol)] 

    for i,m in enumerate(mols):
        atoms[i] = moltoatoms([m])
        ir[i] = IRFF_NP(atoms=atoms[i],libfile='ffield.json',nn=True)
        ir[i].calculate(atoms[i])
        print('\nEnergy of molecule {:4d}: \n'.format(i),ir[i].E)
        # print(m.mol_index)
        # view(atoms)

    D, Y = {}, {}
    R, Bp,B = {}, {}, {}
    for bd in bonds:
        D[bd] = []
        Y[bd] = []
        Bp[bd] = []
        B[bd] = []
        R[bd] = []

    for i, A in enumerate(images):
        ir_total.calculate_Delta(A)
        positions = A.positions
        for n,m in enumerate(mols):
            atoms[n].positions = positions[m.mol_index]
            ir[n].calculate(atoms[n])
            for ii in range(ir[n].natom-1):
                for jj in range(ii+1, ir[n].natom):
                    if ir[n].bop[ii][jj] > 0.0001:
                        bd = ir[n].atom_name[ii] + '-' + ir[n].atom_name[jj]
                        bdr = ir[n].atom_name[jj] + '-' + ir[n].atom_name[ii]
                        if bd in bonds:
                            if message_function==1:
                               D[bd].append([ir_total.D_si[0][m.mol_index[ii]]-ir[n].bop_si[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[ii]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[ii]]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[jj]]-ir[n].bop_pp[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[jj]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_si[0][m.mol_index[jj]]-ir[n].bop_si[ii][jj] ])
                            else:
                               D[bd].append([ir_total.Deltap[m.mol_index[ii]]-ir[n].bop[ii][jj], 
                                            ir[n].bop[ii][jj],
                                            ir_total.Deltap[m.mol_index[jj]]-ir[n].bop[ii][jj]])
                            Bp[bd].append([ir[n].bop_si[ii][jj],ir[n].bop_pi[ii][jj],ir[n].bop_pp[ii][jj]])
                            B[bd].append([ir[n].bosi[ii][jj],ir[n].bopi[ii][jj],ir[n].bopp[ii][jj]])
                            R[bd].append(ir[n].r[ii][jj])
                            Y[bd].append(ir[n].esi[ii][jj])
                        elif bdr in bonds:
                            if message_function==1:
                               D[bdr].append([ir_total.D_si[0][m.mol_index[jj]]-ir[n].bop_si[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[jj]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[jj]]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[ii]]-ir[n].bop_pp[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[ii]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_si[0][m.mol_index[ii]]-ir[n].bop_si[ii][jj] ])
                            else:
                               D[bdr].append([ir_total.Deltap[m.mol_index[jj]]-ir[n].bop[ii][jj], 
                                              ir[n].bop[ii][jj],
                                              ir_total.Deltap[m.mol_index[ii]]-ir[n].bop[ii][jj]])
                            Bp[bdr].append([ir[n].bop_si[ii][jj], ir[n].bop_pi[ii][jj], ir[n].bop_pp[ii][jj]])
                            B[bdr].append([ir[n].bosi[ii][jj], ir[n].bopi[ii][jj], ir[n].bopp[ii][jj]])
                            R[bdr].append(ir[n].r[ii][jj])
                            Y[bdr].append(ir[n].esi[ii][jj])
    del ir
    del ir_total
    return D, Bp, B, R, Y

def get_md_data_inv(trajs=[], bonds=[],
                    rcut={"H-O":1.22,"H-H":1.2,"O-O":1.4,"others": 1.8},
                    message_function=2,ffield='ffield.json'):
    ''' Prepare data for penalty term for translation invariant of
        molecules.
    '''
    nbd           = {}
    D, Dt         = {}, {}
    D_mol, Dt_mol = {}, {}
    for bd in bonds:
        D[bd]      = []
        Dt[bd]     = []
        D_mol[bd]  = []
        Dt_mol[bd] = []

    for traj in trajs:
        images = Trajectory(traj)
        A = images[0]
        mols  = Molecules(A,rcut=rcut,check=True)
        nmol  = len(mols)
        with open('penalty.log','a') as f:
             print('\nnumber of molecules in trajectory:\n {:s}  {:d}'.format(traj,nmol),file=f)

        ir_total = IRFF_NP(atoms=A, libfile='ffield.json',nn=True)
        ir_total.calculate(A)
        # print('\nTotal energy: \n',ir_total.E)
        ir    = [None for i in range(nmol)] 
        atoms = [None for i in range(nmol)] 

        for i,m in enumerate(mols):
            atoms[i] = moltoatoms([m])
            ir[i] = IRFF_NP(atoms=atoms[i],libfile='ffield.json',nn=True)
            ir[i].calculate(atoms[i])
            # print(m.mol_index)
            
        for i, A in enumerate(images):
            ir_total.calculate_Delta(A)
            positions = A.positions
            for n,m in enumerate(mols):
                atoms[n].positions = positions[m.mol_index]
                ir[n].calculate(atoms[n])
                for ii in range(ir[n].natom-1):
                    for jj in range(ii+1, ir[n].natom):
                        if ir[n].bop[ii][jj] > 0.0001:
                            bd = ir[n].atom_name[ii] + '-' + ir[n].atom_name[jj]
                            bdr = ir[n].atom_name[jj] + '-' + ir[n].atom_name[ii]
                            if bd in bonds:
                               if message_function==1:
                                  D_mol[bd].append([ir[n].D_si[0][ii]-ir[n].bop_si[ii][jj], 
                                            ir[n].D_pi[0][ii]-ir[n].bop_pi[ii][jj], 
                                            ir[n].D_pp[0][ii]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir[n].D_pp[0][jj]-ir[n].bop_pp[ii][jj], 
                                            ir[n].D_pi[0][jj]-ir[n].bop_pi[ii][jj], 
                                            ir[n].D_si[0][jj]-ir[n].bop_si[ii][jj] ])
                                  D[bd].append([ir_total.D_si[0][m.mol_index[ii]]-ir[n].bop_si[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[ii]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[ii]]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[jj]]-ir[n].bop_pp[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[jj]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_si[0][m.mol_index[jj]]-ir[n].bop_si[ii][jj] ])
                                  Dt_mol[bd].append([ir[n].D_si[0][jj]-ir[n].bop_si[ii][jj], 
                                            ir[n].D_pi[0][jj]-ir[n].bop_pi[ii][jj], 
                                            ir[n].D_pp[0][jj]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir[n].D_pp[0][ii]-ir[n].bop_pp[ii][jj], 
                                            ir[n].D_pi[0][ii]-ir[n].bop_pi[ii][jj], 
                                            ir[n].D_si[0][ii]-ir[n].bop_si[ii][jj] ])
                                  Dt[bd].append([ir_total.D_si[0][m.mol_index[jj]]-ir[n].bop_si[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[jj]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[jj]]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[ii]]-ir[n].bop_pp[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[ii]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_si[0][m.mol_index[ii]]-ir[n].bop_si[ii][jj] ])
                               else:
                                  D_mol[bd].append([ir[n].Deltap[ii]-ir[n].bop[ii][jj], 
                                                ir[n].bop[ii][jj], ir[n].Deltap[jj]-ir[n].bop[ii][jj]])
                                  D[bd].append([ir_total.Deltap[m.mol_index[ii]]-ir[n].bop[ii][jj], 
                                            ir[n].bop[ii][jj],
                                            ir_total.Deltap[m.mol_index[jj]]-ir[n].bop[ii][jj]])
                                  Dt_mol[bd].append([ir[n].Deltap[jj]-ir[n].bop[ii][jj], 
                                            ir[n].bop[ii][jj], ir[n].Deltap[ii]-ir[n].bop[ii][jj]])
                                  Dt[bd].append([ir_total.Deltap[m.mol_index[jj]]-ir[n].bop[ii][jj], 
                                            ir[n].bop[ii][jj],
                                            ir_total.Deltap[m.mol_index[ii]]-ir[n].bop[ii][jj]])
                            elif bdr in bonds:
                               if message_function==1:
                                  Dt_mol[bd].append([ir[n].D_si[0][ii]-ir[n].bop_si[ii][jj], 
                                            ir[n].D_pi[0][ii]-ir[n].bop_pi[ii][jj], 
                                            ir[n].D_pp[0][ii]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir[n].D_pp[0][jj]-ir[n].bop_pp[ii][jj], 
                                            ir[n].D_pi[0][jj]-ir[n].bop_pi[ii][jj], 
                                            ir[n].D_si[0][jj]-ir[n].bop_si[ii][jj] ])
                                  Dt[bd].append([ir_total.D_si[0][m.mol_index[ii]]-ir[n].bop_si[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[ii]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[ii]]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[jj]]-ir[n].bop_pp[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[jj]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_si[0][m.mol_index[jj]]-ir[n].bop_si[ii][jj] ])
                                  D_mol[bd].append([ir[n].D_si[0][jj]-ir[n].bop_si[ii][jj], 
                                            ir[n].D_pi[0][jj]-ir[n].bop_pi[ii][jj], 
                                            ir[n].D_pp[0][jj]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir[n].D_pp[0][ii]-ir[n].bop_pp[ii][jj], 
                                            ir[n].D_pi[0][ii]-ir[n].bop_pi[ii][jj], 
                                            ir[n].D_si[0][ii]-ir[n].bop_si[ii][jj] ])
                                  D[bd].append([ir_total.D_si[0][m.mol_index[jj]]-ir[n].bop_si[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[jj]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[jj]]-ir[n].bop_pp[ii][jj], 
                                            ir[n].bop[ii][jj], 
                                            ir_total.D_pp[0][m.mol_index[ii]]-ir[n].bop_pp[ii][jj], 
                                            ir_total.D_pi[0][m.mol_index[ii]]-ir[n].bop_pi[ii][jj], 
                                            ir_total.D_si[0][m.mol_index[ii]]-ir[n].bop_si[ii][jj] ])
                               else:
                                  D_mol[bdr].append([ir[n].Deltap[jj]-ir[n].bop[ii][jj], 
                                            ir[n].bop[ii][jj], ir[n].Deltap[ii]-ir[n].bop[ii][jj]])
                                  D[bdr].append([ir_total.Deltap[m.mol_index[jj]]-ir[n].bop[ii][jj], 
                                        ir[n].bop[ii][jj],
                                        ir_total.Deltap[m.mol_index[ii]]-ir[n].bop[ii][jj]])
                                  Dt_mol[bdr].append([ir[n].Deltap[ii]-ir[n].bop[ii][jj], 
                                            ir[n].bop[ii][jj], ir[n].Deltap[jj]-ir[n].bop[ii][jj]])
                                  Dt[bdr].append([ir_total.Deltap[m.mol_index[ii]]-ir[n].bop[ii][jj], 
                                        ir[n].bop[ii][jj],
                                        ir_total.Deltap[m.mol_index[jj]]-ir[n].bop[ii][jj]])
        del ir
        del ir_total
    for bd in bonds:
        nbd[bd] = len(D[bd])
    return D, Dt, D_mol, Dt_mol, nbd

def get_bond_data(ii, jj, images=None, traj='md.traj', bonds=None,
                  message_function=2,ffield='ffield.json'):
    if images is None:
        images = Trajectory(traj)
    D = []
    Y = []
    Bp= []
    B = []
    R = []
    ir = IRFF_NP(atoms=images[0],
                 libfile=ffield,
                 nn=True, vdwnn=False)
                 
    if bonds is None:
        bonds = [ir.atom_name[ii]+'-'+ir.atom_name[jj]]

    bd = ir.atom_name[ii] + '-' + ir.atom_name[jj]
    bdr = ir.atom_name[jj] + '-' + ir.atom_name[ii]

    for i, atoms in enumerate(images):
        ir.calculate_Delta(atoms)
        if bd in bonds:
           if message_function==1:
              D.append([ir.D_si[0][ii]-ir.bop_si[ii][jj], 
                        ir.D_pi[0][ii]-ir.bop_pi[ii][jj], 
                        ir.D_pp[0][ii]-ir.bop_pp[ii][jj], 
                        ir.bop[ii][jj], 
                        ir.D_pp[0][jj]-ir.bop_pp[ii][jj], 
                        ir.D_pi[0][jj]-ir.bop_pi[ii][jj], 
                        ir.D_si[0][jj]-ir.bop_si[ii][jj] ])
           else:
              D.append([ir.Deltap[ii]-ir.bop[ii][jj], ir.bop[ii][jj], ir.Deltap[jj]-ir.bop[ii][jj]])
        elif bdr in bonds:
           if message_function==1:
              D.append([ir.D_si[0][jj]-ir.bop_si[ii][jj], 
                        ir.D_pi[0][jj]-ir.bop_pi[ii][jj], 
                        ir.D_pp[0][jj]-ir.bop_pp[ii][jj], 
                        ir.bop[ii][jj], 
                        ir.D_pp[0][ii]-ir.bop_pp[ii][jj], 
                        ir.D_pi[0][ii]-ir.bop_pi[ii][jj], 
                        ir.D_si[0][ii]-ir.bop_si[ii][jj] ])
           else:
              D.append([ir.Deltap[jj]-ir.bop[ii][jj],ir.bop[ii][jj],ir.Deltap[ii]-ir.bop[ii][jj]])
        Bp.append([ir.bop_si[ii][jj],ir.bop_pi[ii][jj],ir.bop_pp[ii][jj]])
        B.append([ir.bosi[ii][jj], ir.bopi[ii][jj], ir.bopp[ii][jj]])                   
        R.append(ir.r[ii][jj])
        Y.append(ir.esi[ii][jj])
    del ir
    return D, Bp, B, R, Y

