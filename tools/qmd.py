#!/usr/bin/env python
from os import system,getcwd
import argh
import argparse
#from irff.dft.SinglePointEnergy import SinglePointEnergies
from ase.io import read
from irff.dft.qe import qemd,qeopt # ,write_qe_in
from irff.molecule import compress,press_mol
from irff.data.mdtodata import MDtoData


def md(ncpu=20,T=300,tstep=50,dt=1.0,gen='poscar.gen',
       kpts=(1,1,1),index=-1):
    atoms = read(gen,index=index)
    # A = press_mol(A)
    print('\n-  running QE(PW) md ...')
    qemd(atoms=atoms,ncpu=ncpu,
         kpts=kpts,
         T=T,dt=dt,tstep=tstep)

def opt(ncpu=8,T=2500,gen='poscar.gen',kpts=(6,6,1),l=0):
    atoms = read(gen,index=-1)
    # A = press_mol(A)
    print('\n-  running QE opt ...')
    qeopt(atoms,ncpu=ncpu,VariableCell=l,
          coord='crystal',
          conv_thr=0.000000001,
          forc_conv_thr=0.00001,
          K_POINTS=kpts)  #ecutwfc=90.0,ecutrho=900.0,

def traj():
    cwd = getcwd()
    d = MDtoData(structure='qe',dft='qe',direc=cwd,batch=10000)
    d.get_traj()
    d.close()

if __name__ == '__main__':
   ''''
   *.traj: contains the structure to be calculated by DFT
   label : the trajectory name include the energies and sturcutres calculated by DFT
   frame : number of frames to be collected to calculate the energy, if frame> the frame of *.traj contains then
           equal the number of frame of *.traj contains
   cpu   : number of cpu to be parallelly used 
   '''
   # SinglePointEnergies('swing.traj',label='nm2-s',frame=10,dft='qe',cpu=4)
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [md,opt,traj])
   argh.dispatch(parser)



