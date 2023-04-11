#!/usr/bin/env python
# coding: utf-8
import numpy as np
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units
from ase.visualize import view
from irff.irff_np import IRFF_NP
# from irff_np import IRFF_NP
# from irff.irff import IRFF
import matplotlib.pyplot as plt
import argh
import argparse


def bde(i=2,j=3,rmin=0.4,ffield='ffield.json',nn='T',gen='poscar.gen'):
    atoms = read(gen)
    nn_=True if nn=='T'  else False
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 rcut=None,
                 nn=nn_,massages=1)

    ir.calculate_Delta(atoms)
    natom = ir.natom

    positions = atoms.positions
    v = positions[j] - positions[i]
    r = np.sqrt(np.sum(np.square(v)))
    u = v/r
    # print(u)

    r_,eb,bosi,bop_si,esi,bop,bop_pi,bop_pp,bo = [],[],[],[],[],[],[],[],[]
    e,eba,eo = [],[],[]
    esi,epi,epp = [],[],[]

    for i_ in range(50):
        r = rmin + i_*0.015
        atoms.positions[j] = atoms.positions[i] + u*r
        v = positions[j] - positions[i]
        R = np.sqrt(np.sum(np.square(v)))
        ir.calculate(atoms)
        r_.append(r)
        eb.append(ir.ebond[i][j])
        eba.append(ir.ebond[i][j] + ir.eover[i] + ir.eover[j]) 
        eo.append(ir.eover[i] + ir.eover[j]) 
        bo.append(ir.bo0[i][j])
        bosi.append(ir.bosi[i][j])

        esi.append(-ir.sieng[i][j])
        epi.append(-ir.pieng[i][j])
        epp.append(-ir.ppeng[i][j])

        bop.append(ir.bop[i][j])
        bop_si.append(ir.bop_si[i][j])
        bop_pi.append(ir.bop_pi[i][j])
        bop_pp.append(ir.bop_pp[i][j])

        e.append(ir.E)

    plt.figure()

    plt.subplot(2,2,1)  
    plt.plot(r_,e,label=r'$E_{tot}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(2,2,2)  
    #plt.ylabel( r'$\sigma$ Bond-Energy (eV)')
    # plt.xlabel(r'$Radius$ $(Angstrom)$')
    plt.plot(r_,esi,label=r'$E_{bond}^{\sigma}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(2,2,3)  
    plt.plot(r_,epi,label=r'$E_{bond}^{\pi}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(2,2,4)  
    plt.plot(r_,epp,label=r'$E_{bond}^{\pi\pi}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('Ebond.eps') 
    plt.close()


def energies(rmin=0.4,ffield='ffield.json',nn='T',traj='md.traj',massages=2):
    images= Trajectory(traj)
    atoms = images[0]
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 rcut=None,
                 nn=nn,massages=massages)
    ir.calculate(atoms)
    natom = ir.natom


    r_,eb,bosi,bop_si,esi,bop,bop_pi,bop_pp = [],[],[],[],[],[],[],[]
    e,eo,eu,el,ea,ep,et,ev,eh,ef,ec,etc,es = [],[],[],[],[],[],[],[],[],[],[],[],[]

    for atoms in images:
        ir.calculate(atoms)
        eb.append(ir.Ebond)
        e.append(ir.E)
        eo.append(ir.Eover)
        eu.append(ir.Eunder)
        el.append(ir.Elone)
        ea.append(ir.Eang)
        ep.append(ir.Epen)
        et.append(ir.Etor)
        ef.append(ir.Efcon)
        ev.append(ir.Evdw)
        eh.append(ir.Ehb)
        ec.append(ir.Ecoul)
        etc.append(ir.Etcon)
        es.append(ir.Eself)

    plt.figure()

    plt.subplot(4,3,1)  
    plt.plot(e,label=r'$E_{tot}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,2)  
    #plt.ylabel( r'$\sigma$ Bond-Energy (eV)')
    # plt.xlabel(r'$Radius$ $(Angstrom)$')
    plt.plot(eb,label=r'$E_{bond}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,3)  
    plt.plot(eo,label=r'$E_{over}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,4)  
    plt.plot(eu,label=r'$E_{under}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,5)  
    plt.plot(el,label=r'$E_{lone}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,6)  
    plt.plot(ea,label=r'$E_{angle}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,7)  
    plt.plot(ep,label=r'$E_{penalty}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,8)  
    plt.plot(et,label=r'$E_{tor}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,9)  
    plt.plot(ef,label=r'$E_{four}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,10)  
    plt.plot(ev,label=r'$E_{vdw}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,11)  
    plt.plot(eh,label=r'$E_{hbond}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(4,3,12)  
    plt.plot(ec,label=r'$E_{coulb}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('Energies.eps') 
    plt.close()


def over(i=2,j=3,rmin=0.4,ffield='ffield.json',nn='T',gen='poscar.gen'):
    atoms = read(gen)
    nn_=True if nn=='T'  else False
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 rcut=None,
                 nn=nn_,massages=1)

    ir.calculate_Delta(atoms)
    natom = ir.natom

    positions = atoms.positions
    v = positions[j] - positions[i]
    r = np.sqrt(np.sum(np.square(v)))
    u = v/r
    # print(u)

    r_,eb,bosi,bop_si,bop,bop_pi,bop_pp,bo = [],[],[],[],[],[],[],[]
    eba,eo,dlpi,dlpj,ev,boe = [],[],[],[],[],[]
    esi,epi,epp = [],[],[]
   
    for i_ in range(50):
        r = rmin + i_*0.015
        atoms.positions[j] = atoms.positions[i] + u*r
        v = positions[j] - positions[i]
        R = np.sqrt(np.sum(np.square(v)))
        ir.calculate(atoms)
        r_.append(r)
        eb.append(ir.ebond[i][j])
        eba.append(ir.ebond[i][j] + ir.eover[i] + ir.Evdw) 
        ev.append(ir.Evdw)

        eo.append(ir.eover[i] + ir.eover[j]) 
        bo.append(ir.bo0[i][j])
        bosi.append(ir.bosi[i][j])
        esi.append(-ir.sieng[i][j])
        boe.append(ir.esi[i][j])

        bop.append(ir.bop[i][j])
        bop_si.append(ir.bop_si[i][j])
        bop_pi.append(ir.bop_pi[i][j])
        bop_pp.append(ir.bop_pp[i][j])

        dlpi.append(ir.Delta_lpcorr[i])
        dlpj.append(ir.Delta_lpcorr[j])

    plt.figure()
    plt.subplot(3,2,1)  
    #plt.ylabel( r'$\sigma$ Bond-Energy (eV)')
    # plt.xlabel(r'$Radius$ $(Angstrom)$')
    plt.plot(r_,eb,label=r'$E_{bond}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,2)  
    plt.plot(r_,eba,label=r'$E_{bond}$ +  $E_{vdw}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')


    plt.subplot(3,2,3)  
    plt.plot(r_,bo,label=r'$E_{over}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')


    plt.subplot(3,2,4)  
    plt.plot(r_,bosi,label=r'$BO_{\sigma}$ t=1', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')


    plt.subplot(3,2,5)  
    plt.plot(r_,boe,label=r'$BO_{\sigma}^e$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,6)  
    plt.plot(r_,dlpi,label=r'$\Delta^{i}_{lpcorr}$(%s)' %ir.atom_name[i], 
             color='r', linewidth=2, linestyle='-')
    plt.plot(r_,dlpj,label=r'$\Delta^{j}_{lpcorr}$(%s)' %ir.atom_name[j], 
             color='b', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('Eover.eps') 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./mpbd.py <mp> to run it
       use --h to see options
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [bde,energies,over])
   argh.dispatch(parser)

