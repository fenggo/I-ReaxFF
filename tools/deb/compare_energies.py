#!/usr/bin/env python
# coding: utf-8
import numpy as np
#import jax.numpy as jnp
from os import system
import numpy as np
import csv
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units
#from ase.visualize import view
#from irff.irff_jax import IRFF
from irff.irff_np import IRFF_NP
# from .irff.AtomDance import AtomDance
from irff.md.gulp import write_gulp_in,get_reax_energy
from irff.md.lammps import writeLammpsIn,writeLammpsData,get_reaxff_energies
import matplotlib.pyplot as plt
from irff.tools.ffieldtolib import ffieldtolib

 
def plot_compare(jax_energy,gulp_energy,show=True,mode=None,label='gulp'):
    plt.figure(figsize=(15, 10))
    plt.subplot(2,3,1)
    # fig, ax1 = plt.subplots(3,3)
    plt.plot(jax_energy[1],alpha=0.8,linestyle='-',marker='^',
             markerfacecolor='none',ms=8,color='r',label='IRFF-Ebond')
    # plt.yticks([-0.096,-0.0959,-0.0958,-0.0957,-0.0956,-0.0955,-0.0954,-0.0953,-0.0952,-0.0951,-0.0950,-0.0925,-0.0924,-0.0923,-0.0922,-0.0921,-0.0920,-0.0919,-0.0918,-0.0917,])
    # plt.yticks(jnp.arange(-0.95175,-0.95155,0.005))
    plt.legend(loc='best', edgecolor='red')
    # ax1.twiny()
    plt.plot(gulp_energy[1], alpha=0.8, linestyle='-', marker='o',
             markerfacecolor='none',ms=3, color='b', label='{:s}-Ebond'.format(label))
    # plt.yticks(jnp.arange(-65.27,-65.25,0.005))
    plt.legend(loc='best', edgecolor='blue')

    # plt.subplot(3,3,2)
    # plt.plot(jax_energy[3],alpha=0.8,linestyle='-',marker='^',color='r',label='IRFF-Eover')
    # plt.legend(loc='best',edgecolor='red')
    # plt.plot(gulp_energy[3], alpha=0.8, linestyle='-', marker='o', color='b', label='{:s}-Eover'.format(label))
    # plt.legend(loc='best', edgecolor='blue')

    # plt.subplot(3,3,3)
    # plt.plot(jax_energy[2],alpha=0.8,linestyle='-',marker='^',color='r',label='IRFF-Eunder')
    # plt.legend(loc='best',edgecolor='red')
    # plt.plot(gulp_energy[2], alpha=0.8, linestyle='-', marker='o', color='b', label='{:s}-Eunder'.format(label))
    # plt.legend(loc='best', edgecolor='blue')

    plt.subplot(2,3,2)
    plt.plot(jax_energy[5],alpha=0.8,linestyle='-',marker='^',
             markerfacecolor='none',ms=8,color='r',label='IRFF-Eang')
    plt.legend(loc='best',edgecolor='red')
    plt.plot(gulp_energy[5], alpha=0.8, linestyle='-', marker='o', 
             markerfacecolor='none',ms=3,color='b', label='{:s}-Eang'.format(label))
    plt.legend(loc='best', edgecolor='blue')

    plt.subplot(2,3,3)
    plt.plot(jax_energy[12],alpha=0.8,linestyle='-',marker='^',
              markerfacecolor='none',ms=8,color='r',label='IRFF-Ecoul')
    plt.legend(loc='best',edgecolor='red')
    plt.plot(gulp_energy[12], alpha=0.8, linestyle='-', marker='o', 
              markerfacecolor='none',ms=3,color='b', label='{:s}-Ecoul'.format(label))
    plt.legend(loc='best', edgecolor='blue')

    plt.subplot(2,3,4)
    plt.plot(jax_energy[8],alpha=0.8,linestyle='-',marker='^',
             markerfacecolor='none',ms=8,color='r',label='IRFF-Etor')
    plt.legend(loc='best',edgecolor='red')
    plt.plot(gulp_energy[8], alpha=0.8, linestyle='-', marker='o',
             markerfacecolor='none',ms=3, color='b', label='{:s}-Etor'.format(label))
    plt.legend(loc='best', edgecolor='blue')


    # Ebv = np.array(Ev) + np.array(Eb) + np.array(Eo) + np.array(Eu)
    # plt.subplot(2,3,4)
    # plt.plot(jax_energy[11],alpha=0.8,linestyle='-',marker='^',
    #          markerfacecolor='none',ms=8,color='r',label='IRFF-Ehb')
    # plt.legend(loc='best',edgecolor='red')
    # plt.plot(gulp_energy[11], alpha=0.8, linestyle='-', marker='o', 
    #          ms=3,color='b', label='{:s}-Ehb'.format(label))
    # plt.legend(loc='best', edgecolor='blue')

    plt.subplot(2,3,5)
    plt.plot(jax_energy[10],alpha=0.8,linestyle='-',marker='^',
             markerfacecolor='none',ms=8,color='r',label='IRFF-Evdw')
    plt.legend(loc='best',edgecolor='red')
    plt.plot(gulp_energy[10], alpha=0.8, linestyle='-', marker='o', 
             markerfacecolor='none',ms=3,color='b', label='{:s}-Evdw'.format(label))
    plt.legend(loc='best', edgecolor='blue')

    plt.subplot(2,3,6)
    e = np.array(jax_energy[0]) - min(jax_energy[0])
    plt.plot(e,alpha=0.8,linestyle='-',marker='^',color='r',
             markerfacecolor='none',ms=8,label='IRFF-Total Energy')
    plt.legend(loc='best',edgecolor='red')
    e = np.array(gulp_energy[0]) - min(gulp_energy[0])
    plt.plot(e, alpha=0.8, linestyle='-', marker='o', color='b', 
             markerfacecolor='none',ms=3,label='{:s}-Total Energy'.format(label))
    plt.legend(loc='best', edgecolor='blue')
    plt.savefig('deb_energies.pdf')
    # if show: plt.show()
    plt.close()

def deb_energy(images,atomi=0,atomj=1,debframe=[],show=True):
    ir = IRFF_NP(atoms=images[0],nn=True,
                 libfile='ffield.json',
                 autograd=True)
    ir.calculate(images[0])
    Eb,Ea,Ec,e = [],[],[],[]
    Ehb,Eo,Ev,Eu,El = [],[],[],[],[]
    Etor,Ef,Ep,Et = [],[],[],[]
    # fcsv = open('energies.csv','w')
    # csv_write = csv.writer(fcsv)
    # csv_write.writerow(['r','etotal','ebond','elone','eover','eunder','eangle',
    #                     'econj','epen','etor',
    #                     'efcon','evdw','ecoul','ehb'])

    for i_,atoms in enumerate(images):
        ir.calculate(images[i_])
        # print('%d Energies: ' %i_,'%12.4f ' %ir.E, 'Ebd: %8.4f' %ir.ebond[0][1],'Ebd: %8.4f' %ir.ebond[2][3] )
        r  = ir.r[atomi][atomj]
        Eb.append(ir.Ebond)
        Ea.append(ir.Eang)
        Eo.append(ir.Eover)
        Ev.append(ir.Evdw)
        Eu.append(ir.Eunder)
        El.append(ir.Elone)
        Ep.append(ir.Epen)
        Et.append(ir.Etcon)
        Ef.append(ir.Efcon)
        Etor.append(ir.Etor)
        Ehb.append(ir.Ehb)
        ecoul = ir.Ecoul if abs(ir.Ecoul)>0.00000001 else 0.0
        Ec.append(ecoul)
        e.append(ir.E-ir.zpe)
        # csv_write.writerow([r,ir.E,ir.Ebond,ir.Elone,ir.Eover,ir.Eunder,ir.Eang, ir.Etcon,ir.Epen,ir.Etor,
        #                     ir.Efcon,ir.Evdw,ir.Ecoul,ir.Ehb])
        
        # print(e[-1])
    # fcsv.close()
    return e,Eb,Eu,Eo,El,Ea,Et,Ep,Etor,Ef,Ev,Ehb,Ec


def deb_gulp_energy(images,atomi=0,atomj=1,ffield='reax.lib',show=False):
    ''' test reax with GULP, and run validation-set'''
    fcsv = open('gulp.csv','w')
    csv_write = csv.writer(fcsv)
    csv_write.writerow(['r','etotal','ebond','elone','eover','eunder','eangle',
                        'econj','epen','etor',
                        'efcon','evdw','ecoul','ehb'])

    GE={}
    GE['ebond'],GE['elonepair'],GE['eover'],GE['eunder'],GE['eangle'], \
    GE['econjugation'],GE['evdw'],GE['Total-Energy'] = \
             [],[],[],[],[],[],[],[]
    GE['epenalty'],GE['etorsion'],GE['fconj'],GE['ecoul'],GE['eself'],GE['ehb']\
           = [],[],[],[],[],[] 

    for atoms in images: 
        write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',
                      lib=ffield)
        system('gulp<inp-gulp>out')

        (e_,eb_,el_,eo_,eu_,ea_,ep_,
         etc_,et_,ef_,ev_,ehb_,ecl_,esl_)= get_reax_energy(fo='out')
        positions = atoms.get_positions()
        vr = positions[atomi] - positions[atomj]
        r  = np.sqrt(np.sum(vr*vr,axis=0))

        csv_write.writerow([r,e_,eb_,el_,eo_,eu_,ea_, etc_,ep_,et_,
                            ef_,ev_,ecl_,ehb_])

        GE['ebond'].append(eb_)
        GE['elonepair'].append(el_)
        GE['eover'].append(eo_)
        GE['eunder'].append(eu_)
        GE['eangle'].append(ea_)
        GE['econjugation'].append(etc_)
        GE['epenalty'].append(ep_)
        GE['etorsion'].append(et_)
        GE['fconj'].append(ef_)
        GE['evdw'].append(ev_)
        GE['ecoul'].append(ecl_)
        GE['ehb'].append(ehb_)
        GE['eself'].append(esl_)
        GE['Total-Energy'].append(e_)
        # print(e_)

    # GE['Total-Energy'] = np.array(GE['Total-Energy'])
    # print('GE[total-Energy]',GE['Total-Energy'])
    return GE['Total-Energy'],GE['ebond'],GE['eunder'],GE['eover'],\
           GE['elonepair'],GE['eangle'],GE['econjugation'],GE['epenalty'],\
           GE['etorsion'],GE['fconj'],GE['evdw'],GE['ehb'],GE['ecoul']

def deb_lammps_energy(images,atomi=0,atomj=1,ffield='ffield'):
    ''' test ReaxFF-nn for lammps '''

    GE={}
    GE['ebond'],GE['elonepair'],GE['eover'],GE['eunder'],GE['eangle'], \
    GE['econjugation'],GE['evdw'],GE['Total-Energy'] = \
             [],[],[],[],[],[],[],[]
    GE['epenalty'],GE['etorsion'],GE['fconj'],GE['ecoul'],GE['eself'],GE['ehb']\
           = [],[],[],[],[],[] 
    thermo = ''' thermo_style    custom step temp epair etotal press &
                   v_eb v_ea v_elp v_emol v_ev v_epen v_ecoa &
                   v_ehb v_et v_eco v_ew v_ep v_efi v_eqeq 
             '''
    atoms = images[0]
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp      = ' '.join(species)
    for atoms in images: 
        writeLammpsData(atoms,data='data.lammps',specorder=None, 
                        masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                                'N':14.0000,'F':18.9980},
                        force_skew=False,
                        velocities=False,units="real",atom_style='charge')
        writeLammpsIn(log='lmp.log',timestep=0.1,total=0,restart=None,
                species=species,
                pair_coeff ='* * {:s} {:s}'.format('ffield',sp),
                pair_style = 'reaxff control nn yes checkqeq yes',  # without lg set lgvdw no
                fix = 'fix   1 all nve ',
                fix_modify = ' ',
                more_commond = ' ',
                thermo_style =thermo,
                data='data.lammps',
                restartfile='restart')
        system('lammps<in.lammps>out')

        (e_,eb_,el_,eo_,eu_,ea_,ep_,
         etc_,et_,ef_,ev_,ehb_,ecl_,esl_)= get_reaxff_energies(logname='lmp.log')
        positions = atoms.get_positions()
        vr = positions[atomi] - positions[atomj]
        r  = np.sqrt(np.sum(vr*vr,axis=0))

        GE['ebond'].append(eb_)
        GE['elonepair'].append(el_)
        GE['eover'].append(eo_)
        GE['eunder'].append(eu_)
        GE['eangle'].append(ea_)
        GE['econjugation'].append(etc_)
        GE['epenalty'].append(ep_)
        GE['etorsion'].append(et_)
        GE['fconj'].append(ef_)
        GE['evdw'].append(ev_)
        GE['ecoul'].append(ecl_)
        GE['ehb'].append(ehb_)
        GE['eself'].append(esl_)
        GE['Total-Energy'].append(e_)
        # print(e_)

    # GE['Total-Energy'] = np.array(GE['Total-Energy'])
    # print('GE[total-Energy]',GE['Total-Energy'])
    return GE['Total-Energy'],GE['ebond'],GE['eunder'],GE['eover'],\
           GE['elonepair'],GE['eangle'],GE['econjugation'],GE['epenalty'],\
           GE['etorsion'],GE['fconj'],GE['evdw'],GE['ehb'],GE['ecoul']

if __name__ == '__main__':
    images = Trajectory('md.traj')
    irff_energy = deb_energy(images=images)
    # ffieldtolib()
    # gulp_energy = deb_gulp_energy(images, ffield='reaxff_nn')
    lmp_energy = deb_lammps_energy(images, ffield='ffield')
    plot_compare(irff_energy, lmp_energy,label='LAMMPS')

