from .siesta import single_point as single_point_siesta
from .qe import single_point as single_point_qe
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Ry
from ase import Atoms
import numpy as np


def SinglePointEnergies(traj,label='aimd',xcf='VDW',xca='DRSLL',basistype='DZP',
                        EngTole=0.0000001,frame=50,cpu=4,dft='siesta',kpts=(1,1,1),
                        dE=0.2,d2E=0.1,colmin=2,select=False):
    ''' get single point energy and labeling data '''
    images = Trajectory(traj)
    tframe = len(images)
    E,E_,dEs = [],[],[]
    if tframe>frame:
       if frame>1:
          ind_  = list(np.linspace(0,tframe-1,num=frame,dtype=np.int32))
       else:
          ind_  = [tframe-1]
    else:
       ind_  = [i for i in range(tframe)]

    if len(ind_)>1 and 0 in ind_:
       ind_.pop(0)

    his      = TrajectoryWriter(label+'.traj',mode='w')
    energies = []
    d2Es     = []
    dE_      = 0.0
    d2E_     = 0.0

    for i,atoms in enumerate(images):
        energy = atoms.get_potential_energy()
        extreme_point = False

        if i>0: 
           if i<(tframe-1):
              deltEl = energy - energies[-1]
              deltEr = images[i+1].get_potential_energy() - energy
              dE_ = abs(deltEl)
              d2E_= abs(deltEr-deltEl)
           else:
              deltEl =  energy - energies[-1]
              deltEr =  deltEl
              dE_ = abs(deltEl)

           if (deltEr>0.0 and deltEl<0.0) or (deltEr<0.0 and deltEl>0.0) or dE_<0.00001:
              extreme_point = True

           if select:
              if dE_>dE or extreme_point:
                 if i not in ind_:
                    ajacent_not_in = True
                    for j in range(1,colmin):
                        ii = i-j
                        # ii_= i+j
                        if ii>0 and (ii not in ind_): #and (ii_ not in ind_):
                           ajacent_not_in = False
                    if ajacent_not_in:
                       ind_.append(i)

        dEs.append(dE_)
        d2Es.append(d2E_)
        energies.append(energy)

    ide  = np.argmax(dEs)
    id2e = np.argmax(d2Es)

    if (ide not in ind_) and (ide+1 not in ind_) and (ide-1 not in ind_): 
       ind_.append(ide)
    if id2e not in ind_ and (id2e+1 not in ind_) and (id2e-1 not in ind_): 
       ind_.append(id2e)

    ind_.sort()

    LabelDataLog = '         AtomicConfigure   E(ML)  E(DFT)   Diff    dE   d2E   \n'
    LabelDataLog+= '      --------------------------------------------------------\n'

    for i in ind_:
        atoms = images[i]
        e_    = atoms.get_potential_energy()
        dE_   = dEs[i]
        d2E_  = d2Es[i]

        if dft=='siesta':
           atoms_= single_point_siesta(atoms,xcf=xcf,xca=xca,basistype=basistype,cpu=cpu)
        elif dft=='qe':
           atoms_= single_point_qe(atoms,kpts=kpts,cpu=cpu)
        else:
           raise RuntimeError('-  This method not implimented!')
        e     = atoms_.get_potential_energy()
        E.append(e)
        E_.append(e_)

        diff_ = abs(e-e_)
        LabelDataLog += '     {:3d}   {:9.5f}  {:9.5f}  {:6.6f}  {:5.4f}   {:5.4f}\n'.format(i,
                         e_,e,diff_,dE_,d2E_)
        with open('SinglePointEnergies.log','a') as fs:
             fs.write('%d MLP: %9.5f DFT: %9.5f Diff: %6.6f dE: %5.4f d2E: %5.4f\n' %(i,
                      e_,e,diff_,dE_,d2E_))

        if diff_>EngTole:            #  or i==ind_[-1]
           his.write(atoms=atoms_)

    his.close()
    images = None
    dEmax  = dEs[ide]
    d2Emax = d2Es[id2e]
    return E,E_,dEmax,d2Emax,LabelDataLog

