#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase import Atoms
from irff.dft.SinglePointEnergy import SinglePointEnergies
from irff.irff_np import IRFF_NP


images = Trajectory('gulp.traj')
tframe = len(images)
frame  = 50
E = []
ind   = [i for i in range(tframe)]


ind_     = []
energies = []
dEs      = []
d2Es     = []
dEa      = []
d2Ea     = []
dE       = 0.0
d2E      = 0.0

for i,atoms in enumerate(images):
    energy = atoms.get_potential_energy()
    if i>0 :
       if i<(tframe-1):
          deltEl =  energy - energies[-1]
          deltEr =  images[i+1].get_potential_energy() - energy
          dE     = deltEl
          d2E    = deltEr-deltEl
       else:
          deltEl =  energy - energies[-1]
          dE     = deltEl

       dEa.append(dE)
       d2Ea.append(d2E)

       dEs.append(abs(dE))
       d2Es.append(abs(d2E))

    energies.append(energy)
    print('step:',i,'dE:',dE,'d2E:',d2E)

e_mean = np.mean(energies)
maxDiff = np.max(energies) - np.min(energies)

i = np.argmax(dEs)
i_= np.argmax(d2Es)

j = np.argmin(dEs)
j_= np.argmin(d2Es)

print('* dEmax: ',i,dEs[i],' d2Emax: ',i_,d2Es[i_],
      ' dEmin: ',dEa[j],'d2Emin: ',d2Ea[j_],
      ' maxDiff: ',maxDiff)


if i>0:
   ir = IRFF_NP(atoms=images[i-1],
                libfile='ffield.json',
                nn=True)
   ir.calculate(images[i-1])
   E        = ir.E
   Ebond    = ir.Ebond
   Eang     = ir.Eang
   Eover    = ir.Eover
   Eunder   = ir.Eunder
   Elone    = ir.Elone
   Epen     = ir.Epen
   Etcon    = ir.Etcon
   Efcon    = ir.Efcon
   Etor     = ir.Etor
   Ehb      = ir.Ehb
   Ecoul    = ir.Ecoul
   ir.calculate(images[i])

   dE        = ir.E - E
   dEbond    = ir.Ebond - Ebond
   dEang     = ir.Eang - Eang
   dEpen     = ir.Epen - Epen
   dEtcon    = ir.Etcon - Etcon
   dEfcon    = ir.Efcon - Efcon
   dEtor     = ir.Etor - Etor
   dEhb      = ir.Ehb - Ehb
 
   print('* dEmax: ',dEs[i])
   print('* dE: ',dE)
   print('* dEbond: ',dEbond)
   print('* dEang: ',dEang)
   print('* dEpen: ',dEpen)
   print('* dEtcon: ',dEtcon)
   print('* dEfcon: ',dEfcon)
   print('* dEtor: ',dEtor)
   print('* dEhb: ',dEhb)

# print(' * dEmax: ',np.max(dEs),' d2Emax: ',np.max(d2Es))

# for e in energies:
#     print('energies:',e,'ave:',e_mean,'diff:',abs(e-e_mean))
