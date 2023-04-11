from ase.optimize import BFGS,QuasiNewton,FIRE
from ase.optimize.basin import BasinHopping
from ase.vibrations import Vibrations
from ase.units import kB
from ase.io import read,write
from ..irff import IRFF
# using GULP instead of IRFF https://wiki.fysik.dtu.dk/ase/ase/calculators/gulp.html
from .irmd import IRMD
from ..plot import view
from ase.io.trajectory import Trajectory
import numpy as np


def opt(atoms=None,gen='poscar.gen',fmax=0.3,step=100,
        optimizer=BFGS,
        vdwnn=False,nn=True):
    if atoms is None:
       atoms = read(gen)
    atoms.calc = IRFF(atoms=atoms,libfile='ffield.json',rcut=None,
                      nn=nn,vdwnn=vdwnn)

    def check(atoms=atoms):
        epot_      = atoms.get_potential_energy()
        r          = atoms.calc.r.detach().numpy()
        i_         = np.where(np.logical_and(r<0.5,r>0.0001))
        n          = len(i_[0])

        try:
           assert not np.isnan(epot_), '-  Energy is NaN!'
        except:
           atoms.write('poscarN.gen')
           raise ValueError('-  Energy is NaN!' )

    optimizer_ = optimizer(atoms,trajectory="opt.traj")
    optimizer_.attach(check,interval=1)
    optimizer_.run(fmax,step)
    images     = Trajectory('opt.traj')
    # view(images[-1])
    return images


def bhopt(atoms=None,gen='poscar.gen',fmax=0.3,step=100,dr=0.5,temperature=100,
          optimizer=BFGS,
          vdwnn=False,nn=True,v=False):
    if atoms is None:
       atoms = read(gen)
    atoms.calc = IRFF(atoms=atoms,libfile='ffield.json',rcut=None,
                      nn=nn,vdwnn=vdwnn)

    optimizer = BasinHopping(atoms=atoms,              # the system to optimize
                      temperature=temperature * kB,    # 'temperature' to overcome barriers
                      dr=dr,                           # maximal stepwidth
                      optimizer=optimizer,
                      fmax=fmax,                       # maximal force for the optimizer
                      trajectory="opt.traj")
    optimizer.run(step)
    if v:
       images = Trajectory('opt.traj')
       view(images[-1])
    return atoms


def freq(atoms=None):
    if atoms is None:
       atoms = read('md.traj',index=0)
    atoms.calc = IRFF(atoms=atoms,libfile='ffield.json',rcut=None,nn=True,massage=2)
    # Compute frequencies
    frequencies = Vibrations(atoms, name='freq')
    frequencies.run()
    # Print a summary
    frequencies.summary()
    frequencies.write_dos()

    # Write jmol file if requested
    # if write_jmol:
    frequencies.write_jmol()


def md(atoms=None,gen='poscar.gen',step=100,nn=True,ffield='ffield.json',initT=300,timeStep=0.1,
       vdwnn=False,print_interval=1):
    irmd = IRMD(time_step=timeStep,totstep=step,atoms=atoms,gen=gen,Tmax=10000,
                ro=0.8,rmin=0.5,initT=initT,vdwnn=vdwnn,print_interval=print_interval,
                nn=nn,ffield=ffield)
    irmd.run()

    irmd.close()
    del irmd
    images = Trajectory('md.traj')
   #  if v:
   #     view(images[-1]) # ,viewer='x3d'
    return images

