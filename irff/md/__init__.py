from ase.optimize import BFGS,QuasiNewton
from ase.vibrations import Vibrations
from ase.io import read,write
from ..irff import IRFF
from .irmd import IRMD
from ..plot import view
from ase.io.trajectory import Trajectory
import numpy as np


def opt(atoms=None,gen='poscar.gen',fmax=0.3,step=100,v=True):
    if atoms is None:
       atoms = read(gen)
    atoms.calc = IRFF(atoms=atoms,libfile='ffield.json',rcut=None,nn=True)

    def check(atoms=atoms):
        epot_      = atoms.get_potential_energy()
        r          = atoms.calc.r.numpy()
        i_         = np.where(np.logical_and(r<0.5,r>0.0001))
        n          = len(i_[0])

        try:
           assert not np.isnan(epot_), '-  Energy is NaN!'
        except:
           atoms.write('poscarN.gen')
           raise ValueError('-  Energy is NaN!' )

    optimizer = BFGS(atoms,trajectory="opt.traj")
    optimizer.attach(check,interval=1)
    optimizer.run(fmax,step)
    if v:
       images = Trajectory('opt.traj')
       view(images[-1])
    return images[-1]


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


def md(atoms=None,gen='poscar.gen',step=100,model='mpnn',massages=1,
       intT=300,timeStep=0.1,v=True):
    irmd = IRMD(time_step=timeStep,totstep=step,atoms=atoms,gen=gen,Tmax=10000,
                ro=0.8,rtole=0.5,intT=intT,
                massages=massages)
    irmd.run()

    irmd.close()
    del irmd
    images = Trajectory('md.traj')
    if v:
       view(images[-1]) # ,viewer='x3d'
    return images[-1]

