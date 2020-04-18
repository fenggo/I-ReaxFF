from __future__ import print_function
from os import getcwd
import numpy as np
from ase.calculators.calculator import Calculator
from .reax import ReaxFF


class IRFF(Calculator, object):
  '''
     Atomistic Machine-Learning Potential ASE calculator
  '''
  implemented_properties = ['energy', 'forces']

  def __init__(self, label='simulation_box',atoms=None):
      Calculator.__init__(self, label=label, atoms=atoms)
      cwd = getcwd()
      self.model = ReaxFF(libfile='ffield',
                          direcs={label:cwd},dft='ase',
                          opt=[],optword='all',
                          batch_size=1,
                          atomic=True,
                          clip_op=False,
                          interactive=True,
                          to_train=False) 
      self.model.initialize()
      self.model.session(learning_rate=1.0e-10,method='AdamOptimizer')


  def calculate(self, atoms, properties, system_changes):
      """Calculation of the energy of system and forces of all atoms."""
      # The inherited method below just sets the atoms object,
      # if specified, to self.atoms.
      Calculator.calculate(self, atoms, properties, system_changes)

      if properties == ['energy']:
         energy = self.model.calculate_energy(atoms)
         self.results['energy'] = energy[0]

      if properties == ['forces']:
         forces = self.model.calculate_forces(atoms)
         self.results['forces'] = forces[0]
      

