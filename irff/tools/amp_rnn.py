#!/usr/bin/env python
"""Simple test of the Amp calculator, using Gaussian descriptors and neural
network model. Randomly generates data with the EMT potential in MD
simulations."""
from os.path import isfile
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import system
from ase.calculators.emt import EMT
from ase.lattice.surface import fcc110
from ase import Atoms, Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms
from ase.io.trajectory import Trajectory

from amp import Amp
from amp.descriptor.gaussian import Gaussian,make_symmetry_functions
# from amp.descriptor.bispectrum import Bispectrum
from amp.model.tflow import NeuralNetwork
# from amp.model.neuralnetwork import NeuralNetwork
# from amp.descriptor.zernike import Zernike
from amp.model import LossFunction
import tensorflow as tf
import numpy as np


class ple:
  def __init__(self):
      self.plt = plt
      self.plt.figure()             # test
      self.plt.ylabel('Energies (eV)')
      self.plt.xlabel('Step')

  def scatter(self,x,edft,eamp,eamp_,dolabel=True):
      if dolabel:
         self.plt.scatter(x,edft,marker='o',edgecolor='k',
                 s=25,c='k',alpha=0.01,label='DFT')
        
         self.plt.scatter(x,eamp_,marker='+',edgecolor='b',
                 s=25,c='b',alpha=0.01,label='AMP(untrained)')
           
         self.plt.scatter(x,eamp,marker='^',edgecolor='k',
                 s=25,c='r',alpha=0.01,label='AMP(trained)')
      else:
         self.plt.scatter(x,edft,marker='o',edgecolor='k',
                 s=25,c='k',alpha=0.01)
        
         self.plt.scatter(x,eamp_,marker='+',edgecolor='b',
                 s=25,c='b',alpha=0.01)
           
         self.plt.scatter(x,eamp,marker='^',edgecolor='k',
                 s=25,c='r',alpha=0.01)  

  def plot(self):
      self.plt.legend(loc='best')
      self.plt.savefig('energies_scatter.eps') 
      self.plt.close() 



def plot_energies(edft,eamp,eamp_=None):
    edft = np.reshape(edft,[-1])
    eamp = np.reshape(eamp,[-1])
    plt.figure()           
    plt.ylabel('Energies (eV)')
    plt.xlabel('Step')

    n   = len(edft)
    x   = np.linspace(0,n,n)
    err = edft-eamp

    max1 = np.max(edft)
    max2 = np.max(eamp)
    max_ = max([max1,max2])

    plt.plot(edft-max_,linestyle='-',marker='o',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='b',
             ms=6,c='b',alpha=0.01,label='DFT')

    if not eamp_ is None:
       plt.plot(x,eamp_-max_,linestyle='-',marker='+',markerfacecolor='none',
                markeredgewidth=1,markeredgecolor='k',
                ms=6,c='r',alpha=0.01,label='AMP(untrained)')

    plt.errorbar(x,eamp-max_,yerr=err,fmt='-o',ecolor='r',color='r',
                 elinewidth=2,capsize=4,label='AMP(trained)')
    plt.text(0.0, 0.0, '%.3f' %max_, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='best')
    plt.savefig('energies.eps') 
    plt.close() 


def get_mean_energy(images):
    energies = []
    for image in images:
        energies.append(image.get_potential_energy())
    energies_ = np.array(energies)
    nimg   = len(images)
    mean_e = np.mean(energies_)
    return nimg,mean_e


def train_rnn(baseframe=100,tframe=8,total_step=10,traj='ethane.traj',
              convergence={'energy_rmse': 0.25,'force_rmse': 0.5},
              elements = ['C','H','O'],
              hiddenlayers=(64,64,64,64,64,64,64,64,64,64,64),
              optim='ADAM',
              cores=4):
    """Gaussian/tflow train test."""
    p = ple()
    label = 'amp'
    all_images = Trajectory(traj)
    nimg,mean_e=get_mean_energy(all_images)

    G = make_symmetry_functions(elements=elements, type='G2',
                etas=np.logspace(np.log10(0.05), np.log10(5.),
                                 num=4))
    G += make_symmetry_functions(elements=elements, type='G5',
                 etas=[0.005],
                 zetas=[1., 4.],
                 gammas=[+1., -1.])
    G = {element: G for element in elements} # Gs=G

    if not isfile('amp.amp'):
       print('\nset up calculator ...\n')
       calc = Amp(descriptor=Gaussian(mode='atom-centered',Gs=G),
                  model=NeuralNetwork(hiddenlayers=hiddenlayers,
                                      convergenceCriteria=convergence,
                                      activation='tanh',
                                      energy_coefficient=1.0,
                                      force_coefficient=None,
                                      optimizationMethod=optim,
                                      parameters={'energyMeanScale':mean_e},
                                      maxTrainingEpochs=100000), 
                  label=label,
                  cores=cores)              # 'l-BFGS-b' or 'ADAM'
       trained_images = [all_images[j] for j in range(0,baseframe)]
       calc.train(overwrite=True,images=trained_images)
       del calc
    else:
       calc =Amp.load('amp.amp')
       calc.model.parameters['convergence'] = convergence
       calc.model.lossfunction = LossFunction(convergence=convergence)
       trained_images = [all_images[j] for j in range(0,baseframe)]
       calc.train(overwrite=True,images=trained_images)
       del calc


    tstep = int((nimg - baseframe)/tframe)
    if total_step>tstep:
       total_step = tstep 
    print('Max train cycle of %d is allowed.' %total_step)

    edfts,eamps,eamps_ = [],[],[]
    dolabel=True
    basestep = int(baseframe/tframe)

    for step in range(basestep,total_step+basestep):
        new_images = [all_images[j] for j in range(0+step*tframe,tframe+step*tframe)]
        trained_images.extend(new_images)

        x,edft,eamp,eamp_ = [],[],[],[]
        ii = step*tframe

        # -----    test     -----
        calc1 =Amp.load('amp.amp')
        for i,image in enumerate(new_images):
            x.append(ii)
            eamp_.append(calc1.get_potential_energy(image))
            eamps_.append(eamp_[-1])
            edft.append(image.get_potential_energy())
            edfts.append(edft[-1])
            ii += 1
        del calc1

        # -----    train     -----
        calc =Amp.load('amp.amp')
        calc.model.lossfunction = LossFunction(convergence=convergence)
        # calc.model.convergenceCriteria=convergence
        calc.train(overwrite=True,images=trained_images)
        del calc

        # -----    test     -----
        calc2 = Amp.load('amp.amp')
        print('\n----   current training result   ---- \n')
        for i,image in enumerate(new_images):
            eamp.append(calc2.get_potential_energy(image))
            eamps.append(eamp[-1])
            print("energy(AMP) = %f energy(DFT) = %f" %(eamp[-1],edft[i]))
            # print("forces = %s" % str(calc2.get_forces(image)))
        del calc2

        plot_energies(edfts,eamps,eamp_=None)
        system('epstopdf energies.eps')
        
        p.scatter(x,edft,eamp,eamp_,dolabel=dolabel)
        if dolabel:
           dolabel=False

    p.plot()
    system('epstopdf energies_scatter.eps')


def train_amp(baseframe=200,traj='ethane.traj',
              convergence={'energy_rmse': 0.25,'force_rmse': 0.5},
              elements = ['C','H','O'],
              cores=4):
    """Gaussian/tflow train test."""
    p = ple()
    label = 'amp'
    all_images = Trajectory(traj)
    nimg,mean_e=get_mean_energy(all_images)

    G = make_symmetry_functions(elements=elements, type='G2',
                etas=np.logspace(np.log10(0.05), np.log10(5.),
                                 num=4))
    G += make_symmetry_functions(elements=elements, type='G5',
                 etas=[0.005],
                 zetas=[1., 4.],
                 gammas=[+1., -1.])
    G = {element: G for element in elements} # Gs=G

    if not isfile('amp.amp'):
       # print('\nset up calculator ...\n')
       calc = Amp(descriptor=Gaussian(mode='atom-centered',Gs=G),
                  model=NeuralNetwork(hiddenlayers=(1024,1024,1024,512,512,256,256,256,256,128,128),
                                      convergenceCriteria=convergence,
                                      activation='tanh',
                                      energy_coefficient=1.0,
                                      force_coefficient=None,
                                      optimizationMethod='ADAM',
                                      parameters={'energyMeanScale':mean_e},
                                      maxTrainingEpochs=100000), 
                  label=label,
                  cores=cores)              # 'l-BFGS-b' or 'ADAM'
       trained_images = [all_images[j] for j in range(0,baseframe)]
       calc.train(overwrite=True,images=trained_images)
       del calc
    else:
       calc =Amp.load('amp.amp')
       calc.model.parameters['convergence'] = convergence
       calc.model.lossfunction = LossFunction(convergence=convergence)
       trained_images = [all_images[j] for j in range(0,baseframe)]
       calc.train(overwrite=True,images=trained_images)
       del calc

    edfts,eamps,eamps_ = [],[],[]
    dolabel=True
    basestep = int(baseframe/tframe)
        
    system('epstopdf energies.eps')        
    p.scatter(x,edft,eamp,eamp_,dolabel=dolabel)
    p.plot()

    plot_energies(edfts,eamps,eamp_=eamps_)
    system('epstopdf energies_scatter.eps')


if __name__ == '__main__':
    train_rnn(tframe=2,total_step=4,restart=True)


