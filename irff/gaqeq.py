from __future__ import print_function
from os.path import isfile
import numpy as np
from .genetic import genetic
from .qeq import qeq
from .mdtodata import MDtoData
from .reaxfflib import read_lib,write_lib
import pickle
import time


class logger(object):
  """docstring for lo"""
  def __init__(self,flog='training.log'):
      self.flog = flog

  def info(self,msg): 
      print(msg)
      flg = open(self.flog,'a')
      print(msg,file=flg)
      flg.close()


class ConvergenceOccurred(Exception):
  """ Kludge to decide when scipy's optimizers are complete.
  """
  pass



class train_qeq(object):
  ''' 
     charges calculate for ReaxFF potential
  '''
  def __init__(self,libfile=None,direcs={},
               rcutoff=10.0,dft='siesta',
               batch=50,
               itmax=50,tol=1.0e-8,
               plr=True,
               board=False,
               to_train=True,
               convergence=1.0):
      self.board    = board
      self.to_train = to_train
      self.direcs   = direcs
      self.libfile  = libfile
      self.dft      = dft
      self.batch    = batch
      self.lib_best = self.libfile+'_best'
      p,zpe,self.spec,self.bonds,self.offd,self.angs,self.torp,self.hbs=read_lib(libfile=libfile)
      self.p_       = p
      self.batch    = batch
      self.itmax    = itmax
      self.plr      = plr
      self.time     = time.time()
      self.logger   = logger('trainq.log')
      self.convergence = convergence
      self.loss_best= 1.0e9
      self.initialize()


  def initialize(self,libfile=None):
      if not libfile is None:
         self.p_,zpe,spec,bonds,offd,angs,torp,hbs=read_lib(libfile=self.lib_best)
      self.p,self.v = {},{}
      for key in self.p_:
          kpre = key.split('_')[0]
          if kpre in ['chi','mu','gamma']:
             print('-  Initilize variable: %s = %f ...' %(key,self.p_[key]))
             self.p[key] = self.p_[key]

      self.Q,self.d,self.images,self.qs = {},{},{},{}
      for mol in self.direcs: 
          self.d[mol] = MDtoData(structure=mol,dft=self.dft,
                                 direc=self.direcs[mol],batch=self.batch)

          self.images[mol] = self.d[mol].get_images()
          self.qs[mol]     = self.d[mol].qs
          self.Q[mol]      = qeq(p=self.p,atoms=self.images[mol][0],
                                 itmax=self.itmax)


  def get_loss(self):
      self.loss = {}
      self.Loss = 0.0
      for mol in self.direcs: 
          q_,qtot = [],[]
          for atoms in self.images[mol]:
              self.Q[mol].p = self.p
              self.Q[mol].calc(atoms)
              q_.append(self.Q[mol].q[:-1])
              qtot.append(self.Q[mol].q[-1])

          self.loss[mol] = np.sqrt(np.sum(np.square(np.array(q_)-self.qs[mol])))
          self.Loss     += np.sum(np.square(qtot))
          self.Loss     += self.loss[mol]
      return self.Loss
 

  def GetFitness(self,genes,LB=False):
      fdic = {}
      for i,v in enumerate(self.geneName):
          if not LB:
             self.p[v] = float(genes[i]*self.geneFc[i]+self.geneSft[i])
   
      Fitness_ = self.get_loss()
      Fitness  = 1.0e9 if np.isnan(Fitness_) else Fitness_
      return Fitness


  def genetical(self,maxAge=20,poolSize=40,nGenes=500,
                optimalFitness=1.0,maxSeconds=None,
                Fc ={'chi':13.0,'mu':13.0,'gamma':2.0},
                zeroSft=True):
      ''' Genetic Algriom '''
      self.geneFc        = []
      self.geneSft       = []
      self.geneName      = []
      self.lib_best      = self.libfile+'_GAbest'

      for v in ['chi','mu','gamma']:
          for sp in self.spec:
              vn   = v+'_'+sp
              self.geneName.append(vn)
              self.geneFc.append(Fc[v])
              if not zeroSft:
                 sft_ = self.p_[vn]-0.5*Fc[v] 
                 sft  = sft_ if sft_>0 else 0.0 
              else:
                 sft = 0.0001
              self.geneSft.append(sft)

      self.Loss_Best = self.GetFitness(self.geneFc,LB=True)

      self.ga = genetic(targetLen=len(self.geneName),
                        maxAge=maxAge,
                        poolSize=poolSize,
                        nGenes=100,
                        optimalFitness=optimalFitness,
                        geneName=self.geneName,
                        get_fitness=self.GetFitness,
                        maxSeconds=maxSeconds)


  def evolve(self):
      ''' find best population '''
      if isfile('CurrentPOP.pkl'):
         f = open('CurrentPOP.pkl', 'rb') 
         self.ga.parents = pickle.load(f)   
         f.close()   
         
         ft = []
         # print(self.ga.parents)
         for chrom in self.ga.parents: # recalculate fitness
             fitness = self.GetFitness(chrom.Genes)
             ft.append(fitness)
             
         self.ga.historicalFitnesses = sorted(ft,reverse=True) 
         self.ga.bestID = np.argmin(ft) 

      self.g = 0
      startTime  = time.time()
      for timedOut,chrom in self.ga.to_evolve():
          self.g  += 1
          t = time.time()-startTime
          self.ga.logger.generation(chrom,self.ga.geneName,t,generation=g)
          self.ga.usedStrategies.append(chrom.Strategy)

          f = open('CurrentPOP.pkl', 'wb')  # save population to file
          pickle.dump(self.ga.parents,f)
          f.close()

          self.geneToLib(chrom,libfile='ffiel_best')
          if self.ga.optimalFitness>=chrom.Fitness:
             return chrom
             self.geneToLib(chrom)


  def geneToLib(self,chromosome,libfile='ffieldGAbest'):
      genes = chromosome.Genes
      for i,v in enumerate(self.geneName):
          self.p_[v] = float(genes[i]*self.geneFc[i]+self.geneSft[i])
      write_lib(self.p_,self.spec,self.bonds,self.offd,
                self.angs,self.torp,self.hbs,
                libfile=libfile,
                loss=chromosome.Fitness,
                logo='!-ReaxFF-From-Genetic-Algorithms-Generation-%s' %self.g)


  def close(self):
      self.p_ = None



