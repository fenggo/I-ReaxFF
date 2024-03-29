from os.path import isfile
import time
import random
# import statistics
from bisect import bisect_left
from enum import Enum
from math import exp
import pickle
import numpy as np
# from ..reaxfflib import read_lib,write_lib
# import json as js


class Strategies(Enum):
      Create       = 'Create'
      Mutate       = 'Mutate'
      Crossover    = 'Crossover'
      CustomMutate = 'CustomMutate'

class logger(object):
  """ logging """
  def __init__(self,flog='ga.log'):
      self.flog = flog
      self.Lookup = {Strategies.Create:   'create   ',
                     Strategies.Mutate:   'mutate   ',
                     Strategies.Crossover:'crossover',
                     Strategies.CustomMutate: 'CustomMutate'}

  def generation(self,candidate,geneName,t,generation=0,msg='-  Generation:'): 
      msg_ = 'Generation: %4d %s loss: %2.3f time: %2.2f' %(generation,
                                self.Lookup[candidate.Strategy],
                                candidate.Fitness,t)
      print('\n---------------------------------------------------------------')
      print('-                       Genetic Algriom                       -')
      print('-                       ---------------                       -')
      print('-  Generation: {:4d}                                           -'.format(generation))
      print('-  Best from : {:9s}                                        -'.format(self.Lookup[candidate.Strategy]))
      print('-  Fitness   : {:8.4f}                                          -'.format(candidate.Fitness))
      print('-  Used Time : {:5.2f}                                           -'.format(t))
      print('---------------------------------------------------------------\n')

      flg = open(self.flog,'a')
      print(msg_,file=flg)
      flg.close()

  def operation(self,op,fitness):
      print('-  operation: {0} {1} {2} ...\r'.format(op,'fitness',fitness),end='\r')


class Chromosome:
  def __init__(self,genes,fitness,strategy,Age=0):
      self.Genes     = genes
      self.Fitness   = fitness
      self.Strategy  = strategy
      self.Age       = Age


class genetic:
  ''' a genetic algriom'''
  def __init__(self,targetLen=4,
                    maxAge=500,
                    poolSize=100,
                    nGenes=100,
                    geneSet=None,
                    geneName=None,
                    optimalFitness=None,
                    custom_mutate=None,
                    get_fitness=None,
                    maxSeconds=None):
      self.targetLen      = targetLen
      self.nGenes         = nGenes
      self.maxAge         = maxAge
      if geneSet is None:
         self.geneSet     = list(np.linspace(0.0,1.0,nGenes))
      else:
         self.geneSet     = geneSet
      self.geneName       = geneName
      self.custom_mutate  = custom_mutate
      self.optimalFitness = optimalFitness
      self.get_fitness    = get_fitness
      self.poolSize       = poolSize
      self.logger         = logger('ga.log')
      self.maxSeconds     = maxSeconds
      self.parents        = None
      self.historicalFitnesses = None
      self.strategies()
      print('\n---------------------------------------------------------------')
      print('-                       Genetic Algriom                       -')
      print('-                       ---------------                       -')
      print('-                   By: gfeng.alan@foxmail.com                -')
      print('---------------------------------------------------------------\n')
 
  def CustomMutate(self,parent):
      return self.custom_mutate()

  def Mutate(self,parent):
      childGenes = parent.Genes[:]
      index = random.randrange(0, len(parent.Genes))
      newGene, alternate = random.sample(self.geneSet, 2)
      childGenes[index] = alternate if newGene == childGenes[index] else newGene
      fitness = self.get_fitness(childGenes)
      self.logger.operation('mutate',fitness)
      return Chromosome(childGenes,fitness,Strategies.Mutate)

  def Crossover(self,parent,index,parents):
      parentGenes= parent.Genes
      donorIndex = random.randrange(0, len(parents))
      if donorIndex == index:
         donorIndex = (donorIndex + 1) % len(parents)

      # crossover
      otherParent= parents[donorIndex].Genes
      childGenes = parentGenes[:]

      if len(parentGenes) <= 2 or len(otherParent) < 2:
          return childGenes

      length = random.randint(1,len(parentGenes)-2)
      start  = random.randrange(0,len(parentGenes)-length)
      childGenes[start:start+length] = otherParent[start:start+length]

      if childGenes is None:
          # parent and donor are indistinguishable
          parents[donorIndex] = self.GenerateParent()
          return self.Mutate(parents[index])

      fitness = self.get_fitness(childGenes)
      self.logger.operation('crossover',fitness)
      return Chromosome(childGenes,fitness,Strategies.Crossover)

  def GenerateParent(self):
      genes = []
      while len(genes) < self.targetLen:
            sampleSize = min(self.targetLen-len(genes),len(self.geneSet))
            genes.extend(random.sample(self.geneSet,sampleSize))
      fitness = self.get_fitness(genes)
      self.logger.operation('create',fitness)
      return Chromosome(genes,fitness,Strategies.Create)

  def NewChild(self,parent,index,parents):
      strategy = random.choice(self.usedStrategies)
      return self.strategyLookup[strategy](parent,index,parents)

  def strategies(self):
      self.strategyLookup = {
            Strategies.Create: lambda p,i,o: self.GenerateParent(),
            Strategies.Mutate: lambda p,i,o: self.Mutate(p),
            Strategies.Crossover: lambda p,i,o: self.Crossover(p,i,o),
            Strategies.CustomMutate: lambda p,i,o: self.CustomMutate(p)}
      self.usedStrategies = [Strategies.Mutate,Strategies.Crossover]
      if not self.custom_mutate is None:
         self.usedStrategies.append(Strategies.CustomMutate)

  def evolve(self):
      ''' find best population '''
      if isfile('CurrentPOP.pkl'):
         f = open('CurrentPOP.pkl', 'rb') 
         self.parents = pickle.load(f)   
         f.close()   

         ft = []
         for chrom in self.parents: # recalculate fitness
             fitness = self.GetFitness(chrom.Genes)
             ft.append(fitness)
             
         self.historicalFitnesses = sorted(ft,reverse=True) 
         self.bestID = np.argmin(ft) 

      g = 0
      self.startTime  = time.time()
      for timedOut,child in self.to_evolve():
          g  += 1
          if timedOut:
             return child
          t = time.time()-self.startTime
          self.logger.generation(child,self.geneName,t,generation=g)
          self.usedStrategies.append(child.Strategy)

          f = open('CurrentPOP.pkl', 'wb')  # save population to file
          pickle.dump(self.parents,f)
          f.close()
          
          if self.optimalFitness>=child.Fitness:
             return child

  def to_evolve(self):
      self.startTime  = time.time()
 
      if self.parents is None:
         bestParent = self.GenerateParent()
         self.parents = [bestParent]
         self.historicalFitnesses = [bestParent.Fitness]
         yield False,bestParent
      else:
           bestParent = self.parents[self.bestID]
      curPool = min(len(self.parents),self.poolSize)

      for i in range(curPool,self.poolSize):
          parent = self.GenerateParent()
          if self.maxSeconds is not None and time.time()-self.startTime>self.maxSeconds:
             yield True,parent
          if parent.Fitness<bestParent.Fitness:
             yield False,parent
             bestParent = parent
             self.historicalFitnesses.append(parent.Fitness)
          self.parents.append(parent)

      lastParentIndex = self.poolSize-1
      self.pindex     = 1

      while True:
            if self.maxSeconds is not None and time.time()-self.startTime>self.maxSeconds:
               yield True,bestParent

            self.pindex = self.pindex-1 if self.pindex>0 else lastParentIndex
            parent = self.parents[self.pindex]
            child  = self.NewChild(parent,self.pindex,self.parents)

            if parent.Fitness < child.Fitness:
               if self.maxAge is None:
                  continue
               parent.Age += 1
               if self.maxAge > parent.Age:
                  continue

               index = bisect_left(self.historicalFitnesses,child.Fitness,0,
                                   len(self.historicalFitnesses))
               proportionSimilar = index/len(self.historicalFitnesses)

               if random.random() < exp(-proportionSimilar):
                  self.parents[self.pindex] = child
                  continue
               bestParent.Age = 0
               self.parents[self.pindex] = bestParent
               continue

            if child.Fitness>=parent.Fitness:
               # same fitness
               child.Age = parent.Age + 1
               self.parents[self.pindex] = child
               continue

            child.Age = 0
            self.parents[self.pindex] = child
            if child.Fitness < bestParent.Fitness:
               bestParent = child
               yield False, bestParent
               self.historicalFitnesses.append(bestParent.Fitness)


