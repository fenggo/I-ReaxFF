from __future__ import print_function
from os.path import exists,isfile
from os import system,getcwd,chdir
from ase.io import read,write
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt
from math import ceil
from .molecule import packmol
from .dft.siesta import siesta_md,SinglePointEnergies
from .md.gmd import nvt_wt as gulp_nvt
from .dft.mdtodata import MDtoData
from .dft.prep_data import prep_data
from .reaxfflib import read_lib
import tensorflow as tf
from .md.irmd import IRMD
from .training import train
from .training import train_mpnn
from .AtomOP import AtomOP
from .dingtalk import send_msg
import numpy as np
import json as js


def plot_energies(it,edft,eamp,label_dft='SIESTA',label_ml='IRFF'):
    plt.figure()             # test
    plt.ylabel('Energies (eV)')
    plt.xlabel('Iterations')

    plt.plot(it,edft,linestyle='-',marker='o',markerfacecolor='snow',
             markeredgewidth=1,markeredgecolor='k',
             ms=5,color='k',alpha=0.01,label=label_dft)

    plt.plot(it,eamp,linestyle='-',marker='^',markerfacecolor='snow',
             markeredgewidth=1,markeredgecolor='r',
             ms=5,color='r',alpha=0.01,label=label_ml)

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('energies.eps') # transparent=True
    plt.close() 


def get_zpe(p,atom_name):
    zpe_ = 0.0
    for atom in atom_name:
        zpe_ += p['atomic_'+atom]
    return zpe_
    

class LearningMachine(object):
  ''' recursive learning machine constructed by I-ReaxFF and siesta '''
  def __init__(self,direcs=None,
               ncpu=40,
               maxiter=1000,
               dt_mlmd=0.1,
               dt_aimd=0.3,
               max_batch=10,
               dft_step=5,
               batch=50,
               step=20000,
               md_step=20,
               model='irff',
               mpopt=3,
               massages=2,
               T=300,
               Tmax=10000,
               label=None,
               covergence=0.1,
               accCriteria=0.95,
               accMin=0.91,
               accMax=0.96,
               accInc=1.0,
               resetAcc=5,
               learnWay=1,
               FreeAtoms=None,
               colFrame=25,
               mdInc=1.2,
               rodic=None,
               rtole=0.46,
               checkLone=2,
               nn=True,
               bo_layer=[8,1]):
      ''' max_batch: max number of batch 
              batch: batch size
      '''
      self.accCriteria  = accCriteria # accuracy covergence criteria
      self.accMin       = accMin      # Minimum accuracy covergence after a training process
      self.accMax       = accMax 
      self.accInc       = accInc      # accuracy inrease factor
      self.resetAcc     = resetAcc    # reset accuracy criteria after this train cycle
      self.rtole        = rtole       # minimum bond length used to supervise ML-MD simulation
      self.covergence   = covergence
      self.learnWay     = learnWay    # the way of learing
      self.ncpu         = ncpu
      self.step         = step
      self.batch        = batch
      self.direcs       = direcs
      self.maxiter      = maxiter
      self.max_batch    = max_batch   # max number in direcs to train
      self.dft_step     = dft_step
      self.dt_mlmd      = dt_mlmd
      self.dt_aimd      = dt_aimd
      self.T            = T
      self.Tmax         = Tmax
      self.md_step      = md_step
      self.colFrame     = colFrame
      self.model        = model
      self.mpopt        = mpopt
      self.massages     = massages
      self.mdInc        = mdInc       # MD step increase factor
      self.nn           = nn
      self.bo_layer     = bo_layer
      self.checkLone    = checkLone
      self.FreeAtoms    = FreeAtoms
      self.get_ro(rodic)
      self.c = AtomOP(rtole=self.rtole)
      
      if self.mpopt==1:
         self.massages  = 1
      elif self.mpopt==2 or self.mpopt==3:
         self.massages  = 2

      if self.model=='irff':
         self.trainer = train
      elif self.model=='mpnn':
         self.trainer = train_mpnn
      
      if label is None:
         self.label     = self.c.label
      else:
         self.label     = label

      with open('learning.log','w') as l:
           print('------------------------------------------------------------------------',file=l)
           print('-                                                                      -',file=l)
           print('-            On-the-Fly Learning-Machine by SIESTA and IRFF            -',file=l)
           print('-                          Author: Feng Guo                            -',file=l)
           print('-                     Email: gfeng.alan@gmail.com                      -',file=l)
           print('-                  https://github.com/fenggo/I-ReaxFF                  -',file=l)
           print('-                                                                      -',file=l)
           print('-    Please Cite: Computational Materials Science 172 (2020) 109393    -',file=l)
           print('-                                                                      -',file=l)
           print('------------------------------------------------------------------------',file=l)
           print('\n',file=l)


  def run(self):
      ''' recursive training loop '''
      it       = []
      e_gulp   = []
      e_siesta = []
      cwd      = getcwd()
      gen      = 'poscar.gen'

      if self.direcs is None:
         self.direcs = {}
      if not exists('aimd'):
         system('mkdir aimd')

      i        = 0
      step     = 0
      data_dir = {}
      running  = True
      self.save_config()

      while running:
          run_dir = 'aimd/'+self.label+'-'+str(i)
          if self.learnWay<=2:
             data_dir[self.label+'-'+str(i)] = cwd+'/'+run_dir
          elif self.learnWay == 3:
             data_dir[self.label+'-'+str(i)] = cwd+'/'+run_dir+'/'+self.label+'.traj'

          if exists(run_dir):
             i += 1
             continue
          
          if self.learnWay<=2:
             gen_   = 'md.traj' if isfile('md.traj') else 'poscar.gen'
          elif self.learnWay == 3:
             gen_   = 'poscar.gen'

          atoms  = read(gen_,index=-1)
          atoms  = self.c.check(atoms=atoms,
                                wcheck=self.checkLone,
                                i=i,
                                rtole=self.rtole)
          atoms.write('poscar.gen')
          
          e_aimd = self.aimd(cwd,run_dir,self.dft_step,gen)
          e_siesta.append(e_aimd)                                      # get siesta results              
                                                                       # start training  
          trajs_ = prep_data(label=self.label,direcs=data_dir,
                             split_batch=self.batch,max_batch=self.max_batch,
                             frame=10000000)                           # get trajs for training
          if isfile('training.log'):                                   # 
             system('rm training.log')                                 #
          trajs_.update(self.direcs)                                   #
          
          tf.compat.v1.disable_eager_execution()                       # training untill
          accu = -1000.0                                               # acc >= accMin
          tIter= 1                                    
          while accu<=self.accMin and running:        
                self.load_config()
                loss,accu,accMax,p,zpe,tstep = self.trainer(direcs=trajs_,
                                                     step=self.step,
                                                     batch=self.batch,
                                                     convergence=self.accCriteria,
                                                     nn=self.nn,
                                                     mpopt=self.mpopt,
                                                     bo_layer=self.bo_layer)
                self.load_config()
                if self.accMin>self.accCriteria-0.01:
                   self.accMin = self.accCriteria-0.01 
                if tIter==self.resetAcc and tstep==self.step:
                   self.accCriteria = float(accMax)
                   self.accMin      = float(accMax - 0.05)
                   self.save_config()
                tIter += 1

          system('cp ffield.json ffield_iter%s.json' %str(i))        
          
          if accu >= self.accCriteria and tstep<self.step and tIter<=2:# get md steps
             self.md_step = int(np.ceil(self.md_step*self.mdInc))      # 
             if self.accCriteria<self.accMax:                          # accracy adjusted dynamicly
                self.accCriteria = self.accCriteria*self.accInc        #
             if self.accMin<self.accCriteria-0.01:                     #
                self.accMin = self.accMin*self.accInc                  #
          elif accu < self.accCriteria:                                # adjust MD steps according 
             if self.md_step>1:                                        # training results
                self.md_step  = int(self.md_step/self.mdInc)           #  
          self.save_config()

          e_gmd,mdsteps = self.mlmd(cwd,run_dir,p,gen,self.Tmax,i)       # run ML-MD test training results
          e_gulp.append(e_gmd[0])                                           
          step += self.md_step

          diff  = abs(e_gmd[0]-e_aimd)
          it.append(i)

          plot_energies(it,e_siesta,e_gulp)                           # plot learning status 
                                                                      # 
          with open('energies.log','a') as fe:                        # write learning status 
               for i_,e_ in enumerate(e_gmd):                         # to file
                   if i_==0:                                          #
                      fe.write(' %d %f %f diff: %f \n' %(step+i_,e_gmd[i_],e_aimd,diff))
                   else:                                              #
                      fe.write(' %d %f \n' %(step+i_,e_gmd[i_]))      #
          
          with open('learning.log','a') as l:
               l.write('Iter: %d loss: %f acc: %f mds: %d Eirff: %f Esiesta: %f\n' %(i,loss,accu,mdsteps,e_gmd[i_],e_aimd))

          if diff<self.covergence: 
             print('-  Convergence occured.')
             send_msg('-  Convergence occured.')
             return diff
          if i>self.maxiter:
             print('-  Max iteration reached,the loss %7.4f and accuracy %7.4f.' %(loss,accu))
             send_msg('-  Max iteration reached, the loss %7.4f and accuracy %7.4f.' %(loss,accu))
             return diff
          i += 1


  def aimd(self,cwd,run_dir,tstep,gen):
      system('mkdir '+run_dir)
      chdir(cwd+'/'+run_dir)
      system('cp ../../*.psf ./')

      if self.learnWay<=2:
         images = siesta_md(ncpu=self.ncpu,T=self.T,dt=self.dt_aimd,us='F',tstep=tstep,
                            gen=cwd+'/'+gen,FreeAtoms=self.FreeAtoms)
         eaimd = images[0].get_potential_energy()
      elif self.learnWay==3:
         system('cp ../../md.traj ./')
         system('cp ../../ffield.json ./')
         E = SinglePointEnergies('stretch.traj',label=self.label,frame=self.colFrame)
         eaimd = E[0]
      chdir(cwd)
      return eaimd


  def mlmd(self,cwd,run_dir,p,gen,Tmax,Iter):        # run classic MD to test training results
      if gen.endswith('.traj'):
         images = Trajectory('md.traj')
         if len(images)<=1:
            gen = cwd+'/'+run_dir+'/siesta.traj'
            
      if self.model=='gulp':
         system('./r2l<ffield>reax.lib') 
         gulp_nvt(T=self.T,time_step=self.dt_mlmd,tot_step=self.md_step,gen=gen,
                  mode='w',wt=10)
         atoms_gmd = read(gen,index=-1)
         atom_name = atoms_gmd.get_chemical_symbols()
         zpe_      = get_zpe(p,atom_name)
         egmd      = atoms_gmd.get_potential_energy()-zpe_   # run gulp test training results
      else:
         tf.compat.v1.enable_eager_execution()
         irmd = IRMD(time_step=self.dt_mlmd,totstep=self.md_step,gen=gen,Tmax=Tmax,
                     ro=self.ro,rtole=self.rtole,Iter=Iter,intT=self.T,
                     massages=self.massages)
         if self.learnWay==1:
            irmd.run()
         if self.learnWay==2:
            irmd.opt()
         mdsteps= irmd.step
         Emd  = irmd.Epot
         irmd.close()
      return Emd,mdsteps


  def get_ro(self,rodic):
      if rodic is None:
         self.rodic= {'C-C':1.35,'C-H':1.05,'C-N':1.45,'C-O':1.35,
                      'N-N':1.35,'N-H':1.05,'N-O':1.30,
                      'O-O':1.35,'O-H':1.05,
                      'H-H':0.8,
                      'others':1.35} 
      else:
      	 self.rodic= rodic

      atoms     = read('poscar.gen')
      self.atom_name = atoms.get_chemical_symbols()
      self.natom     = len(self.atom_name)
      self.ro   = np.zeros([self.natom,self.natom])

      for i in range(self.natom):
          for j in range(self.natom):
              bd  = self.atom_name[i] + '-' + self.atom_name[j]
              bdr = self.atom_name[j] + '-' + self.atom_name[i]
              if bd in self.rodic:
                 self.ro[i][j] = self.rodic[bd]
              elif bdr in self.rodic:
                 self.ro[i][j] = self.rodic[bdr]
              else:
                 self.ro[i][j] = self.rodic['others']


  def save_config(self):
      with open('input.json','w') as fj:
           InPut = {'accCriteria':self.accCriteria,
                    'accMin':self.accMin,
                    'accMax':self.accMax,
                    'accInc':self.accInc,
                    'resetAcc':self.resetAcc,
                    'step':self.step,
                    'dft_step':self.dft_step,
                    'md_step':self.md_step,
                    'mdInc':self.mdInc,
                    'dt_aimd':self.dt_aimd,
                    'dt_mlmd':self.dt_mlmd,
                    'rtole':self.rtole,
                    'Tmax':self.Tmax,
                    'maxiter':self.maxiter,
                    'max_batch':self.max_batch,
                    'checkLone':self.checkLone,
                    'mpopt':self.mpopt,
                    'learnWay':self.learnWay,
                    'colFrame':self.colFrame,
                    'FreeAtoms':self.FreeAtoms}
           js.dump(InPut,fj,sort_keys=True,indent=2)


  def load_config(self):
      with open('input.json','r') as fj:
           InPut = js.load(fj)
           self.accCriteria  = InPut['accCriteria']
           self.accMin       = InPut['accMin']
           self.accMax       = InPut['accMax']
           self.accInc       = InPut['accInc']
           self.resetAcc     = InPut['resetAcc']
           self.step         = InPut['step'] 
           self.dft_step     = InPut['dft_step'] 
           self.md_step      = InPut['md_step'] 
           self.mdInc        = InPut['mdInc'] 
           self.dt_aimd      = InPut['dt_aimd'] 
           self.dt_mlmd      = InPut['dt_mlmd'] 
           self.rtole        = InPut['rtole'] 
           self.Tmax         = InPut['Tmax'] 
           self.maxiter      = InPut['maxiter']
           self.max_batch    = InPut['max_batch']
           self.checkLone    = InPut['checkLone']
           self.mpopt        = InPut['mpopt']
           self.learnWay     = InPut['learnWay']
           self.colFrame     = InPut['colFrame']
           self.FreeAtoms    = InPut['FreeAtoms']


  def close(self):
      print('-  LM compeleted.')
      self.atom_name = None
      self.ro        = None



if __name__ == '__main__':
   direcs = {'nm6_5':'/home/feng/siesta/nm6_5',
             'nm6_14':'/home/feng/siesta/nm6_14' }

   lm = LearningMachine(direcs=direcs,
                        ncpu=4,
                        maxiter=2000,
                        dt_mlmd=0.1,
                        dt_aimd=1.0,
                        max_batch=20,dft_step=5,
                        batch=50,step=50000,
                        md_step=50,
                        T=300,
                        Tmax=10000,
                        label='case',
                        covergence=0.01,
                        accCriteria=0.95)
   lm.run()
   lm.close()


