from os.path import exists,isfile
from os import getcwd,chdir,mkdir,listdir,popen #,system,
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
import matplotlib.pyplot as plt
from math import ceil
from .molecule import packmol
from .dft.siesta import siesta_md 
from .dft.SinglePointEnergy import SinglePointEnergies
from .md.gulp import nvt as gulp_nvt
from .data.mdtodata import MDtoData
from .data.prep_data import prep_data
from .reaxfflib import read_lib
import tensorflow as tf
from .md.irmd import IRMD
from .training import train_reax
from .training import train_mpnn
from .AtomDance import AtomDance,check_zmat
from .dingtalk import send_msg
import numpy as np
import json as js


def plot_energies(it,edft,eamp,label_dft='SIESTA',label_ml='IRFF'):
    plt.figure()                                     # test
    mine = np.min(edft)

    plt.ylabel('Energies + %f (eV)' %(-mine))
    plt.xlabel('Iterations')
    
    plt.plot(edft-mine,linestyle='-',marker='o',markerfacecolor='snow',
             markeredgewidth=1,markeredgecolor='k',
             ms=5,color='k',alpha=0.8,label=label_dft)

    plt.plot(eamp-mine,linestyle='-',marker='^',markerfacecolor='snow',
             markeredgewidth=1,markeredgecolor='r',
             ms=5,color='r',alpha=0.8,label=label_ml)

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('energies.pdf')                      # transparent=True
    plt.close() 


def get_zpe(p,atom_name):
    zpe_ = 0.0
    for atom in atom_name:
        zpe_ += p['atomic_'+atom]
    return zpe_
    

class LearningMachine(object):
  ''' recursive learning machine constructed by I-ReaxFF and siesta '''
  def __init__(self,initConfig='poscar.gen',
               direcs=None,ncpu=40,
               maxiter=1000,
               dt_mlmd=0.1, dt_aimd=0.3,dft_step=5,
               batch=50,max_batch=100,
               step=20000,md_step=20,MaxMDstep=500,colFrame=10,
               mom_step=100,
               mpopt=1, messages=1,
               optword='nocoul',
               T=300,Tmax=1000,
               label=None,
               convergence=0.01,nconvergence=3,
               accCriteria=0.9,lossCriteria=1000.0,lossTole=1.0,accTole=0.005,
               accMax=0.9,accInc=1.0001,mdInc=1.2,resetAcc=1,
               learnWay=5,FreePairs=None,beta=None,
               FreeAtoms=None,FirstAtom=None,
               rodic=None,
               rmin=0.6,rmax=1.3,angmax=15.0,
               CheckZmat=True,
               EngTole=0.05,dEtole=0.2,dEstop=2.0,
               nn=True,vdwnn=False,
               bo_layer=[2,1],mf_layer=[9,2],be_layer=[6,1],vdw_layer=None,#[6,1],
               be_univeral_nn='all',bo_univeral_nn='all',mf_univeral_nn='all',vdw_univeral_nn='all',
               EnergyFunction=1,MessageFunction=1,
               spv_bm=False,bom={'others':1.0},
               spv_be=True,bore={'others':0.2},
               spv_ang=True,lambda_ang=0.02,
               weight={'others':2.0},
               lambda_bd=10000.0,lambda_reg=0.0001,regularize=True,
               lambda_me=0.1,learning_rate=0.0001,
               writelib=5000,ffield='ffield.json',
               xcf='VDW',xca='DRSLL',basistype='DZP'):
      ''' max_batch: max number of batch 
              batch: batch size
      '''
      self.initConfig     = initConfig
      self.label          = initConfig.split('.')[0]
      self.aidir          = 'aimd_' + self.label
      self.accCriteria    = accCriteria # accuracy convergence criteria
      self.lossCriteria   = lossCriteria    
      self.lossTole       = lossTole    
      self.accTole        = accTole      # Minimum accuracy convergence after a training process
      self.accMax         = accMax 
      self.accInc         = accInc      # accuracy inrease factor
      self.resetAcc       = resetAcc    # reset accuracy criteria after this train cycle
      self.rmin           = rmin       # minimum bond length used to supervise ML-MD simulation
      self.rmax           = rmax
      self.angmax         = angmax
      self.CheckZmat      = CheckZmat
      self.convergence    = convergence
      self.nconvergence   = nconvergence
      self.learnWay       = learnWay    # the way of learing
      self.ncpu           = ncpu
      self.step           = step
      self.batch          = batch
      self.direcs         = direcs
      self.maxiter        = maxiter
      self.max_batch      = max_batch   # max number in direcs to train
      self.xcf            = xcf
      self.xca            = xca
      self.basistype      = basistype
      self.dft_step       = dft_step
      self.dt_mlmd        = dt_mlmd
      self.dt_aimd        = dt_aimd
      self.T              = T
      self.Tmax           = Tmax
      self.md_step        = md_step
      self.mom_step       = mom_step
      self.beta           = beta
      self.colFrame       = colFrame
      self.EngTole        = EngTole
      self.dEtole         = dEtole
      self.dEstop         = dEstop
      self.MaxMDstep      = MaxMDstep
      self.mpopt          = mpopt
      self.optword        = optword
      self.messages       = messages
      self.mdInc          = mdInc       # MD step increase factor
      self.ffield         = ffield
      self.nn             = nn
      self.vdwnn          = vdwnn
      if not self.nn:
         self.vdwnn       = False
      self.bo_layer       = bo_layer
      self.mf_layer       = mf_layer
      self.be_layer       = be_layer
      self.vdw_layer      = vdw_layer
      self.bo_univeral_nn = bo_univeral_nn
      self.be_univeral_nn = be_univeral_nn
      self.mf_univeral_nn = mf_univeral_nn
      self.vdw_univeral_nn= vdw_univeral_nn
      self.bore           = bore
      self.bom            = bom
      self.spv_bm         = spv_bm
      self.spv_be         = spv_be
      self.spv_ang        = spv_ang
      self.lambda_me      = lambda_me
      self.weight         = weight
      self.lambda_bd      = lambda_bd
      self.regularize     = regularize
      self.lambda_reg     = lambda_reg
      self.lambda_ang     = lambda_ang
      self.learning_rate  = learning_rate
      self.EnergyFunction = EnergyFunction
      self.MessageFunction= MessageFunction
      self.writelib       = writelib
      self.freeatoms      = FreeAtoms
      self.freepairs      = FreePairs
      self.a              = AtomDance(poscar=self.initConfig,nn=self.nn,
                                    ffield=ffield,
                                    rmin=self.rmin-0.05,rmax=self.rmax,
                                    FirstAtom=FirstAtom,freeatoms=self.freeatoms)
      if self.freepairs is None:
         self.a.get_freebond(freeatoms=self.freeatoms)
      else:
      	 self.a.freebonds = self.freepairs
      	 
      self.natom        = self.a.natom
      self.atom_name    = self.a.atom_name
      self.get_ro(rodic)
      # if self.freeatoms is None:
      #    self.freeatoms = [i for i in range(self.natom)]

      if self.mpopt==1:
         self.messages  = 1
      elif self.mpopt==2 or self.mpopt==3:
         self.messages  = 2

      if self.nn:
         self.trainer = train_mpnn
      else:
         self.trainer = train_reax
      
      if label is None:
         aimd_ = self.aidir.split('_')
         if len(aimd_)>1:
            self.label  = aimd_[-1]
         else:
            self.label  = self.a.label
      else:
         self.label     = label

      if not isfile('learning.log'):
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
      if not exists(self.aidir):
         mkdir(self.aidir)

      iter_        = 0
      step         = 0
      mdsteps      = self.md_step
      momsteps     = self.mom_step
      data_dir     = {}
      running      = True
      Deformed     = 0
      if self.learnWay==5:
         if self.freeatoms is None:
            self.freeatoms = []
         ms        = len(self.freeatoms)
      else:
         ms        = 0

      self.learnpair = None
      groupi,groupj  = None,None
      self.save_config()
      converg        = False
      nconvergence   = 0
      unreasonable_zv,uzv_ = None,None
      zmat_variable  = None
      zmatopt        = None
      zmats          = None
      zvlo,zvhi      = 0.0,0.0
      u_zvlo,u_zvhi  = 0.0,0.0

      while running:
          optlog   = None
          relaxlog = None
          learnWay = self.learnWay
          mom_step = self.mom_step*(ms+1) if learnWay==5 else self.mom_step
          run_dir  = self.aidir+'/'+self.label+'-'+str(iter_)
          data_dir[self.label+'-'+str(iter_)] = cwd+'/'+run_dir+'/'+self.label+'.traj'

          if exists(run_dir):
             iter_ += 1
             continue
          if isfile('md.traj'): 
             atoms = read('md.traj',index=-1)
             # print('-  atomic structure from MD trajectories.')
          else:
             print('-  cannot find MD trajectory, use learnWay=1 in the first iter.')
             atoms = read(self.initConfig)
             learnWay = 1
          if not self.freeatoms is None:
             atoms = self.a.check_momenta(atoms,freeatoms=self.freeatoms)
             
          if learnWay in [4,5,6]:
             momsteps += mdsteps
             if momsteps >= mom_step:
                momsteps  = 0
                if learnWay == 4:
                   atoms,self.learnpair,groupi,groupj = self.a.bond_momenta(atoms)
                   if self.learnpair is None:
                      learnWay = 3
                elif learnWay == 5:
                   if ms<self.natom:
                      i = self.a.zmat_id[ms]
                      if i not in self.freeatoms: 
                         self.freeatoms.append(i)
                         for i_ in self.a.neighbors[i]:
                             if len(self.a.neighbors[i_])==1:
                                if i_ not in self.freeatoms: 
                                   self.freeatoms.append(i_)
                      # if j not in self.freeatoms: 
                      #    self.freeatoms.append(j)
                      #    for j_ in self.a.neighbors[j]:
                      #        if len(self.a.neighbors[j_])==1:
                      #           if j_ not in self.freeatoms: 
                      #              self.freeatoms.append(j_)
                      self.save_config()
                      if ms+1<self.natom:
                         j = self.a.zmat_id[ms+1]
                         # self.learnpair = [i,j]
                         atoms,_,_ = self.a.set_bond_momenta(i,j,atoms)
                      else:
                         learnWay = 3
                         self.learnpair = None
                   if learnWay ==5: learnWay = 4
                # learnWay  = 2
                ms       += 1

          # self.get_fixatom()
          # atoms.set_constraint(self.fixatoms)

          atoms.write('poscar.gen')                                    # for aimd

          if learnWay==1: 
             dft_step = self.dft_step
          else:
             dft_step = int(mdsteps/self.colFrame)+1

          e_aimd,eml_,dEmax_,d2Emax_,LabelDataLog = self.aimd(cwd,run_dir,dft_step,gen,learnWay)
          e_siesta.append(e_aimd[0])                                   # get siesta results              
                                                                       # start training  
          trajs_ = prep_data(label=self.label,direcs=data_dir,
                             split_batch=self.batch,max_batch=self.max_batch,
                             frame=1000,dft='siesta')                  # get trajs for training
          with open('training.log','w') as fe:                         # 
               fe.write('## training loss and accuries\n')             #
          trajs_.update(self.direcs)                                   #

          self.load_config()
          tf.compat.v1.disable_eager_execution()                       # training untill
          accu = -1000.0                                               # acc >= accMin
          tIter= 1   
          loss = self.lossCriteria+0.1      
          training = True                   
          while (accu<=self.accCriteria or loss>=self.lossCriteria) and training:        
                self.load_config()
                loss,accu,accMax,p,zpe,tstep = self.trainer(direcs=trajs_,
                                                     ffield=self.ffield,
                                                     step=self.step,writelib=self.writelib,
                                                     batch=self.batch,
                                                     convergence=self.accCriteria,
                                                     lossConvergence=self.lossCriteria,
                                                     nn=self.nn,vdwnn=self.vdwnn,
                                                     mpopt=self.mpopt,optword=self.optword,
                                                     bo_layer=self.bo_layer,
                                                     mf_layer=self.mf_layer,
                                                     be_layer=self.be_layer,
                                                     vdw_layer=self.vdw_layer,
                                                     bo_univeral_nn=self.bo_univeral_nn,
                                                     be_univeral_nn=self.be_univeral_nn,
                                                     mf_univeral_nn=self.mf_univeral_nn,
                                                     vdw_univeral_nn=self.vdw_univeral_nn,
                                                     EnergyFunction=self.EnergyFunction,
                                                     MessageFunction=self.MessageFunction,
                                                     bore=self.bore, bom=self.bom,
                                                     spv_be=self.spv_be,spv_bm=self.spv_bm,
                                                     spv_ang=self.spv_ang,lambda_ang=self.lambda_ang,
                                                     lambda_me=self.lambda_me,weight=self.weight,
                                                     regularize=self.regularize,lambda_reg=self.lambda_reg,
                                                     lambda_bd=self.lambda_bd,
                                                     learning_rate=self.learning_rate)
                self.load_config()
                if tIter>=self.resetAcc and tstep>=self.step:
                   if accu>self.accCriteria-self.accTole and loss<self.lossCriteria+self.lossTole:
                      self.accCriteria = float(accMax)
                      self.lossCriteria= float(loss)
                      self.save_config()
                      training = False
                tIter += 1

          self.lossCriteria = float(loss)
          if self.CheckZmat:
             if not zmatopt is None:
                optlog = self.a.get_optimal_zv(atoms,zmatopt,self.initConfig)
             if mdsteps>10:
                unreasonable_zv,u_zvlo,u_zvhi = self.a.get_zmat_info(zmats)

          if mdsteps<10: mdsteps = 10
          zmats   = None
          images  = None
          zmatopt = zmat_variable
          if learnWay==6:
             if ms<len(self.freeatoms):
                i = self.freeatoms[ms]
                self.a.pes(i,atoms,nbin=dft_step,dr=0.1,traj='md.traj')
             if ms == len(self.freeatoms) -1:
             	self.learnWay=3
          else:
             if Deformed>=1.0:
                images,relaxlog = self.a.zmat_relax(atoms=atoms,zmat_variable=zmat_variable,nbin=mdsteps,
                                                    zvlo=zvlo,zvhi=zvhi,traj='md.traj')
             elif not unreasonable_zv is None:
                images,relaxlog = self.a.zmat_relax(atoms=atoms,zmat_variable=unreasonable_zv,nbin=mdsteps,
                                                    zvlo=u_zvlo,zvhi=u_zvhi,traj='md.traj')
             else:
                e_gmd,mdsteps,Deformed,zmats,zmat_variable,zvlo,zvhi = self.mlmd(cwd,run_dir,atoms, 10000,
                                                iter_,learnWay,beta=self.beta,learnpair=self.learnpair,
                                                groupi=groupi,groupj=groupj) 
                if self.CheckZmat and mdsteps<=3:
                   mdsteps= 10
                   images,relaxlog = self.a.continous(atoms,nbin=mdsteps,traj='md.traj')
             if not images is None:
                Deformed,zmat,zmat_variable,zvlo,zvhi = check_zmat(atoms=images[-1],rmin=self.rmin,
                                       rmax=self.rmax,angmax=self.angmax,zmat_id=self.a.zmat_id,
                                       zmat_index=self.a.zmat_index,InitZmat=self.a.InitZmat)

          step += mdsteps                                          # run ML-MD test training results     

          if accu >= self.accCriteria and tstep<self.step and tIter<=2: # get md steps
             if self.md_step<self.MaxMDstep and mdsteps>=self.md_step:
                self.md_step = int(np.ceil(self.md_step*self.mdInc))    # 
             if self.accCriteria<self.accMax:                           # accracy adjusted dynamicly
                self.accCriteria = min(self.accCriteria*self.accInc,self.accMax)  
          elif accu < self.accCriteria:                                 # adjust MD steps according 
             if self.md_step>1:                                         # training results
                self.md_step  = int(self.md_step/self.mdInc)            #                 

          if self.md_step>=self.MaxMDstep*0.5:
             if self.T<self.Tmax:
                self.T = self.T*self.mdInc

          self.save_config()

          if learnWay in [3,4,5]: e_gmd = eml_
          e_gulp.append(e_gmd[0])
          diff  = abs(e_gmd[0]-e_aimd[0])
          it.append(iter_)

          plot_energies(it,e_siesta,e_gulp)                             # plot learning status                                   
          self.learninglog(iter_,loss,accu,mdsteps,e_gulp[-1],e_siesta[-1],dEmax_,d2Emax_,LabelDataLog,
          	               learnWay,Deformed,
                           self.a.InitZmat,unreasonable_zv,u_zvlo,u_zvhi,optlog,relaxlog)
          converg_ = converg
          if diff<=self.convergence: 
             converg = True
          else:
             converg = False

          if converg and converg_: 
             nconvergence += 1
          else:
             nconvergence  = 0
          if nconvergence > self.nconvergence:
             print('-  Learning is Convergenced.')
             send_msg('-  Learning is Convergenced.')
             return diff

          if iter_>self.maxiter:
             print('-  Max iteration reached,the loss %7.4f and accuracy %7.4f.' %(loss,accu))
             send_msg('-  Max iteration reached, the loss %7.4f and accuracy %7.4f.' %(loss,accu))
             return diff
          elif loss==-1:
             send_msg('-  Some error occurred!')
             return diff
          iter_ += 1

  def learninglog(self,i,loss,accu,mdsteps,eirff,edft,dEmax,d2Emax,LabelDataLog,
                  learnWay,Deformed,
                  InitZmat,uz,uzlo,uzhi,optlog,relaxlog):
      with open('learning.log','a') as l:
         l.write('                  ------------------------------------                  \n')
         l.write('                       Learning iteration:  {:4d}                        \n'.format(i))
         l.write('                  ------------------------------------                  \n')
         l.write(' \n')
         l.write('                     I&II  Labeling and Calling DFT \n')
         l.write('            Labeled atomic configurations and DFT calculations \n')
         l.write(' \n')
         l.write(LabelDataLog)
         # for d_,d in enumerate(LabeledData):
         #     if d_%10==0:
         #        l.write('\n')
         #     l.write(' {:3d}'.format(d))
         l.write(' \n')
         l.write('                           III  Training \n')
         l.write('                     Train the potential model  \n')
         l.write(' \n')
         l.write('Loss of the model:  {:7.4f}  \n'.format(loss))
         l.write('Accuracy of the potential model:  {:7.4f}  \n'.format(accu))
         l.write('\n')
         l.write('                           IV Exploring \n')
         l.write('      Exploring the atomic configuration space by MD simulations \n')
         l.write('\n')
         l.write('MD Steps:  {:d}  \n'.format(mdsteps))
         l.write('Energy from ML and DFT:  {:f}  .vs.  {:f} \n'.format(eirff,edft))
         l.write('The max first derivative and second derivative:  {:f}  {:f} \n'.format(dEmax,d2Emax))
         l.write('The learning method of this iter:  {:d} \n'.format(learnWay))
         if not relaxlog is None:
            if Deformed>=1:
               l.write('Molecule structure is deformed, using zmatmix relax\n' )
            elif not uz is None:
               l.write('Unreasonable zmatrix variable:\n')
            l.write(relaxlog)
         if not optlog is None:
            l.write(optlog)
         l.write(' \n')

  def aimd(self,cwd,run_dir,tstep,gen,learnWay):
      mkdir(run_dir)
      chdir(cwd+'/'+run_dir)
      popen('cp ../../*.psf ./')
      emlmd  = []
      ind_   = []
      dEmax  = 0.0
      d2Emax = 0.0

      if learnWay==1:
         images = siesta_md(label=self.label,ncpu=self.ncpu,T=self.T,dt=self.dt_aimd,us='F',tstep=tstep,
                            gen=cwd+'/'+gen,FreeAtoms=self.freeatoms,
                            xcf=self.xcf,xca=self.xca,basistype=self.basistype)
         eaimd = [images[0].get_potential_energy()]
         popen('cp %s ../../md.traj' %(self.label+'.traj'))
      elif learnWay>=2:
         popen('cp ../../md.traj ./')
         popen('cp ../../ffield.json ./')
         # print(' * files in this dir \n',listdir())
         E,E_,dEmax,d2Emax,ind_ = SinglePointEnergies('md.traj',label=self.label,EngTole=self.EngTole,
                                                 frame=tstep,select=True,
                                                 dE=self.dEtole, # d2E=self.d2Etole,
                                                 xcf=self.xcf,xca=self.xca,basistype=self.basistype,
                                                 cpu=self.ncpu)
         eaimd = E
         emlmd = E_
      else:
         raise RuntimeError('-  learn way not supported yet!')
      chdir(cwd)
      return eaimd,emlmd,dEmax,d2Emax,ind_

  def mlmd(self,cwd,run_dir,atoms,Tmax,Iter,learnWay,learnpair=None,beta=None,
           groupi=None,groupj=None):        
      ''' run classic MD to test training results '''
      mdstep = max(int(self.md_step/5),2) if learnWay==2 else self.md_step
      irmd = IRMD(atoms=atoms,label=self.label,Iter=Iter,initT=self.T,
                  time_step=self.dt_mlmd,totstep=mdstep,Tmax=Tmax,
                  ro=self.ro,rmin=self.rmin,rmax=self.rmax,angmax=self.angmax,
                  CheckZmat=self.CheckZmat,InitZmat=self.a.InitZmat,
                  zmat_id=self.a.zmat_id,zmat_index=self.a.zmat_index,
                  dEstop=self.dEstop,dEtole=self.dEtole,nn=self.nn,vdwnn=self.vdwnn,
                  learnpair=learnpair,beta=beta,groupi=groupi,groupj=groupj,freeatoms=self.freeatoms)
      if learnWay==2:
         Deformed,zmats,zv,zvlo,zvhi = irmd.opt()
      else:
         Deformed,zmats,zv,zvlo,zvhi = irmd.run()
      mdsteps= irmd.step
      Emd    = irmd.Epot
      irmd.close()
      return Emd,mdsteps,Deformed,zmats,zv,zvlo,zvhi

  def get_ro(self,rodic):
      if rodic is None:
         self.rodic= {'C-C':1.35,'C-H':1.05,'C-N':1.45,'C-O':1.35,
                      'N-N':1.35,'N-H':1.05,'N-O':1.30,
                      'O-O':1.35,'O-H':1.05,
                      'H-H':0.8,
                      'others':1.35} 
      else:
           self.rodic= rodic

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

  def get_fixatom(self):
      atoms             = self.a.ir.atoms
      if self.freeatoms is None:
         self.fixatoms  = None
      else:
         self.fixatoms  = FixAtoms(indices=[atom.index for atom in atoms if atom.index not in self.freeatoms])


  def save_config(self):
      with open('options.json','w') as fj:
           options = {'accCriteria':self.accCriteria,
                    'lossCriteria':self.lossCriteria,
                    'lossTole':self.lossTole,
                    'rmax':self.rmax,
                    'angmax':self.angmax,
                    'accTole':self.accTole,
                    'accMax':self.accMax,
                    # 'accInc':self.accInc,
                    # 'resetAcc':self.resetAcc,
                    'step':self.step,
                    'md_step':self.md_step,
                    'mom_step':self.mom_step,
                    'MaxMDstep':self.MaxMDstep,
                    'EngTole':self.EngTole,
                    'dEtole':self.dEtole,
                    'dEstop':self.dEstop,
                    'beta':self.beta,
                    # 'mdInc':self.mdInc,
                    'regularize':self.regularize,
                    'lambda_reg':self.lambda_reg,
                    'lambda_bd':self.lambda_bd,
                    'lambda_me':self.lambda_me,
                    'lambda_ang':self.lambda_ang,
                    'learning_rate':self.learning_rate,
                    'rmin':self.rmin,
                    'T':self.T,
                    'maxiter':self.maxiter,
                    'max_batch':self.max_batch,
                    'optword':self.optword,
                    'learnWay':self.learnWay,
                    'colFrame':self.colFrame,
                    'CheckZmat':self.CheckZmat,
                    'freeatoms':self.freeatoms,
                    'freepairs':self.freepairs,
                    'bore':self.bore,
                    'writelib':self.writelib}
           if self.learnWay==1:
              options['dft_step'] = self.dft_step
           js.dump(options,fj,sort_keys=True,indent=2) 

  def load_config(self):
      with open('options.json','r') as fj:
           options = js.load(fj)
           self.accCriteria  = options['accCriteria']
           self.lossCriteria = options['lossCriteria']
           self.lossTole     = options['lossTole']
           self.rmax         = options['rmax']
           self.angmax       = options['angmax']
           self.accTole      = options['accTole']
           self.accMax     = options['accMax']
           # self.accInc     = options['accInc']
           # self.resetAcc   = options['resetAcc']
           self.step         = options['step'] 
           if self.learnWay==1:
              self.dft_step     = options['dft_step'] 
           self.md_step      = options['md_step'] 
           self.mom_step     = options['mom_step'] 
           self.MaxMDstep    = options['MaxMDstep'] 
           self.EngTole      = options['EngTole'] 
           self.dEtole       = options['dEtole'] 
           self.dEstop       = options['dEstop'] 
           self.beta         = options['beta'] 
           # self.mdInc      = options['mdInc'] 
           self.regularize   = options['regularize'] 
           self.lambda_reg   = options['lambda_reg'] 
           self.lambda_bd    = options['lambda_bd']
           self.lambda_me    = options['lambda_me']
           self.lambda_ang   = options['lambda_ang']
           self.rmin         = options['rmin'] 
           self.T            = options['T'] 
           self.maxiter      = options['maxiter']
           self.max_batch    = options['max_batch']
           self.optword      = options['optword']
           self.learnWay     = options['learnWay']
           self.learning_rate= options['learning_rate']
           self.colFrame     = options['colFrame']
           self.CheckZmat    = options['CheckZmat']
           self.bore         = options['bore']
           self.writelib     = options['writelib']
           self.freepairs    = options['freepairs']
           if self.freeatoms!= options['freeatoms']:
              self.freeatoms = options['freeatoms']
              self.a.get_freebond(self.freeatoms)
           
  def close(self):
      print('-  LM compeleted.')
      self.atom_name = None
      self.ro        = None


if __name__ == '__main__':
   direcs = {'nm6_5':'/home/feng/siesta/nm6_5',
             'nm6_14':'/home/feng/siesta/nm6_14' }

   lm = LearningMachine(initConfig='noco6.gen',
                        direcs=direcs,
                        ncpu=4,
                        maxiter=100,
                        dt_mlmd=0.2,dt_aimd=1.0,dft_step=5,
                        batch=50,max_batch=50,
                        step=20000,md_step=50,mom_step=20,
                        colFrame=25,
                        MaxMDstep=100,
                        T=800,Tmax=800,
                        convergence=0.04,accCriteria=0.9,accMax=0.962,
                        accTole=0.01,accInc=1.001,
                        lossCriteria=26.0,lossTole=2.0,
                        CheckZmat=False,
                        resetAcc=1,
                        mdInc=1.1,
                        nn=True,
                        messages=1,
                        mefac=0.005,
                        learnWay=5,
                        weight={'noco':2.0,'others':2.0},
                        EngTole=0.085,dEtole=0.14,dEstop=1.0,
                        rmin=0.7,rmax=1.3,angmax=30.0,
                        bo_layer=[4,1],mf_layer=[6,2],be_layer=[3,1],
                        EnergyFunction=3,MessageFunction=1,
                        FreeAtoms=[2,3,12,13],
                        writelib=5000,
                        xcf='GGA',xca='PBE',basistype='split')
   
   lm.run()
   lm.close()


