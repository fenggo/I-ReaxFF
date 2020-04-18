from __future__ import print_function
from os.path import isfile
from .mdtodata import MDtoData
from .reaxfflib import read_lib,write_lib
import numpy as np
import tensorflow as tf
import numpy as np
from .genetic import genetic,Chromosome,Strategies
import pickle
import time


rcutoff= 10.0
rd     = 1.0/rcutoff**7.0
Q8     = 20.0*rd
Q7     = -70.0*rd*rcutoff
Q6     = 84.0*rd*rcutoff*rcutoff
Q5     = -35.0*rd*rcutoff*rcutoff*rcutoff
Q1     = 1.0


def create_gif(mol):
    ''' '''
    epses = listdir('./')
    image_list,frames = [],[]
    ind = []
    for im in epses:
        if im.find('result_'+mol+'_')>=0 and im.find('.eps')>=0:
           number = im.split('_')[2]
           number = int(number[:-4])
           ind.append(number)
           image_list.append(im)

    ind = np.array(ind)
    image_list = np.array(image_list)
    indices = ind.argsort()
    image_list = image_list[indices]

    filenames = ['result_'+mol+'_%02i.eps' %index for index in range(len(images))]
    command = ('convert -delay 100 %s -loop 0 animation.gif' %' '.join(filenames))
    os.system(command)


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


def qtap(r,rmax):
    fr    = np.where(r<rmax,1.0,0.0)  
    r_    = np.where(r<rmax,r,0.0)  
    tp    = (((Q8*r_+Q7)*r_+Q6)*r_+Q5)*r_*r_*r_*r_ + fr
    return tp



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
               convergence=1.0,
               poolSize=100,
               maxpoolSize=500):
      self.board    = board
      self.to_train = to_train
      self.direcs   = direcs
      self.libfile  = libfile
      self.lib_best = self.libfile+'_best'
      p,zpe,self.spec,self.bonds,self.offd,self.angs,self.torp,self.hbs=read_lib(libfile=libfile)
      self.p_       = p
      self.poolSize = poolSize
      self.maxpoolSize = maxpoolSize
      self.batch    = batch
      self.itmax    = itmax
      self.plr      = plr
      self.time     = time.time()
      self.logger   = logger('trainq.log')
      self.convergence = convergence
      self.loss_best= 1.0e9
      self.write_best = False
      self.Q        = {}
      for mol in self.direcs: 
          self.Q[mol] = Qeq(direc=self.direcs[mol],
                            itmax=self.itmax,batch=self.batch)


  def set_variables(self,libfile=None,Train=True):
      if not libfile is None:
         self.p_,zpe,spec,bonds,offd,angs,torp,hbs=read_lib(libfile=libfile)
      self.p,self.v = {},{}
      for key in self.p_:
          kpre = key.split('_')[0]
          if kpre in ['chi','mu','gamma']:
             # print('-  Initilize variable: %s = %f ...' %(key,self.p_[key]))
             if Train:
                self.v[key] = tf.Variable(self.p_[key],name=key)
             else:
             	# self.p[key] = tf.Variable(self.p_[key],trainable=False,name=key)
             	self.p[key] = tf.placeholder(tf.float32,name=key)
 
      if Train: self.clip_prameters()


  def clip_prameters(self):
      ''' clipe operation: clipe values in resonable range '''   
      for k in self.v:
          vn  = k.split('_')
          key = vn[0]
          if key == 'gamma':
             self.p[k] = tf.clip_by_value(self.v[k],0.0001,5.0)
          elif key in ['chi','mu']:
             self.p[k] = tf.clip_by_value(self.v[k],0.0001,100.0)
          else:
             self.p[k] = self.v[k]


  def update(self):
      self.logger.info('-  updating best variables configrations ...')
      upop = []
      self.p_,zpe,spec,bonds,offd,angs,torp,hbs=read_lib(libfile=self.lib_best)
      for key in self.v:
          upop.append(tf.assign(self.v[key],self.p_[key]))
      self.sess.run(upop)


  def build_graph(self):
      # print('-  building graph ...')
      self.loss = {}
      self.Loss = 0.0
      for mol in self.direcs: 
          self.Q[mol].calc(self.p)
          self.loss[mol] = tf.nn.l2_loss(self.Q[mol].qs-self.Q[mol].q,
              name='loss_%s' %mol)
          self.Loss += tf.reduce_sum(tf.square(self.Q[mol].qtot))
          self.Loss += self.loss[mol] 


  def session(self,learning_rate=3.0-4,method='AdamOptimizer'):
      self.sess= tf.Session()  
      if self.board:
         writer = tf.summary.FileWriter("logs/", self.sess.graph)
         # see logs using command: tensorboard --logdir logs

      if method=='GradientDescentOptimizer':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
      elif method=='AdamOptimizer':
         optimizer = tf.train.AdamOptimizer(learning_rate) 
      elif method=='AdagradOptimizer':
         optimizer = tf.train.AdagradOptimizer(learning_rate) 
      elif method=='MomentumOptimizer':
         optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                self.momentum)  #momentum=0.9
      elif method=='RMSPropOptimizer':
         optimizer = tf.train.RMSPropOptimizer(learning_rate)
      elif method=='NadagradOptimizer':
         optimizer = tf.train.NadagradOptimizer(learning_rate) 

      if self.to_train:
         self.train_step = optimizer.minimize(self.Loss)
      self.sess.run(tf.global_variables_initializer())  


  def setup(self,Train=True,libfile=None):
      self.set_variables(libfile=libfile,Train=Train)
      self.build_graph()   


  def custom_mutate(self):
      self.sess.close()
      self.setup(libfile=self.lib_best)
      fitness = self.run(learning_rate=1.0e-4,method='AdamOptimizer',
                           step=300,print_step=10,write_best=True)
      self.setup()
      self.sess= tf.Session()  
      genes = self.libToGenes(libfile='ffield_best')
      return Chromosome(genes,fitness,Strategies.Create)


  def GetFitness(self,genes):
      fdic = {}
      for i,v in enumerate(self.geneName):
          self.p_[v] = float(genes[i]*self.geneFc[i]+self.geneSft[i])
          fdic[self.p[v]] = self.p_[v]
   
      Fitness_ = self.sess.run(self.Loss,feed_dict=fdic)
      Fitness  = 1.0e9 if np.isnan(Fitness_) else Fitness_
      return Fitness


  def genetical(self,maxAge=20,nGenes=500,
                optimalFitness=1.0,maxSeconds=6000,
                Fc ={'chi':13.0,'mu':13.0,'gamma':2.0},
                zeroSft=True,useLocal=False):
      ''' Genetic Algriom '''
      self.geneFc        = []
      self.geneSft       = []
      self.geneName      = []
      self.lib_best      = self.libfile+'_best'

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

      self.setup(Train=False)
      self.sess= tf.Session()  
      genes = self.libToGenes()
      self.Loss_Best = self.GetFitness(genes)

      custom_mutate=self.custom_mutate if useLocal else None
      self.ga = genetic(targetLen=len(self.geneName),
                        maxAge=maxAge,
                        poolSize=self.poolSize,
                        nGenes=100,
                        optimalFitness=optimalFitness,
                        geneName=self.geneName,
                        get_fitness=self.GetFitness,
                        maxSeconds=maxSeconds,
                        custom_mutate=custom_mutate)

      if isfile('CurrentPOP.pkl'):
         f = open('CurrentPOP.pkl', 'rb') 
         self.ga.parents = pickle.load(f)   
         f.close()   

         ft = []
         for chrom in self.ga.parents: # recalculate fitness
             fitness = self.GetFitness(chrom.Genes)
         self.ga.parents.sort(key=lambda x:x.Fitness, reverse=True)
         for chrom in self.ga.parents: # recalculate fitness
             ft.append(fitness)
      else:
         self.ga.parents = []
         ft = []

      self.ga.historicalFitnesses = sorted(ft,reverse=True) 
      self.ga.bestID = np.argmin(ft) 


  def evolve(self):
      ''' find best population '''
      self.g = 0
      startTime  = time.time()
      for timedOut,chrom in self.ga.to_evolve():
          self.g  += 1
          t = time.time()-startTime
          self.ga.logger.generation(chrom,self.ga.geneName,t,generation=self.g)
          self.ga.usedStrategies.append(chrom.Strategy)

          f = open('CurrentPOP.pkl', 'wb')  # save population to file
          pickle.dump(self.ga.parents,f)
          f.close()

          self.geneToLib(chrom)
          if self.ga.optimalFitness>=chrom.Fitness:
             return chrom
             self.geneToLib(chrom)


  def libToGenes(self,p=None,libfile='ffield'):
      if p is None:
         self.p_,zpe,spec,bonds,offd,angs,torp,hbs=read_lib(libfile=libfile)
         p = self.p_
      genes = []
      for i,gn in enumerate(self.geneName):
          gene = (p[gn] - self.geneSft[i])/self.geneFc[i]
          genes.append(gene)
      return genes


  def geneToLib(self,chromosome,libfile='ffield_best'):
      genes = chromosome.Genes
      for i,v in enumerate(self.geneName):
          self.p_[v] = float(genes[i]*self.geneFc[i]+self.geneSft[i])
      write_lib(self.p_,self.spec,self.bonds,self.offd,
                self.angs,self.torp,self.hbs,
                libfile=libfile,
                loss=chromosome.Fitness,
                logo='!-ReaxFF-From-Genetic-Algorithms-Generation-%s' %self.g)
 

  def sa(self,learning_rate=1.0e-2,method='AdamOptimizer',
         total_step=100,step=200,print_step=10,momentum=0.01,
         Fc ={'chi':50.0,'mu':50.0,'gamma':5.0},
         zeroSft=True):
      ''' like a simulated-annealing algriom '''
      self.geneFc        = []
      self.geneSft       = []
      self.geneName      = []
      self.lib_best      = self.libfile+'_best'

      if isfile('CurrentPOP.pkl'):
         f = open('CurrentPOP.pkl', 'rb') 
         self.parents = pickle.load(f)   
         f.close()   
      else:
         self.parents    = []
         
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
      self.momentum   = momentum
      
      for i in range(total_step):
          self.setup(libfile='ffield_LocalBest')
          self.run(learning_rate=1.0e-2,method=method,
                   step=step,print_step=print_step,write_best=True)

          self.setup(libfile='ffield_LocalBest')
          self.run(learning_rate=1.0e-4,method=method,
                   step=step,print_step=print_step,write_best=True)

          if len(self.parents)>self.poolSize:
             break

      self.parents.sort(key=lambda x:x.Fitness, reverse=True)
      f = open('CurrentPOP.pkl', 'wb')  # save population to file
      pickle.dump(self.parents,f)
      f.close()


  def run(self,learning_rate=1.0e-4,method='MomentumOptimizer',
               step=2000,print_step=10,writelib=100,
               momentum=0.1,write_best=False):
      accs_    ={}
      self.momentum   = momentum
      self.write_best = write_best
      # self.setup()
      self.session(learning_rate=learning_rate,method=method) 
      Loss = -1
      exit_     = 0

      for i in range(step+1):
          loss_last   = Loss
          Loss,loss,_ = self.sess.run([self.Loss,
                                       self.loss,
                                       self.train_step])
          if Loss==loss_last:
             exit_ += 1
          else:
             exit_ == 0
          if exit_>5: break

          if i==0:
             loss_best = Loss
             if self.write_best: self.write_lib(libfile='ffield_LocalBest',loss=loss_best)
             # self.p_best = self.p_
          else:
             if Loss<loss_best:
                loss_best = Loss
                if self.write_best: 
                   self.write_lib(libfile='ffield_LocalBest',loss=loss_best)
                   genes = self.libToGenes(p=self.p_)
                   self.parents.append(Chromosome(genes,loss_best,Strategies.Create))

                # self.p_best = self.get_solution()

          if Loss<self.loss_best:
             self.loss_best = Loss
             if self.write_best: self.write_lib(libfile=self.lib_best,loss=self.loss_best)

          if np.isnan(Loss):
             self.logger.info('NAN error encountered, stop at step %d loss is %f.' %(i,Loss))
             # send_msg('NAN error encountered, stop at step %d loss is %f.' %(i,Loss))
             break
             
          if i%print_step==0:
             current = time.time()
             elapsed_time = current - self.time
             acc = ''
             for key in loss:
                 acc += key+': '+str(loss[key])+' '

             self.logger.info('-  step: %d sqe: %6.4f losses: %s time: %6.4f' %(i,Loss,acc,elapsed_time))
             self.time = current
             
          if (i%writelib==0 or i==step):
             if not self.write_best: self.write_lib(libfile=self.libfile+'_'+str(i),loss=Loss)
             # if self.plr:
             #    self.plot_result()

          if Loss<self.convergence:
             self.write_lib(libfile=self.libfile)
             raise ConvergenceOccurred('coverged, self.accu')

      # for mol in self.mols:   
      #     create_gif(mol)
      tf.reset_default_graph()
      self.sess.close()
      return loss_best


  def get_solution(self):
      p_   = self.sess.run(self.p)
      return p_


  def write_lib(self,libfile='ffield',loss=None,fdic=None):
      if fdic is None:
         p_   = self.sess.run(self.p)
      else:
      	 p_   = self.sess.run(self.p,feed_dict=fdic)
      for k in p_:
          self.p_[k] = p_[k]
      write_lib(self.p_,self.spec,self.bonds,self.offd,
                self.angs,self.torp,self.hbs,
                libfile=libfile,
                loss=loss)


  def close(self):
      print('-  Train Qeq job compeleted.')
      type(self).count -= 1



class Qeq(object):
  ''' 
     charges calculate for ReaxFF potential
  '''
  def __init__(self,mol='myself',
               dft='siesta',direc=None,batch=50,
               rcutoff=10.0,
               itmax=35):
      self.d         = MDtoData(structure=mol,dft=dft,direc=direc,batch=batch)
      self.x         = self.d.x
      self.cell      = self.d.cell
      self.qs        = self.d.qs
      self.box       = np.array([self.cell[0][0],self.cell[1][1],self.cell[2][2]])

      self.itmax     = itmax
      self.rcutoff   = rcutoff
      self.box       = np.array([self.cell[0][0],self.cell[1][1],self.cell[2][2]])
      self.atom_name = self.d.atom_name
      self.natom     = self.d.natom
      self.batch     = batch
     

  def setup(self,p):
      self.qtot      = np.zeros([self.batch,1],dtype=np.float32)
      self.q_        = np.zeros([self.batch,self.natom],dtype=np.float32)    # resolution        
      self.D_        = np.zeros([self.natom,self.natom],dtype=np.float32)
      self.eye       = np.expand_dims(np.eye(self.natom,dtype=int),axis=0)
      self.diag      = 1.0 - self.eye
      self.angstoev  = 14.39975840 
      self.vr        = self.compute_bond(self.x,self.natom)
      chi            = []
      mu             = []           
      gamma          = []  
      for i,atom in enumerate(self.atom_name):
          chi.append(p['chi_'+atom])
          mu.append(p['mu_'+atom])
          gamma.append(p['gamma_'+atom])

      self.chi   = tf.stack(chi,axis=0)
      self.mu    = tf.stack(mu,axis=0)
      self.gamma = tf.stack(gamma,axis=0)
      self.z     = -tf.expand_dims(self.chi,0)
      # self.sess= tf.Session()


  def calc(self,p):
      self.setup(p)
      self.q = self.q_.copy()
      # self.z = self.z_.copy()
      self.D = self.D_.copy()
      gamma_ = tf.sqrt(tf.expand_dims(self.gamma,0)*tf.expand_dims(self.gamma,1))
      gamma2 = 1.0/gamma_**3
      gamma2 = tf.expand_dims(gamma2,0)
      
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  box = self.box*[i,j,k]
                  vr_ = self.vr+box 

                  vr2 = np.square(vr_)
                  R_  = np.sqrt(np.sum(vr2,axis=3))
                  tp  = qtap(R_,self.rcutoff)
                 
                  gam_= (tp/(R_**3+gamma2)**(1.0/3.0))
                  if i==0 and j==0 and k==0:
                     gam_ = gam_*self.diag
                     
                  self.D += gam_*self.angstoev # *self.ut

      D_          = self.D*self.eye
      self.D      = self.D - D_

      Deye        = D_ + 2.0*tf.reshape(self.mu,[1,1,self.natom])*self.eye
      self.D     += Deye
      self.Deye   = tf.reduce_sum(Deye,2)

      #  sparseAxV i.e. D x q 
      qd_,qtot    = self.dot(self.D,self.q,self.qtot)
      r           = self.z - qd_
      rtot        = self.qtot - qtot
      self.bnrm   = tf.reduce_sum(self.z*self.z,1)  

      # sparseAdiagprecon D r
      r_          = r
      z           = self.z
      z           = r/self.Deye
      ztot        = rtot

      self.iteration(z,ztot,r,rtot)
 

  def iteration(self,z,ztot,r,rtot):
      ''' iterativly compute the QEq Charges '''
      for it in range(self.itmax):
          z_    = r/self.Deye
          ztot_ = rtot
          tot_  = ztot_*rtot
          bk_   = tf.reshape(tf.reduce_sum(z*r,1),[self.batch,1]) + tot_

          if it==0:
             p    =z
             ptot = ztot
             p_   = z_ 
             ptot_= ztot_
          else:
             bk   = tf.reshape(bk_/(bkd+0.00000001),[self.batch,1])
             p    = bk*p  + z
             p_   = bk*p_ + z_
             ptot = bk*ptot  + ztot
             ptot_= bk*ptot_ + ztot_
          bkd = bk_
        
          # operation between D |nxn| and p(=z|n+1|)
          z,ztot  = self.dot(self.D,p,ptot)
          akd     = tf.reshape(tf.reduce_sum(z*p_,1),[self.batch,1]) + ztot*ptot_
          ak      = tf.reshape(bk_/(akd+0.00000001),[self.batch,1])
          z_,ztot_= self.dot(self.D,p_,ptot_)

          self.q  = self.q + ak*p
          self.qtot = self.qtot + ak*ptot
          r       = r - ak*z
          rtot    = rtot - ak*ztot

          z       = r/self.Deye
          ztot    = rtot

          # err     = tf.sqrt(tf.reduce_sum(r*r,1))/self.bnrm
          # err_    = self.sess.run(tf.reduce_sum(err))
          # print('-  iteration = %2d  Error = ' %it,err_)
      # self.q += self.qtot
      # q = self.sess.run(self.q)
      # print('\n-  QEq charges after %d iteration:\n\n' %it,q)


  def dot(self,D,q,totq):
      '''
         D = B X N X N 
         q = B X N X 1
      totq = B X 1
      '''
      q    = tf.reshape(q,[self.batch,self.natom,1])
      qd_  = tf.matmul(self.D,q)
      qd_  = tf.reshape(qd_,[self.batch,self.natom])
      qd_ += tf.reshape(totq,[self.batch,1])
      totq_= tf.reduce_sum(q,1)
      return qd_,totq_


  def compute_bond(self,x,natom):
      cell = self.box
      hfcell = 0.5*cell

      # x  = np.array(x)
      xj = np.expand_dims(x,axis=1)
      xi = np.expand_dims(x,axis=2)
      vr = xj - xi

      lm = np.where(vr-hfcell>0)
      lp = np.where(vr+hfcell<0)
      while(lm[0].size!=0 or lm[1].size!=0 or lm[2].size!=0 or \
            lp[0].size!=0 or lp[1].size!=0 or lp[2].size!=0):
          vr   = np.where(vr-hfcell>0,vr-cell,vr)
          vr   = np.where(vr+hfcell<0,vr+cell,vr)     # apply pbc
          lm   = np.where(vr-hfcell>0)
          lp   = np.where(vr+hfcell<0)
      return vr

