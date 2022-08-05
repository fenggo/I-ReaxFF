import matplotlib.pyplot as plt
from os import system, getcwd, chdir,listdir,environ,makedirs
from os.path import isfile,exists,isdir
from .reax_data import get_data 
from .link import links
from .reaxfflib import write_ffield
from .reax import ReaxFF,taper
from .initCheck import Init_Check
from .dingtalk import send_msg
import time
from ase import Atoms
from ase.io.trajectory import Trajectory
import tensorflow as tf
# from tensorflow.contrib.opt import ScipyOptimizerInterface
import numpy as np
import random
import pickle
import json as js
# tf_upgrade_v2 --infile reax.py --outfile reax_v1.py
# tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


def dsigmoid(x):
    ds = tf.sigmoid(x)(1-tf.sigmoid(x))
    return ds


class MPNN(ReaxFF):
  def __init__(self,libfile='ffield',dataset={},
               dft='ase',atoms=None,
               cons=['val','vale','valang','vale',    # 'valboc',
                     'ovun1','ovun2','ovun3','ovun4',
                     'ovun5','ovun6','ovun7','ovun8',
                     'lp2','lp3',#'lp1',
                     'cot1','cot2',
                     'coa1','coa2','coa3','coa4',
                     'pen1','pen2','pen3','pen4',
                     'Depi','Depp',#'Desi','Devdw',
                     #'bo1','bo2','bo3','bo4','bo5','bo6',
                     #'rosi','ropi','ropp',
                     'cutoff','hbtol','acut',#'val8','val9','val10',
                     ], # 
               nn=True,
               optmol=True,lambda_me=0.1,
               opt=None,optword='nocoul',
               mpopt=None,bdopt=None,mfopt=None,
               VariablesToOpt=None,
               batch_size=200,sample='uniform',
               hbshort=6.75,hblong=7.5,
               vdwcut=10.0,
               beup={},                                 # e.g. {'C-C':[(1.5,0.5)]} or {'C-C':[(1.5,'si')]}
               belo={},
               vlo={'others':[(0.0,0.0)]},
               vup={'others':[(10.0,0.0)]},
               pim={'others':10.0},
               spv_be=False,                             
               spv_bo=None,                             # e.g. spv_bo={'C-C':(3.0,0.2,1.0)}
               spv_pi=False,
               spv_ang=False,
               spv_vdw=False,
               fixrcbo=False,
               weight={'others':1.0},
               ro_scale=0.1,
               clip_op=True,
               clip={},
               InitCheck=True,
               resetDeadNeuron=False,
               messages=1,
               be_univeral_nn=None,be_layer=[9,0],
               bo_univeral_nn=None,bo_layer=[6,0],
               mf_univeral_nn=None,mf_layer=[9,0],
               vdw_univeral_nn=None,vdw_layer=[9,0],
               vdwnn=False,VdwFunction=1,
               BOFunction=0,
               EnergyFunction=1,
               MessageFunction=3,
               spec=[],
               sort=False,
               pkl=False,
               lambda_bd=100000.0,
               lambda_pi=1.0,
               lambda_reg=0.01,
               lambda_ang=1.0,
               fluctuation=0.0,
               regularize_bo=True,
               regularize_be=True,
               regularize_mf=True,
               regularize_vdw=False,
               regularize_bias=False,
               optMethod='ADAM',
               maxstep=60000,
               emse=0.9,
               convergence=0.97,
               lossConvergence=1000.0,
               losFunc='n2',
               conf_vale=None,
               huber_d=30.0,
               ncpu=None):
      '''
         Message Passing Neural network build top on ReaxFF potentail
         version 3.0 
           Time: 2018-10-20
           Intelligence ReaxFF Neual Network: Evoluting the Force Field parameters on-the-fly
           2017-11-01
      '''
      self.messages         = messages
      self.EnergyFunction   = EnergyFunction
      self.MessageFunction  = MessageFunction
      self.BOFunction       = BOFunction
      self.pim              = pim
      self.spv_bo           = spv_bo
      self.spv_be           = spv_be
      self.beup             = beup
      self.belo             = belo
      self.spv_pi           = spv_pi
      self.spv_ang          = spv_ang
      self.spv_vdw          = spv_vdw
      self.vup              = vup
      self.vlo              = vlo
      self.fixrcbo          = fixrcbo
      self.regularize_be    = regularize_be
      self.regularize_bo    = regularize_bo
      self.regularize_mf    = regularize_mf
      self.regularize_vdw   = regularize_vdw
      self.regularize_bias  = regularize_bias
      if regularize_vdw or regularize_mf or regularize_bo or regularize_be:
         self.regularize    = True
      else:
         self.regularize    = False
      self.lambda_reg       = lambda_reg
      self.lambda_pi        = lambda_pi
      self.lambda_ang       = lambda_ang
      self.fluctuation      = fluctuation
      self.mf_layer         = mf_layer
      self.be_layer         = be_layer
      self.vdw_layer        = vdw_layer if vdwnn else None

      self.bo_univeral_nn   = bo_univeral_nn
      self.be_univeral_nn   = be_univeral_nn
      self.mf_univeral_nn   = mf_univeral_nn
      self.vdw_univeral_nn  = vdw_univeral_nn
      if mpopt is None:
         self.mpopt = [True,True,True,True]
      else:
         self.mpopt = mpopt
      self.bdopt    = bdopt
      self.mfopt    = mfopt
      ReaxFF.__init__(self,libfile=libfile,dataset=dataset,
                      dft=dft,atoms=atoms,cons=cons,opt=opt,optword=optword,
                      VariablesToOpt=VariablesToOpt,optmol=optmol,lambda_me=lambda_me,
                      batch_size=batch_size,sample=sample,
                      hbshort=hbshort,hblong=hblong,vdwcut=vdwcut,
                      ro_scale=ro_scale,
                      clip_op=clip_op,clip=clip,
                      InitCheck=InitCheck,resetDeadNeuron=resetDeadNeuron,
                      nn=nn,vdwnn=vdwnn,VdwFunction=VdwFunction,
                      bo_layer=bo_layer,spec=spec,sort=sort,pkl=pkl,weight=weight,
                      lambda_bd=lambda_bd,
                      optMethod=optMethod,maxstep=maxstep,
                      emse=emse,convergence=convergence,lossConvergence=lossConvergence,
                      losFunc=losFunc,conf_vale=conf_vale,
                      huber_d=huber_d,ncpu=ncpu)
      self.S        = {}
      self.esi      = {}
      self.fbo      = {}

  def get_loss(self):
      self.Loss = 0.0
      for mol in self.mols:
          mol_ = mol.split('-')[0]
          if mol in self.weight:
             w_ = self.weight[mol]
          elif mol_ in self.weight:
             w_ = self.weight[mol_]
          else:
             w_ = self.weight['others']

          if self.losFunc   == 'n2':
             self.loss[mol] = tf.nn.l2_loss(self.E[mol]-self.dft_energy[mol],
                                 name='loss_%s' %mol)
          elif self.losFunc == 'abs':
             self.loss[mol] = tf.compat.v1.losses.absolute_difference(self.dft_energy[mol],self.E[mol])
          elif self.losFunc == 'mse':
             self.loss[mol] = tf.compat.v1.losses.mean_squared_error(self.dft_energy[mol],self.E[mol])
          elif self.losFunc == 'huber':
             self.loss[mol] = tf.compat.v1.losses.huber_loss(self.dft_energy[mol],self.E[mol],delta=self.huber_d)
          elif self.losFunc == 'CrossEntropy':
             y_min = tf.reduce_min(self.dft_energy[mol])
             a_min = tf.reduce_min(self.E[mol])
             norm  = tf.minimum(y_min,a_min) - 0.00000001
             y     = self.dft_energy[mol]/norm
             y_    = self.E[mol]/norm
             self.loss[mol] =  (-1.0/self.batch)*tf.reduce_sum(y*tf.math.log(y_)+(1-y)*tf.math.log(1.0-y_))
          else:
             raise NotImplementedError('-  This function not supported yet!')

          sum_edft = tf.reduce_sum(input_tensor=tf.abs(self.dft_energy[mol]-self.max_e[mol]))
          self.accur[mol] = 1.0 - tf.reduce_sum(input_tensor=tf.abs(self.E[mol]-self.dft_energy[mol]))/(sum_edft+0.00000001)
         
          self.Loss     += self.loss[mol]*w_
          self.accuracy += self.accur[mol]

      self.ME   = 0.0
      for mol in self.mols:
          mols = mol.split('-')[0] 
          self.ME += tf.square(self.MolEnergy[mols])

      self.loss_penalty  = self.supervise()
      self.Loss         += self.loss_penalty

      if self.optmol:
         self.Loss  += self.ME*self.lambda_me
      self.accuracy  = self.accuracy/self.nmol

  def build_graph(self):
      print('-  building graph ...')
      self.accuracy   = tf.constant(0.0,name='accuracy')
      self.accuracies = {}
      self.get_bond_energy()
      self.get_atom_energy()
      self.get_angle_energy()
      self.get_torsion_energy()
      self.get_vdw_energy()
      self.get_hb_energy()
      self.get_total_energy()
      self.get_loss()
      print('-  end of build.')

  def get_atomic_energy(self):
      i = 0
      for sp in self.spec:
          if self.nsp[sp]==0:
             continue
          self.eatom[sp] = -tf.ones([self.nsp[sp]])*self.p['atomic_'+sp]
          self.EATOM  = self.eatom[sp] if i==0 else tf.concat((self.EATOM,self.eatom[sp]),0)
          i += 1

      for mol in self.mols:
          mols = mol.split('-')[0] 
          zpe_ = tf.gather_nd(self.EATOM,self.atomlink[mol]) 
          self.zpe[mol] = tf.reduce_sum(input_tensor=zpe_,name='zpe') + self.MolEnergy[mols]

  def get_total_energy(self):
      for mol in self.mols:
          # mols = mol.split('-')[0] 
          self.E[mol] = tf.add(self.ebond[mol] + 
                               self.eover[mol] +
                               self.eunder[mol]+
                               self.elone[mol] +
                               self.eang[mol]  +
                               self.epen[mol]  +
                               self.tconj[mol] +
                               self.etor[mol]  +
                               self.efcon[mol] +
                               self.evdw[mol]  +
                               self.ecoul[mol] +
                               self.ehb[mol]   +
                               self.eself[mol], 
                               self.zpe[mol],name='E_%s' %mol)  

  def get_tap(self,r,bd,di,dj):
      if self.vdwnn:
         # r_ = r/self.p['rvdw_'+bd]
         if self.VdwFunction==1:
            tp = self.f_nn('fv',bd,self.nvb[bd],[r],layer=self.vdw_layer[1])
         elif self.VdwFunction==2:
            tpi = self.f_nn('fv',bd,self.nvb[bd],[r,di,dj],layer=self.vdw_layer[1])
            tpj = self.f_nn('fv',bd,self.nvb[bd],[r,dj,di],layer=self.vdw_layer[1])
            tp  = tpi*tpj
         else:
            raise RuntimeError('-  This method not implimented!')
      else:
         tp = 1.0+tf.math.divide(-35.0,tf.pow(self.vdwcut,4.0))*tf.pow(r,4.0)+ \
              tf.math.divide(84.0,tf.pow(self.vdwcut,5.0))*tf.pow(r,5.0)+ \
              tf.math.divide(-70.0,tf.pow(self.vdwcut,6.0))*tf.pow(r,6.0)+ \
              tf.math.divide(20.0,tf.pow(self.vdwcut,7.0))*tf.pow(r,7.0)
      return tp

  def f_nn(self,pre,bd,nbd,x,layer=5):
      ''' Dimention: (nbatch,4) input = 4
                 Wi:  (4,8) 
                 Wh:  (8,8)
                 Wo:  (8,1)  output = 1
      '''
      nd = len(x)
      x_ = []
      for d in x:
          x_.append(tf.reshape(d,[nbd*self.batch]))
      X   = tf.stack(x_,axis=1)        # Dimention: (nbatch,4)
                                       #        Wi:  (4,8) 
      o   =  []                        #        Wh:  (8,8)
      o.append(tf.sigmoid(tf.matmul(X,self.m[pre+'wi_'+bd],name='input')+self.m[pre+'bi_'+bd]))   # input layer

      for l in range(layer):                                                   # hidden layer      
          o.append(tf.sigmoid(tf.matmul(o[-1],self.m[pre+'w_'+bd][l],name='hide')+self.m[pre+'b_'+bd][l]))

      o_ = tf.sigmoid(tf.matmul(o[-1],self.m[pre+'wo_'+bd],name='output') + self.m[pre+'bo_'+bd])  # output layer
      out= tf.reshape(o_,[nbd,self.batch])
      return out

  def fmessage(self,pre,bd,nbd,x,layer=5):
      ''' Dimention: (nbatch,4) input = 4
                 Wi:  (4,8) 
                 Wh:  (8,8)
                 Wo:  (8,3)  output = 3
      '''
      nd = len(x)
      x_ = []
      for d in x:
          x_.append(tf.reshape(d,[nbd*self.batch]))
      X   = tf.stack(x_,axis=1)        # Dimention: (nbatch,4)
                                       #        Wi:  (4,8) 
      o   =  []                        #        Wh:  (8,8)
      o.append(tf.sigmoid(tf.matmul(X,self.m[pre+'wi_'+bd],name='bop_input')+self.m[pre+'bi_'+bd]))   # input layer

      for l in range(layer):                                                   # hidden layer      
          o.append(tf.sigmoid(tf.matmul(o[-1],self.m[pre+'w_'+bd][l],name='bop_hide')+self.m[pre+'b_'+bd][l]))

      o_ = tf.sigmoid(tf.matmul(o[-1],self.m[pre+'wo_'+bd],name='bop_output') + self.m[pre+'bo_'+bd])  # output layer
      out= tf.reshape(o_,[nbd,self.batch,3])
      return out

  def get_bondorder_uc(self,bd):
      # self.frc[bd] = tf.where(tf.logical_or(tf.greater(self.rbd[bd],self.rc_bo[bd]),
      #                                       tf.less_equal(self.rbd[bd],0.001)),
      #                       tf.zeros_like(self.rbd[bd]),tf.ones_like(self.rbd[bd]))

      self.bodiv1[bd] = tf.math.divide(self.rbd[bd],self.p['rosi_'+bd],name='bodiv1_'+bd)
      self.bopow1[bd] = tf.pow(self.bodiv1[bd],self.p['bo2_'+bd])
      if self.BOFunction==0:
          self.eterm1[bd] = (1.0+self.botol)*tf.exp(tf.multiply(self.p['bo1_'+bd],self.bopow1[bd]))#*self.frc[bd] 
      else:
          self.eterm1[bd] = tf.exp(tf.multiply(self.p['bo1_'+bd],self.bopow1[bd]))#*self.frc[bd] 

      self.bodiv2[bd] = tf.math.divide(self.rbd[bd],self.p['ropi_'+bd],name='bodiv2_'+bd)
      self.bopow2[bd] = tf.pow(self.bodiv2[bd],self.p['bo4_'+bd])
      self.eterm2[bd] = tf.exp(tf.multiply(self.p['bo3_'+bd],self.bopow2[bd]))#*self.frc[bd]

      self.bodiv3[bd] = tf.math.divide(self.rbd[bd],self.p['ropp_'+bd],name='bodiv3_'+bd)
      self.bopow3[bd] = tf.pow(self.bodiv3[bd],self.p['bo6_'+bd])
      self.eterm3[bd] = tf.exp(tf.multiply(self.p['bo5_'+bd],self.bopow3[bd]))#*self.frc[bd]

      if self.BOFunction==0:
         fsi_         = taper(self.eterm1[bd],rmin=self.botol,rmax=2.0*self.botol)*(self.eterm1[bd]-self.botol) # consist with GULP
         fpi_         = taper(self.eterm2[bd],rmin=self.botol,rmax=2.0*self.botol)*self.eterm2[bd]
         fpp_         = taper(self.eterm3[bd],rmin=self.botol,rmax=2.0*self.botol)*self.eterm3[bd]
      elif self.BOFunction==1:
         fsi_         = self.f_nn('fsi',bd, self.nbd[bd],[self.eterm1[bd]],layer=self.bo_layer[1])  
         fpi_         = self.f_nn('fpi',bd, self.nbd[bd],[self.eterm2[bd]],layer=self.bo_layer[1])  
         fpp_         = self.f_nn('fpp',bd, self.nbd[bd],[self.eterm3[bd]],layer=self.bo_layer[1]) 
      elif self.BOFunction==2:
         fsi_         = self.f_nn('fsi',bd, self.nbd[bd],[-self.eterm1[bd]],layer=self.bo_layer[1])  
         fpi_         = self.f_nn('fpi',bd, self.nbd[bd],[-self.eterm2[bd]],layer=self.bo_layer[1])  
         fpp_         = self.f_nn('fpp',bd, self.nbd[bd],[-self.eterm3[bd]],layer=self.bo_layer[1]) 
      else:
         raise NotImplementedError('-  BO function not supported yet!')

      self.bop_si[bd] = fsi_ #*self.frc[bd] #*self.eterm1[bd]  
      self.bop_pi[bd] = fpi_ #*self.frc[bd] #*self.eterm2[bd]
      self.bop_pp[bd] = fpp_ #*self.frc[bd] #*self.eterm3[bd]
      self.bop[bd]    = tf.add(self.bop_si[bd],self.bop_pi[bd]+self.bop_pp[bd],name='BOp_'+bd)

  def get_delta(self):
      ''' compute the uncorrected Delta: the sum of BO '''
      self.BOP   = tf.zeros([1,self.batch])                    # for ghost atom, the value is zero
      # self.BON = tf.zeros([1,self.batch])                    # for ghost atom, the value is zero

      if self.MessageFunction==1:
         self.BOPSI = tf.zeros([1,self.batch])    
         self.BOPPI = tf.zeros([1,self.batch]) 
         self.BOPPP = tf.zeros([1,self.batch]) 

      for bd in self.bonds:
          atomi,atomj = bd.split('-') 
          # boc = tf.sqrt(self.p['valboc_'+atomi]*self.p['valboc_'+atomj])
          if self.nbd[bd]>0:
             self.get_bondorder_uc(bd)
             self.BOP   = tf.concat([self.BOP,self.bop[bd]],0)
             # self.BON = tf.concat([self.BOP,self.bop[bd]*self.b],0)
             if self.MessageFunction==1:
                self.BOPSI = tf.concat([self.BOPSI,self.bop_si[bd]],0)
                self.BOPPI = tf.concat([self.BOPPI,self.bop_pi[bd]],0)
                self.BOPPP = tf.concat([self.BOPPP,self.bop_pp[bd]],0)

      if self.MessageFunction==1:
         self.Dpsi   = tf.reduce_sum(input_tensor=tf.gather_nd(self.BOPSI,self.dlist),axis=1,name='Dpsi')
         self.Dppi   = tf.reduce_sum(input_tensor=tf.gather_nd(self.BOPPI,self.dlist),axis=1,name='Dppi')
         self.Dppp   = tf.reduce_sum(input_tensor=tf.gather_nd(self.BOPPP,self.dlist),axis=1,name='Dppp')
         self.Dp     = tf.gather_nd(self.BOP,self.dlist) 
         self.Deltap = tf.reduce_sum(input_tensor=self.Dp,axis=1,name='Dppp')# self.Dpsi + self.Dppi + self.Dppp
      else:
         self.Dp     = tf.gather_nd(self.BOP,self.dlist)  
         self.Deltap = tf.reduce_sum(input_tensor=self.Dp,axis=1,name='Deltap')


  def get_bondorder(self,t,bd,atomi,atomj):
      ''' compute bond-order according the message function '''
      Di      = tf.gather_nd(self.D[t-1],self.dilink[bd])
      Dj      = tf.gather_nd(self.D[t-1],self.djlink[bd])
      h       = self.H[t-1][bd]

      b       = bd.split('-')
      bdr     = b[1]+'-'+b[0]
      flabel  = 'fm' # +str(t)

      if self.MessageFunction==1:
         Dsi_i = tf.gather_nd(self.D_si[t-1],self.dilink[bd]) - self.Hsi[t-1][bd]
         Dpi_i = tf.gather_nd(self.D_pi[t-1],self.dilink[bd]) - self.Hpi[t-1][bd]
         Dpp_i = tf.gather_nd(self.D_pp[t-1],self.dilink[bd]) - self.Hpp[t-1][bd]
         
         Dsi_j = tf.gather_nd(self.D_si[t-1],self.djlink[bd]) - self.Hsi[t-1][bd]
         Dpi_j = tf.gather_nd(self.D_pi[t-1],self.djlink[bd]) - self.Hpi[t-1][bd]
         Dpp_j = tf.gather_nd(self.D_pp[t-1],self.djlink[bd]) - self.Hpp[t-1][bd]

         Dpii  = Dpi_i + Dpp_i
         Dpij  = Dpi_j + Dpp_j
          
         Fi    = self.fmessage(flabel,b[0],self.nbd[bd],[Dsi_i,Dpii,self.H[t-1][bd],Dpij,Dsi_j],
                               layer=self.mf_layer[1])
         Fj    = self.fmessage(flabel,b[1],self.nbd[bd],[Dsi_j,Dpij,self.H[t-1][bd],Dpii,Dsi_i],
                               layer=self.mf_layer[1])
         F     = Fi*Fj

         Fsi,Fpi,Fpp = tf.unstack(F,axis=2)
         bosi = self.Hsi[t-1][bd]*Fsi
         bopi = self.Hpi[t-1][bd]*Fpi
         bopp = self.Hpp[t-1][bd]*Fpp
      elif self.MessageFunction==2:
         Dbi  = Di-h
         Dbj  = Dj-h
         Fi   = self.fmessage(flabel,b[0],self.nbd[bd],[Dbi,Dbj,self.Hsi[t-1][bd],self.Hpi[t-1][bd],self.Hpp[t-1][bd]],
                              layer=self.mf_layer[1])
         Fj   = self.fmessage(flabel,b[1],self.nbd[bd],[Dbj,Dbi,self.Hsi[t-1][bd],self.Hpi[t-1][bd],self.Hpp[t-1][bd]],
                              layer=self.mf_layer[1])
         F    = Fi*Fj
         Fsi,Fpi,Fpp = tf.unstack(F,axis=2)

         bosi = self.Hsi[t-1][bd]*Fsi
         bopi = self.Hpi[t-1][bd]*Fpi
         bopp = self.Hpp[t-1][bd]*Fpp
      elif self.MessageFunction==3:
         Dbi  = Di - h # self.p['valboc_'+atomi]  
         Dbj  = Dj - h # self.p['valboc_'+atomj]  
         Fi   = self.fmessage(flabel,b[0],self.nbd[bd],[Dbi,h,Dbj],layer=self.mf_layer[1])
         Fj   = self.fmessage(flabel,b[1],self.nbd[bd],[Dbj,h,Dbi],layer=self.mf_layer[1])
         F    = Fi*Fj
         Fsi,Fpi,Fpp = tf.unstack(F,axis=2)

         bosi = self.Hsi[t-1][bd]*Fsi
         bopi = self.Hpi[t-1][bd]*Fpi
         bopp = self.Hpp[t-1][bd]*Fpp
      elif self.MessageFunction==4:
         # Dbi  = Di - self.p['val_'+atomi]
         # Dbj  = Dj - self.p['val_'+atomj]
         # f_1  = self.f1(bd,atomi,atomj,Di,Dj,Dbi,Dbj) # over correction 

         Di_boc = Di - self.p['val_'+atomi]        # bo correction
         Dj_boc = Di - self.p['val_'+atomi]
         Fi   = self.f_nn(flabel,b[0],self.nbd[bd],[Di_boc,h,Dj_boc],layer=self.mf_layer[1])
         Fj   = self.f_nn(flabel,b[1],self.nbd[bd],[Dj_boc,h,Di_boc],layer=self.mf_layer[1])

         #one  = lambda:1.0
         #F1   = lambda:f_1*f_1
         #f_11 = tf.cond(pred=tf.greater_equal(self.p['ovcorr_'+bd],0.0001),true_fn=F1,false_fn=one)
         F    = Fi*Fj#*f_11
         
         # By default p_corr13 is always True
         bosi = self.Hsi[t-1][bd]*F
         bopi = self.Hpi[t-1][bd]*F
         bopp = self.Hpp[t-1][bd]*F 
      elif self.MessageFunction==5:
         Dbi  = Di - self.p['val_'+atomi] # Di-h
         Dbj  = Dj - self.p['val_'+atomj] # Dj-h
         Fi   = self.fmessage(flabel,b[0],self.nbd[bd],[Dbi,h,Dbj],layer=self.mf_layer[1])
         Fj   = self.fmessage(flabel,b[1],self.nbd[bd],[Dbj,h,Dbi],layer=self.mf_layer[1])
         F    = Fi*Fj
         Fsi,Fpi,Fpp = tf.unstack(F,axis=2)

         bosi = Fsi
         bopi = Fpi
         bopp = Fpp
      else:
         raise NotImplementedError('-  Message function not supported yet!')
      bo = bosi+bopi+bopp
      return bo,bosi,bopi,bopp

  def f1(self,bd,atomi,atomj,Di,Dj,Div,Djv):
      #Div = Di - self.p['val_'+atomi]   
      #Djv = Dj - self.p['val_'+atomj] 
      self.f2(bd,Div,Djv)
      self.f3(bd,Div,Djv)
      f_1 = 0.5*(tf.math.divide(self.p['val_'+atomi]+self.f_2[bd],
                          self.p['val_'+atomi]+self.f_2[bd]+self.f_3[bd]) + 
                 tf.math.divide(self.p['val_'+atomj]+self.f_2[bd],
                          self.p['val_'+atomj]+self.f_2[bd]+self.f_3[bd]))
      return f_1

  def get_ebond(self,bd):
      b       = bd.split('-')
      bdr     = b[1]+'-'+b[0]
      if self.EnergyFunction==0:
         FBO  = tf.where(tf.greater(self.bosi[bd],0.0),
                         tf.ones_like(self.bosi[bd]),tf.zeros_like(self.bosi[bd]))
         FBOR = 1.0 - FBO
         self.powb[bd] = tf.pow(self.bosi[bd]+FBOR,self.p['be2_'+bd])
         self.expb[bd] = tf.exp(tf.multiply(self.p['be1_'+bd],1.0-self.powb[bd]))

         self.sieng[bd] = self.p['Desi_'+bd]*self.bosi[bd]*self.expb[bd]*FBO 
         self.pieng[bd] = tf.multiply(self.p['Depi_'+bd],self.bopi[bd])
         self.ppeng[bd] = tf.multiply(self.p['Depp_'+bd],self.bopp[bd]) 
         self.esi[bd]   = self.sieng[bd] + self.pieng[bd] + self.ppeng[bd]
         self.EBD[bd]   = -self.esi[bd]
      elif self.EnergyFunction==1:
         self.esi[bd] = self.f_nn('fe',bd, self.nbd[bd],[self.bosi[bd],self.bopi[bd],self.bopp[bd]],
                                  layer=self.be_layer[1])
         self.EBD[bd] = -self.p['Desi_'+bd]*self.esi[bd]
      elif self.EnergyFunction==2:
         self.esi[bd] = self.f_nn('fe',bd, self.nbd[bd],[-self.bosi[bd],-self.bopi[bd],-self.bopp[bd]],
                                  layer=self.be_layer[1])
         self.EBD[bd] = -self.p['Desi_'+bd]*self.esi[bd]
      elif self.EnergyFunction==3:
         self.esi[bd] = self.f_nn('fe',bd, self.nbd[bd],[self.bosi[bd],self.bopi[bd],self.bopp[bd]],
                                  layer=self.be_layer[1])
         self.EBD[bd] = -self.p['Desi_'+bd]*self.esi[bd]*self.bo0[bd]
      elif self.EnergyFunction==4:
         Di      = tf.gather_nd(self.Delta,self.dilink[bd])
         Dj      = tf.gather_nd(self.Delta,self.djlink[bd])
         Dbi = Di-self.bo0[bd]
         Dbj = Dj-self.bo0[bd]
         Fi  = self.f_nn('fe',bd,self.nbd[bd],[Dbi,Dbj,self.bo0[bd]],layer=self.be_layer[1])
         Fj  = self.f_nn('fe',bd,self.nbd[bd],[Dbj,Dbi,self.bo0[bd]],layer=self.be_layer[1])
         self.esi[bd] = Fi*Fj # *self.bo0[bd]
         self.EBD[bd] = -self.p['Desi_'+bd]*self.esi[bd]
      # elif self.EnergyFunction==5:
      #    r_        = self.rbd[bd]/self.p['rosi_'+bd]
      #    mors_exp1 = tf.exp(self.p['be2_'+bd]*(1.0-r_))
      #    mors_exp2 = tf.square(mors_exp1) 

      #    mors_exp10 = tf.exp(self.p['be2_'+bd]*self.p['be1_'+bd]) 
      #    mors_exp20 = tf.square(mors_exp10)

      #    emorse     = 2.0*mors_exp1 - mors_exp2 + mors_exp20 - 2.0*mors_exp10
      #    self.esi[bd] = tf.nn.relu(emorse)
      #    self.EBD[bd] = -self.p['Desi_'+bd]*self.esi[bd]
      elif self.EnergyFunction==5:
         self.sieng[bd] = tf.multiply(self.p['Desi_'+bd],self.bosi[bd])
         self.pieng[bd] = tf.multiply(self.p['Depi_'+bd],self.bopi[bd])
         self.ppeng[bd] = tf.multiply(self.p['Depp_'+bd],self.bopp[bd]) 
         self.esi[bd]   = self.sieng[bd] + self.pieng[bd] - self.ppeng[bd]
         self.EBD[bd]   = -self.esi[bd]
      else:
         raise NotImplementedError('-  This method is not implimented!')

  def get_bond_energy(self):
      self.get_delta()

      self.message_passing()
      self.get_final_sate()

      i = 0                           # get bond energy
      for bd in self.bonds: 
          [atomi,atomj] = bd.split('-') 
          if self.nbd[bd]>0:
             [atomi,atomj] = bd.split('-') 
             self.get_ebond(bd)
             EBDA = self.EBD[bd] if i==0 else tf.concat((EBDA,self.EBD[bd]),0)
             i += 1

      for mol in self.mols:
          self.ebda[mol] = tf.gather_nd(EBDA,self.bdlink[mol])  
          self.ebond[mol]= tf.reduce_sum(input_tensor=self.ebda[mol],axis=0,name='bondenergy')

  def message_passing(self):
      ''' finding the final Bond－order with a message passing '''
      self.H   = [self.bop]                     # 
      self.Hsi = [self.bop_si]                  #
      self.Hpi = [self.bop_pi]                  #
      self.Hpp = [self.bop_pp]                  # 
      self.D   = [self.Deltap]                  # get the initial hidden state H[0]
      if self.MessageFunction==1:
         self.D_si = [self.Dpsi]
         self.D_pi = [self.Dppi]
         self.D_pp = [self.Dppp]

      for t in range(1,self.messages+1):
          print('-  message passing for t=%d ...' %t)
          self.H.append({})                     # get the hidden state H[t]
          self.Hsi.append({})                   #
          self.Hpi.append({})                   #
          self.Hpp.append({})                   #             

          if self.MessageFunction==1:
             BOSI = tf.zeros([1,self.batch]) 
             BOPI = tf.zeros([1,self.batch]) 
             BOPP = tf.zeros([1,self.batch]) 
          else:
             BO    = tf.zeros([1,self.batch])         # for ghost atom, the value is zero

          for bd in self.bonds:
              if self.nbd[bd]>0:
                 atomi,atomj = bd.split('-') 
                 bo,bosi,bopi,bopp = self.get_bondorder(t,bd,atomi,atomj)

                 self.H[t][bd]   = bo
                 self.Hsi[t][bd] = bosi
                 self.Hpi[t][bd] = bopi
                 self.Hpp[t][bd] = bopp

                 if self.MessageFunction==1:
                    BOSI = tf.concat([BOSI,bosi],0)
                    BOPI = tf.concat([BOPI,bosi],0)
                    BOPP = tf.concat([BOPP,bosi],0)
                 else:
                    BO = tf.concat([BO,bo],0)
      
          if self.MessageFunction==1:
             Dsi_   = tf.gather_nd(BOSI,self.dlist)  
             Dsi    = tf.reduce_sum(input_tensor=Dsi_,axis=1)
             Dpi_   = tf.gather_nd(BOPI,self.dlist)  
             Dpi    = tf.reduce_sum(input_tensor=Dpi_,axis=1)
             Dpp_   = tf.gather_nd(BOPP,self.dlist)  
             Dpp    = tf.reduce_sum(input_tensor=Dpp_,axis=1)
             Delta  = Dsi + Dpi + Dpp

             self.D_si.append(Dsi) 
             self.D_pi.append(Dpi) 
             self.D_pp.append(Dpp) 
          else:
             D      = tf.gather_nd(BO,self.dlist)  
             Delta  = tf.reduce_sum(input_tensor=D,axis=1)
          self.D.append(Delta)                  # degree matrix
           
       
  def get_final_sate(self):     
      self.Delta  = self.D[-1]
      self.bo0    = self.H[-1]                  # fetch the final state 
      self.bosi   = self.Hsi[-1]
      self.bopi   = self.Hpi[-1]
      self.bopp   = self.Hpp[-1]

      self.BO0    = tf.zeros([1,self.batch])    # for ghost atom, the value is zero
      self.BO     = tf.zeros([1,self.batch])
      self.BOPI   = tf.zeros([1,self.batch])
      self.BSO    = tf.zeros([1,self.batch])
      BPI         = tf.zeros([1,self.batch])

      for bd in self.bonds:
          if self.nbd[bd]>0:
             # self.fbo[bd]  = taper(self.bo0[bd],rmin=self.botol,rmax=2.0*self.botol)
             self.bo[bd]   = tf.nn.relu(self.bo0[bd] - self.atol)
             self.bso[bd]  = self.p['ovun1_'+bd]*self.p['Desi_'+bd]*self.bo0[bd] 

             self.BO0 = tf.concat([self.BO0,self.bo0[bd]],0)
             self.BO  = tf.concat([self.BO,self.bo[bd]],0)
             self.BSO = tf.concat([self.BSO,self.bso[bd]],0)
             BPI      = tf.concat([BPI,self.bopi[bd]+self.bopp[bd]],0)
             self.BOPI= tf.concat([self.BOPI,self.bopi[bd]],0)

      D_  = tf.gather_nd(self.BO0,self.dlist,name='D_') 
      self.Delta  = tf.reduce_sum(input_tensor=D_,axis=1,name='Delta')  # without valence i.e. - Val 
       
      SO_ = tf.gather_nd(self.BSO,self.dlist,name='SO_') 
      self.BPI = tf.gather_nd(BPI,self.dlist,name='BPI') 
      
      self.SO     = tf.reduce_sum(input_tensor=SO_,axis=1,name='sumover')  
      self.FBOT   = taper(self.BO0,rmin=self.atol,rmax=2.0*self.atol) 
      self.FHB    = taper(self.BO0,rmin=self.hbtol,rmax=2.0*self.hbtol) 

      ####
      # if self.spv_bo: # self.MessageFunction==3:
      #    N_           = tf.math.ceil(D_ - self.p['cutoff'])             
      #    self.N       = tf.reduce_sum(input_tensor=N_,axis=1,name='N')          # N  整数版Delta 
         
      #    for bd in self.bonds:
      #        atomi,atomj = bd.split('-') 
      #        if self.nbd[bd]>0: 
      #           boc   = tf.math.ceil(self.bo0[bd] - self.p['cutoff'])
      #           Ni    = self.p['val_'+atomi] - tf.gather_nd(self.N,self.dilink[bd]) + boc
      #           Nj    = self.p['val_'+atomj] - tf.gather_nd(self.N,self.djlink[bd]) + boc
      #           Si    = tf.where(tf.greater(Ni,0.0001),-0.5,0.5)
      #           Sj    = tf.where(tf.greater(Nj,0.0001),-0.5,0.5)
      #           self.S[bd] = Si + Sj

  def set_m(self):
      ''' set variable for neural networks '''
      self.m = {}
      bond   = []               # make sure the m matrix is unique 
      for si in self.spec:
          for sj in self.spec:
              bd = si + '-' + sj
              if bd not in bond:
                 bond.append(bd)

      if self.mfopt is None:
         self.mfopt = self.spec
      if self.bdopt is None:
         self.bdopt = self.bonds

      self.get_univeral_nn()

      reuse_m = True if (self.bo_layer==self.bo_layer_ and self.BOFunction_==self.BOFunction) else False
      
      if self.BOFunction!=0:
         for p_ in ['fsi','fpi','fpp']:
             if not self.bo_univeral_nn is None:
                self.set_univeral_wb(pref=p_,bd=self.bo_univeral_nn[0],reuse_m=reuse_m,
                                       nin=1,nout=1,layer=self.bo_layer,
                                       nnopt=self.mpopt[0],bias=-1.0)
             self.set_wb(pref=p_,reuse_m=reuse_m,nin=1,nout=1,layer=self.bo_layer,
                         vlist=self.bonds,nnopt=self.mpopt[0],bias=-1.0)
 
      ############ set weight and bias for message neural network ###################
      if (self.mf_layer==self.mf_layer_ and  self.EnergyFunction==self.EnergyFunction_
          and self.MessageFunction_==self.MessageFunction):
         reuse_m = True  
      else:
         reuse_m = False

      nout_ = 3 if self.MessageFunction!=4 else 1
      if self.MessageFunction==1:
         nin_  = 5
      elif self.MessageFunction==5 :
         nin_  = 3 
      elif self.MessageFunction==2:
         nin_  = 5 
      else:
         nin_  = 3

      for t in range(1,self.messages+1):
          b = 0.881373587 if t>1 else -0.867
          if not self.mf_univeral_nn is None:
             self.set_univeral_wb(pref='fm',bd=self.mf_univeral_nn[0],reuse_m=reuse_m,# +str(t)
                                    nin=nin_,nout=nout_,layer=self.mf_layer,
                                    nnopt=self.mpopt[t],bias=b)
          self.set_message_wb(pref='fm',reuse_m=reuse_m,nin=nin_,nout=nout_,   # +str(t)
                              layer=self.mf_layer,nnopt=self.mpopt[t],bias=b) 

      ############ set weight and bias for energy neural network ###################
      if self.EnergyFunction==self.EnergyFunction_ and self.be_layer==self.be_layer_:
         reuse_m = True  
      else:
         reuse_m = False 

      nin_ = 3 # 4 if self.EnergyFunction==1 else 3

      if not self.be_univeral_nn is None:
         self.set_univeral_wb(pref='fe',bd=self.be_univeral_nn[0],reuse_m=reuse_m,
                                nin=nin_,nout=1,layer=self.be_layer,
                                nnopt=self.mpopt[t+1],bias=2.0)
      self.set_wb(pref='fe',reuse_m=reuse_m,nin=nin_,nout=1,layer=self.be_layer,
                  vlist=self.bonds,nnopt=self.mpopt[t+1],bias=2.0)

      nin_ = 1 if self.VdwFunction==1 else 3

      if self.vdwnn:
         reuse_m = True if self.vdw_layer==self.vdw_layer_ and self.VdwFunction==self.VdwFunction_ else False
         if not self.vdw_univeral_nn is None:
            self.set_univeral_wb(pref='fv',bd=self.vdw_univeral_nn[0],reuse_m=reuse_m,
                                   nin=nin_,nout=1,layer=self.vdw_layer,
                                   nnopt=self.mpopt[t+1],bias=0.867)
         self.set_wb(pref='fv',reuse_m=reuse_m,nin=nin_,nout=1,layer=self.vdw_layer,
                     vlist=self.bonds,nnopt=self.mpopt[-1],bias=0.867)

  def set_wb(self,pref='f',reuse_m=True,nnopt=True,nin=8,nout=3,layer=[8,9],vlist=None,bias=0.0):
      ''' set matix varibles '''
      if self.m_ is None:
         self.m_ = {}
      for bd in vlist:
          if pref+'_'+bd in self.univeral_nn:
             self.m[pref+'wi_'+bd] = self.m[pref+'wi']
             self.m[pref+'bi_'+bd] = self.m[pref+'bi']
          elif pref+'wi_'+bd in self.m_ and reuse_m:                   # input layer
             if nnopt and (bd in self.bdopt):
                self.m[pref+'wi_'+bd] = tf.Variable(self.m_[pref+'wi_'+bd],name=pref+'wi_'+bd)
                self.m[pref+'bi_'+bd] = tf.Variable(self.m_[pref+'bi_'+bd],name=pref+'bi_'+bd)
             else:
                self.m[pref+'wi_'+bd] = tf.constant(self.m_[pref+'wi_'+bd],name=pref+'wi_'+bd)
                self.m[pref+'bi_'+bd] = tf.constant(self.m_[pref+'bi_'+bd],name=pref+'bi_'+bd)
          else:
             self.m[pref+'wi_'+bd] = tf.Variable(tf.random.normal([nin,layer[0]],stddev=0.2),
                                                 name=pref+'wi_'+bd)   
             self.m[pref+'bi_'+bd] = tf.Variable(tf.random.normal([layer[0]],stddev=0.1),
                                                 name=pref+'bi_'+bd)  

          self.m[pref+'w_'+bd] = []                                    # hidden layer
          self.m[pref+'b_'+bd] = []
          if pref+'_'+bd in self.univeral_nn:
             for i in range(layer[1]):  
                 self.m[pref+'w_'+bd] = self.m[pref+'w']
                 self.m[pref+'b_'+bd] = self.m[pref+'b']
          elif pref+'w_'+bd in self.m_ and reuse_m:     
             if nnopt and (bd in self.bdopt):                            
                for i in range(layer[1]):   
                    self.m[pref+'w_'+bd].append(tf.Variable(self.m_[pref+'w_'+bd][i],name=pref+'wh'+str(i)+'_'+bd )) 
                    self.m[pref+'b_'+bd].append(tf.Variable(self.m_[pref+'b_'+bd][i],name=pref+'bh'+str(i)+'_'+bd )) 
             else:
                for i in range(layer[1]):   
                    self.m[pref+'w_'+bd].append(tf.constant(self.m_[pref+'w_'+bd][i],name=pref+'wh'+str(i)+'_'+bd )) 
                    self.m[pref+'b_'+bd].append(tf.constant(self.m_[pref+'b_'+bd][i],name=pref+'bh'+str(i)+'_'+bd )) 
          else:
             for i in range(layer[1]):   
                 self.m[pref+'w_'+bd].append(tf.Variable(tf.random.normal([layer[0],layer[0]], 
                                                         stddev=0.2),name=pref+'wh'+str(i)+'_'+bd)) 
                 self.m[pref+'b_'+bd].append(tf.Variable(tf.random.normal([layer[0]], 
                                                         stddev=0.1),name=pref+'bh'+str(i)+'_'+bd )) 
          
          if pref+'_'+bd in self.univeral_nn:
             self.m[pref+'wo_'+bd] = self.m[pref+'wo']
             self.m[pref+'bo_'+bd] = self.m[pref+'bo']
          elif pref+'wo_'+bd in self.m_ and reuse_m:          # output layer
              if nnopt and (bd in self.bdopt):       
                 self.m[pref+'wo_'+bd] = tf.Variable(self.m_[pref+'wo_'+bd],name=pref+'wo_'+bd)
                 self.m[pref+'bo_'+bd] = tf.Variable(self.m_[pref+'bo_'+bd],name=pref+'bo_'+bd)
              else:
                 self.m[pref+'wo_'+bd] = tf.constant(self.m_[pref+'wo_'+bd],name=pref+'wo_'+bd)
                 self.m[pref+'bo_'+bd] = tf.constant(self.m_[pref+'bo_'+bd],name=pref+'bo_'+bd)
          else:
              self.m[pref+'wo_'+bd] = tf.Variable(tf.random.normal([layer[0],nout],stddev=0.2), name=pref+'wo_'+bd)   
              self.m[pref+'bo_'+bd] = tf.Variable(tf.random.normal([nout], stddev=0.01)+bias,
                                                  name=pref+'bo_'+bd)
  
  def set_message_wb(self,pref='f',reuse_m=True,nnopt=True,nin=8,nout=3,layer=[8,9],bias=0.0):
      ''' set matix varibles '''
      if self.m_ is None:
         self.m_ = {}
      for sp in self.spec:
          self.m[pref+'w_'+sp] = []                                    
          self.m[pref+'b_'+sp] = []
          if pref+'_'+sp in self.univeral_nn:
             self.m[pref+'wi_'+sp] = self.m[pref+'wi']
             self.m[pref+'bi_'+sp] = self.m[pref+'bi']
             self.m[pref+'wo_'+sp] = self.m[pref+'wo']
             self.m[pref+'bo_'+sp] = self.m[pref+'bo']
             self.m[pref+'w_'+sp]  = self.m[pref+'w'] 
             self.m[pref+'b_'+sp]  = self.m[pref+'b'] 
          elif pref+'wi_'+sp in self.m_ and reuse_m:
             if nnopt and (sp in self.mfopt):
                self.m[pref+'wi_'+sp] = tf.Variable(self.m_[pref+'wi_'+sp],name=pref+'wi_'+sp)
                self.m[pref+'bi_'+sp] = tf.Variable(self.m_[pref+'bi_'+sp],name=pref+'bi_'+sp)
                self.m[pref+'wo_'+sp] = tf.Variable(self.m_[pref+'wo_'+sp],name=pref+'wo_'+sp)
                self.m[pref+'bo_'+sp] = tf.Variable(self.m_[pref+'bo_'+sp],name=pref+'bo_'+sp)
                for i in range(layer[1]):   
                    self.m[pref+'w_'+sp].append(tf.Variable(self.m_[pref+'w_'+sp][i],name=pref+'wh'+str(i)+'_'+sp )) 
                    self.m[pref+'b_'+sp].append(tf.Variable(self.m_[pref+'b_'+sp][i],name=pref+'bh'+str(i)+'_'+sp )) 
             else:
                self.m[pref+'wi_'+sp] = tf.constant(self.m_[pref+'wi_'+sp],name=pref+'wi_'+sp)
                self.m[pref+'bi_'+sp] = tf.constant(self.m_[pref+'bi_'+sp],name=pref+'bi_'+sp)
                self.m[pref+'wo_'+sp] = tf.constant(self.m_[pref+'wo_'+sp],name=pref+'wo_'+sp)
                self.m[pref+'bo_'+sp] = tf.constant(self.m_[pref+'bo_'+sp],name=pref+'bo_'+sp)
                for i in range(layer[1]):   
                    self.m[pref+'w_'+sp].append(tf.constant(self.m_[pref+'w_'+sp][i],name=pref+'wh'+str(i)+'_'+sp )) 
                    self.m[pref+'b_'+sp].append(tf.constant(self.m_[pref+'b_'+sp][i],name=pref+'bh'+str(i)+'_'+sp )) 
          else:
             self.m[pref+'wi_'+sp] = tf.Variable(tf.random.normal([nin,layer[0]],stddev=0.9),name=pref+'wi_'+sp)   
             self.m[pref+'bi_'+sp] = tf.Variable(tf.random.normal([layer[0]],stddev=0.9),name=pref+'bi_'+sp)  
             self.m[pref+'wo_'+sp] = tf.Variable(tf.random.normal([layer[0],nout],stddev=0.9),name=pref+'wo_'+sp)   
             self.m[pref+'bo_'+sp] = tf.Variable(tf.random.normal([nout],stddev=0.9),name=pref+'bo_'+sp)  
             for i in range(layer[1]):   
                 self.m[pref+'w_'+sp].append(tf.Variable(tf.random.normal([layer[0],layer[0]], 
                                                         stddev=0.9),name=pref+'wh'+str(i)+'_'+sp)) 
                 self.m[pref+'b_'+sp].append(tf.Variable(tf.random.normal([layer[0]], 
                                                         stddev=0.9),name=pref+'bh'+str(i)+'_'+sp)) 

  def set_univeral_wb(self,pref='f',bd='C-C',reuse_m=True,nnopt=True,nin=8,nout=3,
                         layer=[8,9],bias=0.0):
      ''' set universial matix varibles '''
      if self.m_ is None:
         self.m_ = {}

      self.m[pref+'w'] = []                                    # hidden layer
      self.m[pref+'b'] = []

      if pref+'wi' in self.m_:
          bd_ = ''
      else:
          bd_ = '_' + bd
      
      if reuse_m and pref+'wi'+bd_ in self.m_:                   # input layer
         if nnopt:
            self.m[pref+'wi'] = tf.Variable(self.m_[pref+'wi'+bd_],name=pref+'wi')
            self.m[pref+'bi'] = tf.Variable(self.m_[pref+'bi'+bd_],name=pref+'bi')
            self.m[pref+'wo'] = tf.Variable(self.m_[pref+'wo'+bd_],name=pref+'wo')
            self.m[pref+'bo'] = tf.Variable(self.m_[pref+'bo'+bd_],name=pref+'bo')
            for i in range(layer[1]):   
                self.m[pref+'w'].append(tf.Variable(self.m_[pref+'w'+bd_][i],
                                                    name=pref+'wh'+str(i))) 
                self.m[pref+'b'].append(tf.Variable(self.m_[pref+'b'+bd_][i],
                                                    name=pref+'bh'+str(i))) 
         else:
            self.m[pref+'wi'] = tf.constant(self.m_[pref+'wi'+bd_],name=pref+'wi')
            self.m[pref+'bi'] = tf.constant(self.m_[pref+'bi'+bd_],name=pref+'bi')
            self.m[pref+'wo'] = tf.constant(self.m_[pref+'wo'+bd_],name=pref+'wo')
            self.m[pref+'bo'] = tf.constant(self.m_[pref+'bo'+bd_],name=pref+'bo')
            for i in range(layer[1]):   
                self.m[pref+'w'].append(tf.constant(self.m_[pref+'w'+bd_][i],
                                                    name=pref+'wh'+str(i))) 
                self.m[pref+'b'].append(tf.constant(self.m_[pref+'b'+bd_][i],
                                                    name=pref+'bh'+str(i))) 
      else:
         self.m[pref+'wi'] = tf.Variable(tf.random.normal([nin,layer[0]],stddev=0.9),name=pref+'wi')   
         self.m[pref+'bi'] = tf.Variable(tf.random.normal([layer[0]],stddev=0.9),name=pref+'bi')  
         self.m[pref+'wo'] = tf.Variable(tf.random.normal([layer[0],nout],stddev=0.9),name=pref+'wo')   
         self.m[pref+'bo'] = tf.Variable(tf.random.normal([nout], stddev=0.9)+bias,name=pref+'bo')
         for i in range(layer[1]):   
             self.m[pref+'w'].append(tf.Variable(tf.random.normal([layer[0],layer[0]], 
                                                 stddev=0.9),name=pref+'wh'+str(i))) 
             self.m[pref+'b'].append(tf.Variable(tf.random.normal([layer[0]], 
                                                 stddev=0.9),name=pref+'bh'+str(i))) 

  def get_univeral_nn(self):
      self.univeral_nn = []
      if not self.bo_univeral_nn is None:
         if self.bo_univeral_nn=='all':
            univeral_bonds = self.bonds
         else:
            univeral_bonds = self.bo_univeral_nn
         for bd in univeral_bonds:
             b = bd.split('-')
             bdr = b[1] + '-' + b[0]
             self.univeral_nn.append('fsi_'+bd)
             self.univeral_nn.append('fpi_'+bd)
             self.univeral_nn.append('fpp_'+bd)
             self.univeral_nn.append('fsi_'+bdr)
             self.univeral_nn.append('fpi_'+bdr)
             self.univeral_nn.append('fpp_'+bdr)

      if not self.be_univeral_nn is None:
         if self.be_univeral_nn=='all':
            univeral_bonds = self.bonds
         else:
            univeral_bonds = self.be_univeral_nn
         for bd in univeral_bonds:
             b = bd.split('-')
             bdr = b[1] + '-' + b[0]
             self.univeral_nn.append('fe_'+bd)
             self.univeral_nn.append('fe_'+bdr)

      if not self.vdw_univeral_nn is None:
         if self.vdw_univeral_nn=='all':
            univeral_bonds = self.bonds
         else:
            univeral_bonds = self.vdw_univeral_nn
         for bd in univeral_bonds:
             b = bd.split('-')
             bdr = b[1] + '-' + b[0]
             self.univeral_nn.append('fv_'+bd)
             self.univeral_nn.append('fv_'+bdr)

      if not self.mf_univeral_nn is None:
         if self.mf_univeral_nn=='all':
            univeral_bonds = self.spec
         else:
            univeral_bonds = self.mf_univeral_nn
         for sp in univeral_bonds:
             for t in range(1,self.messages+1):
                 self.univeral_nn.append('fm'+'_'+sp) # +str(t)

  def write_lib(self,libfile='ffield',loss=None):
      p_   = self.sess.run(self.p)
      self.p_ = {}
      
      self.MolEnergy_ = self.sess.run(self.MolEnergy)
      for key in self.MolEnergy_:
          self.MolEnergy_[key] = float(self.MolEnergy_[key])
          
      if not loss is None:
         self.p_['score'] = -loss

      for k in p_:
          key = k.split('_')[0]
          if key in ['V1','V2','V3','tor1','cot1']:
             k_ = k.split('_')[1]
             if k_ not in self.torp:
                continue

          self.p_[k] = float(p_[k])
          if key in self.punit:
             self.p_[k] = float(p_[k]/self.unit)

      if self.libfile.endswith('.json'):
         self.m_   = self.sess.run(self.m)

         for key in self.m_:
             k = key.split('_')[0]
             if k[0]=='f' and (k[-1]=='w' or k[-1]=='b'):
                for i,M in enumerate(self.m_[key]):
                    # if isinstance(M, np.ndarray):
                    self.m_[key][i] = M.tolist()
             else:
                self.m_[key] = self.m_[key].tolist()  # covert ndarray to list
         # print(' * save parameters to file ...')
         fj = open(libfile+'.json','w')
         j = {'p':self.p_,'m':self.m_,
              'BOFunction':self.BOFunction,
              'EnergyFunction':self.EnergyFunction,
              'MessageFunction':self.MessageFunction, 
              'VdwFunction':self.VdwFunction,
              'messages':self.messages,
              'bo_layer':self.bo_layer,
              'mf_layer':self.mf_layer,
              'be_layer':self.be_layer,
              'vdw_layer':self.vdw_layer,
              'rcut':self.rcut,
              'rcutBond':self.rcuta,
              'rEquilibrium':self.re,
              'MolEnergy':self.MolEnergy_}
         js.dump(j,fj,sort_keys=True,indent=2)
         fj.close()
      elif self.libfile.endswith('.lib'):
         write_lib(self.p_,self.spec,self.bonds,self.offd,
                   self.angs,self.torp,self.hbs,
                   libfile=libfile)
      else:
         write_ffield(self.p_,self.spec,self.bonds,self.offd,
                      self.angs,self.torp,self.hbs,
                      zpe=self.zpe_,libfile=libfile,
                      loss=loss)

  def supervise(self):
      ''' adding some penalty term to accelerate the training '''
      log_    = -9.21044036697651
      penalty = 0.0
      pen_w   = 0.0
      pen_b   = 0.0
      wb_p    = []
      if self.regularize_be:
         wb_p.append('fe')
      if self.regularize_bo and self.BOFunction:
         wb_p.append('fsi')
         wb_p.append('fpi')
         wb_p.append('fpp')

      if self.vdwnn and self.regularize_vdw:
         wb_p.append('fv')

      w_n     = ['wi','wo',]
      b_n     = ['bi','bo']
      layer   = {'fe':self.be_layer[1],'fsi':self.bo_layer[1],'fpi':self.bo_layer[1],
                 'fpp':self.bo_layer[1]}
      if self.vdwnn:
         layer['fv'] = self.vdw_layer[1]

      wb_message = []
      if self.regularize_mf:
         for t in range(1,self.messages+1):
             wb_message.append('fm')#+str(t)
             layer['fm'] = self.mf_layer[1]#+str(t)

      self.penalty_bop,self.penalty_bo = {},{}
      self.penalty_bo_rcut = {}
      self.penalty_pi,self.penalty_ang = {},{}
      self.penalty_vdw = {} 
      self.penalty_be_cut,self.penalty_be = {},{}
      self.penalty_s_bo,self.penalty_s = {},{}
      self.penalty_rcut = {}

      for bd in self.bonds: 
          atomi,atomj = bd.split('-') 
          bdr = atomj + '-' + atomi
          # log_ = tf.math.log((self.botol/(1.0 + self.botol)))
          if self.fixrcbo:
             rcut_si = tf.square(self.rc_bo[bd]-self.rcut[bd])
          else:
             rcut_si = tf.nn.relu(self.rc_bo[bd]-self.rcut[bd])

          rc_bopi = self.p['ropi_'+bd]*tf.pow(log_/self.p['bo3_'+bd],1.0/self.p['bo4_'+bd])
          rcut_pi = tf.nn.relu(rc_bopi-self.rcut[bd])

          rc_bopp = self.p['ropp_'+bd]*tf.pow(log_/self.p['bo5_'+bd],1.0/self.p['bo6_'+bd])
          rcut_pp = tf.nn.relu(rc_bopp-self.rcut[bd])

          self.penalty_rcut[bd] = rcut_si + rcut_pi + rcut_pp
          penalty = tf.add(self.penalty_rcut[bd]*self.lambda_bd,penalty)

          if self.nbd[bd]>0:
             bd_ = bd  if bd in self.bore else 'others'

             fao = tf.where(tf.greater(self.rbd[bd],self.rcuta[bd]),1.0,0.0)    ##### r> rcuta that bo = 0.0
             self.penalty_bo_rcut[bd]  = tf.reduce_sum(self.bo0[bd]*fao)
             penalty = tf.add(self.penalty_bo_rcut[bd]*self.lambda_bd,penalty)

             # if self.MessageFunction==1:
             #    pen_b    = tf.reduce_sum(self.bo0[bd]*fbo) 
             #    penalty  = tf.add(diffb_*self.lambda_bd,penalty)
             # bop_nn = nn(bop) MUST BE zero if r>rc_bo
             # if self.BOFunction!=0:
             fbo = tf.where(tf.less(self.rbd[bd],self.rc_bo[bd]),0.0,1.0)     ##### bop should be zero if r>rcut_bo
             self.penalty_bop[bd]  = tf.reduce_sum(self.bop[bd]*fbo)          #####  
             penalty  = tf.add(self.penalty_bop[bd]*self.lambda_bd,penalty)

             if self.spv_be:
                # if self.MessageFunction==1:
                # fe  = tf.where(tf.less_equal(self.rbd[bd],self.rc_bo[bd]),1.0,0.0) ##### r< r_e that bo > bore_
                # else:
                #   fe  = tf.where(tf.less_equal(self.rbd[bd],self.rc_bo[bd]),1.0,0.0) 
                self.penalty_be[bd] = tf.constant(0.0)
                if (bd in self.beup) or (bdr in self.beup):
                   bd_ = bd if bd in self.beup else bdr
                   for beup_ in self.beup[bd_]:
                       r_,be_u = beup_  
                       if isinstance(be_u,str):
                          if be_u=='si':
                             be = self.bop_si[bd]
                          elif be_u=='pi':
                             be = self.bop_pi[bd]
                          elif be_u=='pp':
                             be = self.bop_pp[bd]
                          else:
                             raise NotImplementedError('-  This format not supported yet!')
                          fu      = tf.where(tf.greater(self.rbd[bd],r_),1.0,0.0) ##### 
                          fluct_u = 1.0+self.fluctuation
                          pen_e = tf.reduce_sum(input_tensor=tf.nn.relu((self.esi[bd] - be*fluct_u)*fu))
                       else:
                         fu      = tf.where(tf.less_equal(self.rbd[bd],r_),1.0,0.0) ##### 
                         pen_e   = tf.reduce_sum(input_tensor=tf.nn.relu((self.EBD[bd] - be_u)*fu))
                       self.penalty_be[bd] = self.penalty_be[bd] + pen_e
                       
                if (bd in self.belo) or (bdr in self.belo):
                   bd_ = bd if bd in self.belo else bdr
                   for belo_ in self.belo[bd_]:
                       r_,be_l = belo_
                       
                       if isinstance(be_l,str):
                          if be_l=='si':
                             be = self.bop_si[bd]
                          elif be_l=='pi':
                             be = self.bop_pi[bd]
                          elif be_l=='pp':
                             be = self.bop_pp[bd]
                          else:
                             raise NotImplementedError('-  This format not supported yet!')
                          fl      = tf.where(tf.less_equal(self.rbd[bd],r_),1.0,0.0)   #####
                          fluct_l = 1.0-self.fluctuation
                          pen_e = tf.reduce_sum(input_tensor=tf.nn.relu((be*fluct_l-self.esi[bd])*fl))
                       else:
                          fl      = tf.where(tf.greater(self.rbd[bd],r_),1.0,0.0)   #####
                          pen_e   = tf.reduce_sum(input_tensor=tf.nn.relu((be_l - self.EBD[bd])*fl))
                       self.penalty_be[bd] = self.penalty_be[bd] + pen_e
                # if not self.beup and not self.belo:
                #    fe  = tf.where(tf.less_equal(self.rbd[bd],self.rc_bo[bd]),1.0,0.0) ##### r< r_e that bo > bore_
                #    fluct_l = 1.0-self.fluctuation
                #    pen_e = tf.reduce_sum(input_tensor=tf.nn.relu((self.bop_pi[bd]*fluct_l-self.esi[bd])*fe))
                #    self.penalty_be[bd] = self.penalty_be[bd] + pen_e

                #    fluct_u = 1.0+self.fluctuation
                #    pen_e = tf.reduce_sum(input_tensor=tf.nn.relu((self.esi[bd]-self.bop_pi[bd]*fluct_u)*fe))
                #    self.penalty_be[bd] = self.penalty_be[bd] + pen_e
                penalty = tf.add(self.penalty_be[bd]*self.lambda_bd,penalty) 

             # if self.MessageFunction==3:
             #    fe_ = tf.where(tf.greater(self.S[bd],0.00001),1.0,0.0) 
             #    self.penalty_s[bd]  = tf.reduce_sum(input_tensor=self.bo0[bd]*fe_)
             #    penalty  = tf.add(self.penalty_s[bd]*self.lambda_bd,penalty)    ## penalty iterm for bond order

             if self.spv_bo:
                #fe_ = tf.where(tf.greater(self.S[bd],0.00001),1.0,0.0) 
                #self.penalty_s[bd]  = tf.reduce_sum(input_tensor=self.bo0[bd]*fe_)
                #penalty  = tf.add(self.penalty_s[bd]*self.lambda_bd,penalty)      ## penalty iterm for bond order
                self.penalty_bo[bd] = tf.constant(0.0)
                if (bd in self.spv_bo) or (bdr in self.spv_bo):
                   bd_  = bd if bd in self.spv_bo else bdr
                   r    = self.spv_bo[bd_][0]
                   bo_l = self.spv_bo[bd_][1]
                   bo_u = self.spv_bo[bd_][2]
                   fe   = tf.where(tf.less_equal(self.rbd[bd],r),1.0,0.0) ##### r< r_e that bo > bore_
                   self.penalty_bo[bd] += tf.reduce_sum(input_tensor=tf.nn.relu((bo_l-self.bo0[bd])*fe))
                   fe   = tf.where(tf.greater_equal(self.rbd[bd],r),1.0,0.0) ##### r< r_e that bo > bore_
                   self.penalty_bo[bd] += tf.reduce_sum(input_tensor=tf.nn.relu((self.bo0[bd]-bo_u)*fe))

                penalty  = tf.add(self.penalty_bo[bd]*self.lambda_bd,penalty) 

             if self.EnergyFunction != 3: # or self.EnergyFunction == 4 or self.EnergyFunction == 2:
                fesi = tf.where(tf.less_equal(self.bo0[bd],self.botol),1.0,0.0) ##### bo <= 0.0 that e = 0.0
                self.penalty_be_cut[bd]  = tf.reduce_sum(tf.nn.relu(self.esi[bd]*fesi))
                penalty  = tf.add(self.penalty_be_cut[bd]*self.lambda_bd,penalty)

             # penalize term for regularization of the neural networs
             if self.regularize:                             # regularize to avoid overfit
                for k in wb_p:
                    for k_ in w_n:
                        key     = k + k_ + '_' + bd
                        pen_w  += tf.reduce_sum(tf.square(self.m[key]))
                    if self.regularize_bias:
                       for k_ in b_n:
                           key     = k + k_ + '_' + bd
                           pen_b  += tf.reduce_sum(tf.square(self.m[key]))
                    for l in range(layer[k]):                                               
                        pen_w += tf.reduce_sum(tf.square(self.m[k+'w_'+bd][l]))
                        if self.regularize_bias:
                           pen_b += tf.reduce_sum(tf.square(self.m[k+'b_'+bd][l]))

             if self.spv_vdw and self.nvb[bd]>0:
                self.penalty_vdw[bd] = tf.constant(0.0)
                if (bd in self.vup) or (bdr in self.vup):
                   bd_ = bd if bd in self.vup else bdr
                   for vup_ in self.vup[bd_]:
                       r_,ev_u = vup_  
                       fu      = tf.where(tf.less(self.rv[bd],r_),1.0,0.0) ##### 
                       pen_v   = tf.reduce_sum(input_tensor=tf.nn.relu((self.EVDW[bd] - ev_u)*fu))
                       self.penalty_vdw[bd] = self.penalty_vdw[bd] + pen_v  
                       # penalty  = tf.add(self.penalty_bo_nn[bd]*self.lambda_bd,penalty) 

                if (bd in self.vlo) or (bdr in self.vlo):
                   bd_ = bd if bd in self.vlo else bdr
                   for vlo_ in self.vlo[bd_]:
                       r_,ev_l = vlo_
                       fl      = tf.where(tf.greater(self.rv[bd],r_),1.0,0.0) #####
                       pen_v   = tf.reduce_sum(input_tensor=tf.nn.relu((ev_l - self.EVDW[bd])*fl))
                       self.penalty_vdw[bd] = self.penalty_vdw[bd] + pen_v
                penalty  = tf.add(self.penalty_vdw[bd]*self.lambda_bd,penalty)

          # for sp in self.spec: 
          #     pi_ = self.pim[sp] if sp in self.pim else self.pim['others']
          #     if self.nsp[sp]>0:
          #        self.penalty_pi[sp] = tf.reduce_sum(input_tensor=tf.nn.relu(self.Dpi[sp]-pi_))
          #        penalty  = tf.add(self.penalty_pi[sp]*self.lambda_pi,penalty)
      if self.optword.find('noang')<0:
         for ang in self.angs: 
             if self.nang[ang]>0:
                if self.spv_pi:
                   pi_ = self.pim[ang] if ang in self.pim else self.pim['others']
                   self.penalty_pi[ang] = tf.reduce_sum(input_tensor=tf.nn.relu(self.SBO[ang]-pi_))
                   penalty  = tf.add(self.penalty_pi[ang]*self.lambda_pi,penalty)
                if self.spv_ang:
                   self.penalty_ang[ang] = tf.reduce_sum(self.thet2[ang]*self.fijk[ang])
                   penalty  = tf.add(self.penalty_ang[ang]*self.lambda_ang,penalty)

         # self.penalty_pi = tf.reduce_sum(input_tensor=tf.nn.relu(self.DPI-2.0)) 
         # penalty  = tf.add(self.penalty_pi*self.lambda_pi,penalty)
      if self.regularize:                              # regularize
         for sp in self.spec:
             for k in wb_message:
                 for k_ in w_n:
                     key     = k + k_ + '_' + sp
                     pen_w  += tf.reduce_sum(tf.square(self.m[key]))
                 if self.regularize_bias:
                    for k_ in b_n:
                        key     = k + k_ + '_' + sp
                        pen_b  += tf.reduce_sum(tf.square(self.m[key]))
                 for l in range(layer[k]):                                               
                     pen_w += tf.reduce_sum(tf.square(self.m[k+'w_'+sp][l]))
                     if self.regularize_bias:
                        pen_b += tf.reduce_sum(tf.square(self.m[k+'b_'+sp][l]))
         penalty = tf.add(self.lambda_reg*pen_w,penalty)
         penalty = tf.add(self.lambda_reg*pen_b,penalty)
      return penalty

  def get_pentalty(self):
      (penalty_bop,penalty_bo,penalty_bo_rcut,
          penalty_be_cut,penalty_be,
          penalty_s,penalty_s_bo,
          penalty_rcut,rc_bo,
          penalty_vdw) = self.sess.run([self.penalty_bop,self.penalty_bo,self.penalty_bo_rcut,
                                         self.penalty_be_cut,self.penalty_be,
                                         self.penalty_s,self.penalty_s_bo,self.penalty_rcut,self.rc_bo,
                                         self.penalty_vdw],
                                        feed_dict=self.feed_dict)
      rcut = self.rcut
      print('\n------------------------------------------------------------------------')
      print('-                                                                      -')
      print('-                         Penalty Information                          -')
      print('-                                                                      -')
      print('------------------------------------------------------------------------\n')
      for bd in self.bonds:
          if bd in penalty_bop:
             print('bop cutoff penalty of                             {:5s}: {:6.4f}'.format(bd,penalty_bop[bd]))
          if bd in penalty_bo:
             print('BO state penalty of                               {:5s}: {:6.4f}'.format(bd,penalty_bo[bd]))
          if bd in penalty_bo_rcut:
             print('Differency between rcut and rcut-bo Penalty of    {:5s}: {:6.4f} {:6.4f} {:6.4f}'.format(bd,penalty_bo_rcut[bd],rc_bo[bd],rcut[bd]))
          # if bd in penalty_esi:
          #    print('Differency between bosi and esi Penalty of      {:5s}: {:6.4f}'.format(bd,penalty_esi[bd]))
          if bd in penalty_be_cut: 
             print('Bond-Energy at radius cutoff penalty of           {:5s}: {:6.4f}'.format(bd,penalty_be_cut[bd]))
          if bd in penalty_be: 
             print('Bond-Energy fluctuation penalty of                {:5s}: {:6.4f}'.format(bd,penalty_be[bd]))
          if bd in penalty_s_bo:
             print('Penalty of Bond-Order should greater than zero of {:5s}: {:6.4f}'.format(bd,penalty_s_bo[bd]))
          if bd in penalty_s:
             print('Anti-bond penalty of                              {:5s}: {:6.4f}'.format(bd,penalty_s[bd]))
          if bd in penalty_rcut:
             print('Bond-Order at radius cutoff penalty of            {:5s}: {:6.4f}'.format(bd,penalty_rcut[bd]))
          if bd in penalty_vdw:
             print('Vdw fluctuation penalty of                        {:5s}: {:6.4f}'.format(bd,penalty_vdw[bd]))
      print('\n')
      # self.log_energies()
