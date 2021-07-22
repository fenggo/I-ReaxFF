from __future__ import print_function
import matplotlib.pyplot as plt
from os import system, getcwd, chdir,listdir,environ,makedirs
from os.path import isfile,exists,isdir
from .reax_data import get_data 
from .link import links
from .reaxfflib import write_lib
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
  def __init__(self,libfile='ffield',direcs={},
               dft='ase',atoms=None,
               cons=['val','vale',
                     # 'ovun1','ovun2','ovun3','ovun4',
                     # 'ovun5','ovun6','ovun7','ovun8',
                     'lp2','lp3',#'lp1',
                     'cot1','cot2',
                     'coa1','coa2','coa3','coa4',
                     'pen1','pen2','pen3','pen4',
                     'Depi','Depp','cutoff','acut','hbtol',
                     #'val8','val9','val10',
                     ], # 
               nn=True,
               optmol=True,lambda_me=0.1,
               opt=None,optword='nocoul',
               mpopt=None,bdopt=None,mfopt=None,
               VariablesToOpt=None,
               nanv={'boc1':-2.0},
               batch_size=200,sample='uniform',
               hbshort=6.75,hblong=7.5,
               vdwcut=10.0,
               bore={'C-C':0.5,'others':0.45},
               bom={'others':1.2},
               pim={'others':10.0},
               spv_bm=False,
               spv_be=False,
               spv_pi=False,
               spv_ang=False,
               weight={'others':1.0},
               ro_scale=0.1,
               clip_op=True,
               InitCheck=True,
               resetDeadNeuron=False,
               messages=1,
               TwoBodyOnly=False,
               be_univeral_nn=None,bo_layer=[4,1],
               bo_univeral_nn=None,be_layer=[6,1],
               mf_univeral_nn=None,mf_layer=[9,2],
               vdw_univeral_nn=None,vdw_layer=None,#[6,1],
               vdwnn=False,
               EnergyFunction=1,
               MessageFunction=1,
               spec=[],
               sort=False,
               pkl=False,
               lambda_bd=100000.0,
               lambda_pi=1.0,
               lambda_reg=0.0001,
               lambda_ang=1.0,
               regularize=False,
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
      self.bom              = bom
      self.pim              = pim
      self.spv_be           = spv_be
      self.spv_bm           = spv_bm
      self.spv_pi           = spv_pi
      self.spv_ang          = spv_ang
      self.TwoBodyOnly      = TwoBodyOnly
      self.regularize       = regularize 
      self.lambda_reg       = lambda_reg
      self.lambda_pi        = lambda_pi
      self.lambda_ang       = lambda_ang
      self.mf_layer         = mf_layer
      self.be_layer         = be_layer
      self.vdw_layer        = vdw_layer if vdwnn else None
      self.bo_univeral_nn   = bo_univeral_nn
      self.be_univeral_nn   = be_univeral_nn
      self.mf_univeral_nn   = mf_univeral_nn
      self.vdw_univeral_nn  = vdw_univeral_nn
      if mpopt is None:
         self.mpopt = [True for i in range(messages+3)]
      else:
         self.mpopt = mpopt
      self.bdopt    = bdopt
      self.mfopt    = mfopt

      ReaxFF.__init__(self,libfile=libfile,direcs=direcs,
                      dft=dft,atoms=atoms,cons=cons,opt=opt,optword=optword,
                      VariablesToOpt=VariablesToOpt,optmol=optmol,lambda_me=lambda_me,
                      nanv=nanv,batch_size=batch_size,sample=sample,
                      hbshort=hbshort,hblong=hblong,vdwcut=vdwcut,
                      ro_scale=ro_scale,
                      clip_op=clip_op,InitCheck=InitCheck,resetDeadNeuron=resetDeadNeuron,
                      nn=nn,vdwnn=vdwnn,
                      bo_layer=bo_layer,spec=spec,sort=sort,pkl=pkl,weight=weight,
                      bore=bore,lambda_bd=lambda_bd,
                      optMethod=optMethod,maxstep=maxstep,
                      emse=emse,convergence=convergence,lossConvergence=lossConvergence,
                      losFunc=losFunc,conf_vale=conf_vale,
                      huber_d=huber_d,ncpu=ncpu)
      self.H        = []    # hiden states (or embeding states)
      self.D        = []    # degree matrix
      self.Hsi      = []
      self.Hpi      = []
      self.Hpp      = []
      self.esi      = {}
      self.fbo      = {}

  def supervise(self):
      ''' supervised learning term'''
      l_atol  = 0.0
      pen_w   = 0.0
      pen_b = 0.0
      wb_p    = ['fe','fsi','fpi','fpp']
      if self.vdwnn:
         wb_p.append('fv')
      w_n     = ['wi','wo',]
      b_n     = ['bi','b','bo']
      layer   = {'fe':self.be_layer[1],'fsi':self.bo_layer[1],'fpi':self.bo_layer[1],
                 'fpp':self.bo_layer[1]}
      if self.vdwnn:
         layer['fv'] = self.vdw_layer[1]

      wb_message = []
      for t in range(1,self.messages+1):
          wb_message.append('f'+str(t))
          layer['f'+str(t)] = self.mf_layer[1]

      self.diffn,self.diffa,self.diffb = {},{},{} 
      self.diffpi,self.diffang = {},{}
      self.diffe,self.diffesi,self.diffe_ = {},{},{}

      for bd in self.bonds: 
          [atomi,atomj] = bd.split('-') 
          if self.nbd[bd]>0:
             bd_ = bd  if bd in self.bore else 'others'
             if isinstance(self.bore[bd_],tuple):
                re_ = self.bore[bd_][0]
                bore_ = self.bore[bd_][1]
             else:
                bore_ = self.bore[bd_]
                re_   = self.re[bd]
 

             bd_ = bd  if bd in self.bom else 'others'
             if isinstance(self.bom[bd_],tuple):
                rm_  = self.bom[bd_][0]
                bom_ = self.bom[bd_][1]
             else:
                rm_  = self.bom[bd_]
                bom_ = bore_

             
             fao = tf.where(tf.greater(self.rbd[bd],self.rcuta[bd]),1.0,0.0)    ##### r> rcuta that bo = 0.0
             self.diffn[bd]  = tf.reduce_sum(self.bo0[bd]*fao)
             l_atol = tf.add(self.diffn[bd]*self.lambda_bd,l_atol)

             if self.spv_bm: ############ smooth tail #########################
                fao_= tf.where(tf.greater_equal(self.rbd[bd],rm_),1.0,0.0) ##### r > bm_ that bo < 10.0*botol
                self.diffa[bd] = tf.reduce_sum(input_tensor=tf.nn.relu((self.bo0[bd] - bom_)*fao_))
                l_atol = tf.add(self.diffa[bd]*self.lambda_bd,l_atol)

             # bop should be zero if r>rcut
             fbo = tf.where(tf.less_equal(self.rbd[bd],self.rc_bo[bd]),0.0,1.0) ##### r> rc_bo that bo < botol
             self.diffb[bd]  = tf.reduce_sum(self.bop[bd]*fbo) ##### Now the bop should be eterm1,eterm2,eterm3
             l_atol  = tf.add(self.diffb[bd]*self.lambda_bd,l_atol)
             
             if self.spv_be:
                fe  = tf.where(tf.less_equal(self.rbd[bd],re_),1.0,0.0) ##### r< r_e that bo > bore_
                self.diffe[bd]  = tf.reduce_sum(input_tensor=tf.nn.relu((bore_-self.bo0[bd])*fe))
                l_atol  = tf.add(self.diffe[bd]*self.lambda_bd,l_atol)
            
                self.diffe_[bd]  = tf.reduce_sum(input_tensor=tf.nn.relu((bore_-self.esi[bd])*fe))
                l_atol  = tf.add(self.diffe_[bd]*self.lambda_bd,l_atol)         ## penalty iterm for bond energy

             # if self.EnergyFunction == 3:
             #    fesi = tf.where(tf.less_equal(self.bo0[bd],self.botol),1.0,0.0) ##### bo <= 0.0 that e = 0.0
             #    self.diffesi[bd]  = tf.reduce_sum(self.esi[bd]*fesi)
             #    l_atol  = tf.add(self.diffesi[bd]*self.lambda_bd,l_atol)
             
             pen_rcut = tf.nn.relu(self.rc_bo[bd]-self.rcut[bd])
             l_atol= tf.add(pen_rcut*self.lambda_bd,l_atol)
 
             # penalize term for regularization of the neural networs
             if self.regularize:                                         # regularize
                for k in wb_p:
                    for k_ in w_n:
                        key     = k + k_ + '_' + bd
                        pen_w  += tf.reduce_sum(tf.square(self.m[key]))
                    for k_ in b_n:
                        key     = k + k_ + '_' + bd
                        pen_b  += tf.reduce_sum(tf.square(self.m[key]))
                    for l in range(layer[k]):                                               
                        pen_w += tf.reduce_sum(tf.square(self.m[k+'w_'+bd][l]))
                        pen_b += tf.reduce_sum(tf.square(self.m[k+'b_'+bd][l]))

      
         # for sp in self.spec: 
         #     pi_ = self.pim[sp] if sp in self.pim else self.pim['others']
         #     if self.nsp[sp]>0:
         #        self.diffpi[sp] = tf.reduce_sum(input_tensor=tf.nn.relu(self.Dpi[sp]-pi_))
         #        l_atol  = tf.add(self.diffpi[sp]*self.lambda_pi,l_atol)
      if self.optword.find('noang')<0:
         for ang in self.angs: 
             if self.nang[ang]>0:
                if self.spv_pi:
                   pi_ = self.pim[ang] if ang in self.pim else self.pim['others']
                   self.diffpi[ang] = tf.reduce_sum(input_tensor=tf.nn.relu(self.SBO[ang]-pi_))
                   l_atol  = tf.add(self.diffpi[ang]*self.lambda_pi,l_atol)
                if self.spv_ang:
                   self.diffang[ang] = tf.reduce_sum(self.thet2[ang]*self.fijk[ang])
                   l_atol  = tf.add(self.diffang[ang]*self.lambda_ang,l_atol)

         # self.diffpi = tf.reduce_sum(input_tensor=tf.nn.relu(self.DPI-2.0)) 
         # l_atol  = tf.add(self.diffpi*self.lambda_pi,l_atol)
 
      if self.regularize:                                               # regularize
         for sp in self.spec:
             for k in wb_message:
                 for k_ in w_n:
                     key     = k + k_ + '_' + sp
                     pen_w  += tf.reduce_sum(tf.square(self.m[key]))
                 for k_ in b_n:
                     key     = k + k_ + '_' + sp
                     pen_b  += tf.reduce_sum(tf.square(self.m[key]))
                 for l in range(layer[k]):                                               
                     pen_w += tf.reduce_sum(tf.square(self.m[k+'w_'+sp][l]))
                     pen_b += tf.reduce_sum(tf.square(self.m[k+'b_'+sp][l]))

         l_atol = tf.add(self.lambda_reg*pen_w,l_atol)
         l_atol = tf.add(self.lambda_reg*pen_b,l_atol)
      return l_atol

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
          # print(mol,w_)

          if self.losFunc   == 'n2':
             self.loss[mol] = tf.nn.l2_loss(self.E[mol]-self.dft_energy[mol],
                                 name='loss_%s' %mol)
          elif self.losFunc == 'abs':
             self.loss[mol] = tf.compat.v1.losses.absolute_difference(self.dft_energy[mol],self.E[mol])
          elif self.losFunc == 'mse':
             self.loss[mol] = tf.compat.v1.losses.mean_squared_error(self.dft_energy[mol],self.E[mol])
          elif self.losFunc == 'huber':
             self.loss[mol] = tf.compat.v1.losses.huber_loss(self.dft_energy[mol],self.E[mol],delta=self.huber_d)

          sum_edft = tf.reduce_sum(input_tensor=tf.abs(self.dft_energy[mol]-self.max_e[mol]))
          self.accur[mol] = 1.0 - tf.reduce_sum(input_tensor=tf.abs(self.E[mol]-self.dft_energy[mol]))/(sum_edft+0.00000001)
         
          self.Loss     += self.loss[mol]*w_
          self.accuracy += self.accur[mol]

      self.ME   = 0.0
      for mol in self.mols:
          mols = mol.split('-')[0] 
          self.ME += tf.square(self.MolEnergy[mols])

      self.loss_atol = self.supervise()
      self.Loss     += self.loss_atol

      if self.optmol:
         self.Loss  += self.ME*self.lambda_me
      self.accuracy  = self.accuracy/self.nmol

  def build_graph(self):
      print('-  building graph ...')
      self.accuracy   = tf.constant(0.0,name='accuracy')
      self.accuracies = {}
      self.get_bond_energy()

      if self.TwoBodyOnly:
         self.get_atomic_energy()
      else:
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

  # def get_elone(self,atom,D):
  #     NLPOPT              = 0.5*(self.p['vale_'+atom] - self.p['val_'+atom])
  #     self.Delta_e[atom]  = 0.5*(self.p['vale_'+atom] - D)

  #     # self.DE[atom]     = -tf.nn.relu(-tf.math.ceil(self.Delta_e[atom])) 
  #     # self.nlp[atom]    = -self.DE[atom] + tf.exp(-self.p['lp1']*4.0*tf.square(1.0+self.Delta_e[atom]-self.DE[atom]))
  #     self.nlp[atom]      = self.Delta_e[atom]
  #     self.Delta_lp[atom] = NLPOPT- self.Delta_e[atom]                     # nan error
      
  #     self.explp[atom]    = 1.0+tf.exp(-self.p['lp3']*self.Delta_lp[atom])

  #     self.EL[atom] = tf.math.divide(self.p['lp2_'+atom]*self.Delta_lp[atom],self.explp[atom],
  #                                    name='Elone_%s' %atom)

  # def get_theta0(self,ang):
  #     self.sbo[ang] = tf.gather_nd(self.DPI,self.dglist[ang])
  #     #self.pbo[ang] = tf.gather_nd(self.PBO,self.dglist[ang])
  #     #self.rnlp[ang]= tf.gather_nd(self.NLP,self.dglist[ang])
  #     self.SBO[ang] = self.sbo[ang] #- tf.multiply(1.0-self.pbo[ang],self.D_ang[ang]+self.p['val8']*self.rnlp[ang])    
      
  #     ok         = tf.logical_and(tf.less_equal(self.SBO[ang],1.0),tf.greater(self.SBO[ang],0.0))
  #     S1         = tf.where(ok,self.SBO[ang],tf.zeros_like(self.SBO[ang]))    #  0< sbo < 1                  
  #     self.SBO01[ang] = tf.where(ok,tf.pow(S1,self.p['val9']),tf.zeros_like(S1)) 

  #     ok    = tf.logical_and(tf.less(self.SBO[ang],2.0),tf.greater(self.SBO[ang],1.0))
  #     S2    = tf.where(ok,self.SBO[ang],tf.zeros_like(self.SBO[ang]))                     
  #     F2    = tf.where(ok,tf.ones_like(S2),tf.zeros_like(S2))                             #  1< sbo <2
     
  #     S2    = 2.0*F2-S2  
  #     self.SBO12[ang] = tf.where(ok,2.0-tf.pow(S2,self.p['val9']),tf.zeros_like(self.SBO[ang]))     #  1< sbo <2
  #                                                                                         #     sbo >2
  #     SBO2  = tf.where(tf.greater_equal(self.SBO[ang],2.0),
  #                      tf.ones_like(self.SBO[ang]),tf.zeros_like(self.SBO[ang]))

  #     self.SBO3[ang]   = self.SBO01[ang]+self.SBO12[ang]+2.0*SBO2
  #     theta0           = self.p['theta0_'+ang]*(1.0-tf.exp(-self.p['val10']*(2.0-self.SBO3[ang])))
  #     self.theta0[ang] = theta0/57.29577951


  def get_total_energy(self):
      for mol in self.mols:
          # mols = mol.split('-')[0] 
          if self.TwoBodyOnly:
             self.E[mol] = tf.add(self.ebond[mol] + 
                                  self.evdw[mol]  +
                                  self.ecoul[mol] +
                                  self.ehb[mol]   +
                                  self.eself[mol], 
                                  self.zpe[mol],name='E_%s' %mol)   
          else:
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

  def get_tap(self,r,bd):
      if self.vdwnn:
         # r_ = r/self.p['rvdw_'+bd]
         tp = self.f_nn('fv',bd,self.nvb[bd],[r],layer=self.vdw_layer[1])
      else:
         tp = 1.0+tf.math.divide(-35.0,tf.pow(self.vdwcut,4.0))*tf.pow(r,4.0)+ \
              tf.math.divide(84.0,tf.pow(self.vdwcut,5.0))*tf.pow(r,5.0)+ \
              tf.math.divide(-70.0,tf.pow(self.vdwcut,6.0))*tf.pow(r,6.0)+ \
              tf.math.divide(20.0,tf.pow(self.vdwcut,7.0))*tf.pow(r,7.0)
      return tp

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
      self.frc[bd] = tf.where(tf.logical_or(tf.greater(self.rbd[bd],self.rc_bo[bd]),
                                            tf.less_equal(self.rbd[bd],0.001)),
                              tf.zeros_like(self.rbd[bd]),tf.ones_like(self.rbd[bd]))

      self.bodiv1[bd] = tf.math.divide(self.rbd[bd],self.p['rosi_'+bd],name='bodiv1_'+bd)
      self.bopow1[bd] = tf.pow(self.bodiv1[bd],self.p['bo2_'+bd])
      self.eterm1[bd] = tf.exp(tf.multiply(self.p['bo1_'+bd],self.bopow1[bd]))*self.frc[bd] 

      self.bodiv2[bd] = tf.math.divide(self.rbd[bd],self.p['ropi_'+bd],name='bodiv2_'+bd)
      self.bopow2[bd] = tf.pow(self.bodiv2[bd],self.p['bo4_'+bd])
      self.eterm2[bd] = tf.exp(tf.multiply(self.p['bo3_'+bd],self.bopow2[bd]))*self.frc[bd]

      self.bodiv3[bd] = tf.math.divide(self.rbd[bd],self.p['ropp_'+bd],name='bodiv3_'+bd)
      self.bopow3[bd] = tf.pow(self.bodiv3[bd],self.p['bo6_'+bd])
      self.eterm3[bd] = tf.exp(tf.multiply(self.p['bo5_'+bd],self.bopow3[bd]))*self.frc[bd]

      fsi_            = self.f_nn('fsi',bd, self.nbd[bd],[self.eterm1[bd]],layer=self.bo_layer[1])  
      fpi_            = self.f_nn('fpi',bd, self.nbd[bd],[self.eterm2[bd]],layer=self.bo_layer[1])  
      fpp_            = self.f_nn('fpp',bd, self.nbd[bd],[self.eterm3[bd]],layer=self.bo_layer[1]) 

      self.bop_si[bd] = fsi_ #*self.frc[bd] #*self.eterm1[bd]  
      self.bop_pi[bd] = fpi_ #*self.frc[bd] #*self.eterm2[bd]
      self.bop_pp[bd] = fpp_ #*self.frc[bd] #*self.eterm3[bd]
      self.bop[bd]    = tf.add(self.bop_si[bd],self.bop_pi[bd]+self.bop_pp[bd],name='BOp_'+bd)

  def get_bondorder(self,t,bd,atomi,atomj):
      Di      = tf.gather_nd(self.D[t-1],self.dilink[bd])
      Dj      = tf.gather_nd(self.D[t-1],self.djlink[bd])
      h       = self.H[t-1][bd]
      Dbi     = Di-h
      Dbj     = Dj-h

      b       = bd.split('-')
      bdr     = b[1]+'-'+b[0]
      flabel  = 'f'+str(t)
      if self.MessageFunction==1:
         Fi    = self.f_nn(flabel,b[0],self.nbd[bd],[Dbi,Dbj,h],layer=self.mf_layer[1])
         Fj    = self.f_nn(flabel,b[1],self.nbd[bd],[Dbj,Dbi,h],layer=self.mf_layer[1])
         F     = Fi*Fj
       
         bosi  = self.Hsi[t-1][bd]*F
         bopi  = self.Hpi[t-1][bd]*F
         bopp  = self.Hpp[t-1][bd]*F
      elif self.MessageFunction==2:
         Fi    = self.fmessage(flabel,b[0],self.nbd[bd],[Dbi,Dbj,h],layer=self.mf_layer[1])
         Fj    = self.fmessage(flabel,b[1],self.nbd[bd],[Dbj,Dbi,h],layer=self.mf_layer[1])
         F     = Fi*Fj
         Fsi,Fpi,Fpp = tf.unstack(F,axis=2)
       
         bosi  = self.Hsi[t-1][bd]*Fsi
         bopi  = self.Hpi[t-1][bd]*Fpi
         bopp  = self.Hpp[t-1][bd]*Fpp
      else:
         raise NotImplementedError('-  Message function not supported yet!')
      bo       = bosi+bopi+bopp
      return bo,bosi,bopi,bopp

  def get_bond_energy(self):
      BO = tf.zeros([1,self.batch])   # for ghost atom, the value is zero
      for bd in self.bonds:
          if self.nbd[bd]>0:
             self.get_bondorder_uc(bd)
             BO = tf.concat([BO,self.bop[bd]],0)
  
      D           = tf.gather_nd(BO,self.dlist)  
      self.Deltap = tf.reduce_sum(input_tensor=D,axis=1,name='Deltap')

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
      self.H.append(self.bop)                   # 
      self.Hsi.append(self.bop_si)              #
      self.Hpi.append(self.bop_pi)              #
      self.Hpp.append(self.bop_pp)              # 
      self.D.append(self.Deltap)                # get the initial hidden state H[0]

      for t in range(1,self.messages+1):
          print('-  message passing for t=%d ...' %t)
          self.H.append({})                     # get the hidden state H[t]
          self.Hsi.append({})                   #
          self.Hpi.append({})                   #
          self.Hpp.append({})                   #             

          BO = tf.zeros([1,self.batch])         # for ghost atom, the value is zero
          for bd in self.bonds:
              if self.nbd[bd]>0:
                 [atomi,atomj] = bd.split('-') 
                 bo,bosi,bopi,bopp = self.get_bondorder(t,bd,atomi,atomj)

                 self.H[t][bd]   = bo
                 self.Hsi[t][bd] = bosi
                 self.Hpi[t][bd] = bopi
                 self.Hpp[t][bd] = bopp

                 BO = tf.concat([BO,bo],0)
      
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
      SO_ = tf.gather_nd(self.BSO,self.dlist,name='SO_') 
      self.BPI = tf.gather_nd(BPI,self.dlist,name='BPI') 

      self.Delta  = tf.reduce_sum(input_tensor=D_,axis=1,name='Delta')  # without valence i.e. - Val 
      self.SO     = tf.reduce_sum(input_tensor=SO_,axis=1,name='sumover')  
      self.FBOT   = taper(self.BO0,rmin=self.atol,rmax=2.0*self.atol) 
      self.FHB    = taper(self.BO0,rmin=self.hbtol,rmax=2.0*self.hbtol) 

  def get_ebond(self,bd):
      Di      = tf.gather_nd(self.Delta,self.dilink[bd])
      Dj      = tf.gather_nd(self.Delta,self.djlink[bd])

      Dbi     = Di-self.bo0[bd]
      Dbj     = Dj-self.bo0[bd]

      b       = bd.split('-')
      bdr     = b[1]+'-'+b[0]

      if self.EnergyFunction==1:
         self.esi[bd] = self.f_nn('fe',bd, self.nbd[bd],[self.bosi[bd],self.bopi[bd],self.bopp[bd]],
                                  layer=self.be_layer[1])
         self.EBD[bd] = -self.p['Desi_'+bd]*self.esi[bd]*self.bo0[bd]
      elif self.EnergyFunction==2:
         Fi = self.f_nn('fe',bd,self.nbd[bd],[Dbi,Dbj,self.bo0[bd]],layer=self.be_layer[1])
         Fj = self.f_nn('fe',bd,self.nbd[bd],[Dbj,Dbi,self.bo0[bd]],layer=self.be_layer[1])
         self.esi[bd] = Fi*Fj*self.bo0[bd]
         self.EBD[bd] = -self.p['Desi_'+bd]*self.esi[bd]
      elif self.EnergyFunction==3:
         self.esi[bd] = self.f_nn('fe',bd, self.nbd[bd],[self.bosi[bd],self.bopi[bd],self.bopp[bd]],
                                  layer=self.be_layer[1])
         self.sieng[bd] = self.p['Desi_'+bd]*self.esi[bd]*self.bosi[bd]
         self.pieng[bd] = self.p['Depi_'+bd]*self.esi[bd]*self.bopi[bd]
         self.ppeng[bd] = self.p['Depp_'+bd]*self.esi[bd]*self.bopp[bd]
         self.EBD[bd]   = - self.sieng[bd] - self.pieng[bd] - self.ppeng[bd] 
          
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

      reuse_m = True if self.bo_layer==self.bo_layer_ else False
      for p_ in ['fsi','fpi','fpp']:
          if not self.bo_univeral_nn is None:
             self.set_univeral_wb(pref=p_,bd=self.bo_univeral_nn[0],reuse_m=reuse_m,
                                    nin=1,nout=1,layer=self.bo_layer,
                                    nnopt=self.mpopt[0],bias=2.0)
          self.set_wb(pref=p_,reuse_m=reuse_m,nin=1,nout=1,layer=self.bo_layer,
                      vlist=self.bonds,nnopt=self.mpopt[0],bias=2.0)
 
      ############ set weight and bias for message neural network ###################
      reuse_m = True if self.mf_layer==self.mf_layer_ else False
      for t in range(1,self.messages+1):
          b = 0.881373587 if t>1 else -0.867
          nout_ = 1 if self.MessageFunction==1 or self.MessageFunction==3 else 3
          if not self.mf_univeral_nn is None:
             self.set_univeral_wb(pref='f'+str(t),bd=self.mf_univeral_nn[0],reuse_m=reuse_m,
                                    nin=3,nout=nout_,layer=self.mf_layer,
                                    nnopt=self.mpopt[t],bias=b)
          self.set_message_wb(pref='f'+str(t),reuse_m=reuse_m,nin=3,nout=nout_,
                              layer=self.mf_layer,nnopt=self.mpopt[t],bias=b) 

      ############ set weight and bias for energy neural network ###################
      if self.EnergyFunction==self.EnergyFunction_ and self.be_layer==self.be_layer_:
         reuse_m = True  
      else:
         reuse_m = False 
      if not self.be_univeral_nn is None:
         self.set_univeral_wb(pref='fe',bd=self.be_univeral_nn[0],reuse_m=reuse_m,
                                nin=3,nout=1,layer=self.be_layer,
                                nnopt=self.mpopt[t+1],bias=2.0)
      self.set_wb(pref='fe',reuse_m=reuse_m,nin=3,nout=1,layer=self.be_layer,
                  vlist=self.bonds,nnopt=self.mpopt[t+1],bias=2.0)

      if self.vdwnn:
         reuse_m = True if self.vdw_layer==self.vdw_layer_ else False
         if not self.vdw_univeral_nn is None:
            self.set_univeral_wb(pref='fv',bd=self.vdw_univeral_nn[0],reuse_m=reuse_m,
                                   nin=1,nout=1,layer=self.vdw_layer,
                                   nnopt=self.mpopt[t+1],bias=-0.867)
         self.set_wb(pref='fv',reuse_m=reuse_m,nin=1,nout=1,layer=self.vdw_layer,
                     vlist=self.bonds,nnopt=self.mpopt[-1],bias=-0.867)

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
             self.m[pref+'wi_'+bd] = tf.Variable(tf.random.normal([nin,layer[0]],stddev=0.1),name=pref+'wi_'+bd)   
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
                                                         stddev=0.1),name=pref+'wh'+str(i)+'_'+bd)) 
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
              self.m[pref+'wo_'+bd] = tf.Variable(tf.random.normal([layer[0],nout],stddev=0.0001), name=pref+'wo_'+bd)   
              self.m[pref+'bo_'+bd] = tf.Variable(tf.random.normal([nout], stddev=0.0001)+bias,
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
             self.m[pref+'wi_'+sp] = tf.Variable(tf.random.normal([nin,layer[0]],stddev=0.1),name=pref+'wi_'+sp)   
             self.m[pref+'bi_'+sp] = tf.Variable(tf.random.normal([layer[0]],stddev=0.1),name=pref+'bi_'+sp)  
             self.m[pref+'wo_'+sp] = tf.Variable(tf.random.normal([layer[0],nout],stddev=0.1),name=pref+'wo_'+sp)   
             self.m[pref+'bo_'+sp] = tf.Variable(tf.random.normal([nout],stddev=0.1),name=pref+'bo_'+sp)  
             for i in range(layer[1]):   
                 self.m[pref+'w_'+sp].append(tf.Variable(tf.random.normal([layer[0],layer[0]], 
                                                         stddev=0.1),name=pref+'wh'+str(i)+'_'+sp)) 
                 self.m[pref+'b_'+sp].append(tf.Variable(tf.random.normal([layer[0]], 
                                                         stddev=0.1),name=pref+'bh'+str(i)+'_'+sp)) 


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
      
      if reuse_m:                   # input layer
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
         self.m[pref+'wi'] = tf.Variable(tf.random.normal([nin,layer[0]],stddev=0.1),name=pref+'wi')   
         self.m[pref+'bi'] = tf.Variable(tf.random.normal([layer[0]],stddev=0.1),name=pref+'bi')  
         self.m[pref+'wo'] = tf.Variable(tf.random.normal([layer[0],nout],stddev=0.0001),name=pref+'wo')   
         self.m[pref+'bo'] = tf.Variable(tf.random.normal([nout], stddev=0.0001)+bias,name=pref+'bo')
         for i in range(layer[1]):   
             self.m[pref+'w'].append(tf.Variable(tf.random.normal([layer[0],layer[0]], 
                                                 stddev=0.1),name=pref+'wh'+str(i))) 
             self.m[pref+'b'].append(tf.Variable(tf.random.normal([layer[0]], 
                                                 stddev=0.1),name=pref+'bh'+str(i))) 

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
         for sp in self.spec:
             for t in range(1,self.messages+1):
                 self.univeral_nn.append('f'+str(t)+'_'+sp)

  def write_lib(self,libfile='ffield',loss=None):
      p_   = self.sess.run(self.p)
      self.p_ = {}
      
      self.MolEnergy_ = self.sess.run(self.MolEnergy)
      for key in self.MolEnergy_:
          self.MolEnergy_[key] = float(self.MolEnergy_[key])

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
              'EnergyFunction':self.EnergyFunction,
              'MessageFunction':self.MessageFunction, 
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
      else:
         write_lib(self.p_,self.spec,self.bonds,self.offd,
                   self.angs,self.torp,self.hbs,
                   zpe=self.zpe_,libfile=libfile,
                   loss=loss)

