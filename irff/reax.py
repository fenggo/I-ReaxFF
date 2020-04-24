from __future__ import print_function
import matplotlib.pyplot as plt
from os import system, getcwd, chdir,listdir,environ,makedirs
from os.path import isfile,exists,isdir
from .gulp import write_gulp_in,get_reax_energy
from .reax_data import get_data 
from .link import links
from .reaxfflib import read_lib,write_lib
from .initCheck import Init_Check
# from .dingtalk import send_msg
from .RadiusCutOff import setRcut
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


class Chromosome:
  def __init__(self,genes,loss):
      self.Genes  = genes
      self.Loss   = loss
  

class logger(object):
  """docstring for lo"""
  def __init__(self,flog='training.log'):
      self.flog = flog

  def info(self,msg): 
      print(msg)
      flg = open(self.flog,'a')
      print(msg,file=flg)
      flg.close()


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

    for imag in image_list:
        frames.append(imageio.imread(imag))
    imageio.mimsave('result_%s.gif' %mol, frames, 'GIF', duration = 0.3)


def rtaper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = tf.where(tf.less(r,rmin),tf.ones_like(r),tf.zeros_like(r)) # r > rmax then 1 else 0

    ok    = tf.logical_and(tf.less_equal(r,rmax),tf.greater(r,rmin))      # rmin < r < rmax  = r else 0
    r2    = tf.where(ok,r,tf.zeros_like(r))
    r20   = tf.where(ok,tf.ones_like(r),tf.zeros_like(r))

    rterm = tf.math.divide(1.0,tf.pow(rmax-rmin,3))
    rm    = rmax*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmin*r20
    r22   = rterm*rd*rd*trm1
    return tf.add(r22,r3)


def taper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = tf.where(tf.greater(r,rmax),tf.ones_like(r),tf.zeros_like(r)) # r > rmax then 1 else 0

    ok    = tf.logical_and(tf.less_equal(r,rmax),tf.greater(r,rmin))      # rmin < r < rmax  = r else 0
    r2    = tf.where(ok,r,tf.zeros_like(r))
    r20   = tf.where(ok,tf.ones_like(r),tf.zeros_like(r))

    rterm = tf.math.divide(1.0,tf.pow(rmin-rmax,3))
    rm    = rmin*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmax*r20
    r22   = rterm*rd*rd*trm1
    return tf.add(r22,r3)


def DIV(y,x):
    xok = tf.not_equal(x, 0.0)
    f = lambda x: y/x
    safe_f = tf.zeros_like
    safe_x = tf.where(xok,x,tf.ones_like(x))
    return tf.where(xok, f(safe_x), safe_f(x))


def DIV_IF(y,x):
    xok = tf.not_equal(x, 0.0)
    f = lambda x: y/x
    safe_f = tf.zeros_like 
    safe_x = tf.where(xok,x,tf.zeros_like(x)+0.00000001)
    return tf.where(xok, f(safe_x), f(safe_x))


class ReaxFF(object):
  def __init__(self,libfile='ffield',direcs={},
               dft='ase',atoms=None,
               cons=['val','vale'],
               opt=None,optword='nocoul',
               VariablesToOpt=None,
               nanv={'boc1':-2.0},
               batch_size=200,sample='uniform',
               hbshort=6.75,hblong=7.5,
               vdwcut=10.0,
               rcut=None,rcuta=None,re=None,
               bore=0.5,
               interactive=False,
               ro_scale=0.1,
               clip_op=True,
               InitCheck=True,
               atomic=True,
               nn=False,
               nnopt=True,
               bo_layer=[8,4],
               spec=['C','H','O','N'],
               sort=False,
               pkl=True,
               popSize=500,
               fromPop=False,
               board=False,
               bo_penalty=100000.0,
               to_train=True,
               optMethod='ADAM',
               maxstep=60000,
               emse=0.9,
               convergence=0.97,
               losFunc='n2',
               conf_vale=None,
               huber_d=30.0,
               ncpu=None):
      '''
         version 3.0 
           Time: 2018-10-20
           Intelligence ReaxFF Neual Network: Evoluting the Force Field parameters on-the-fly
           2017-11-01
      '''
      self.direcs        = direcs
      self.libfile       = libfile
      self.batch         = batch_size
      self.sample        = sample
      self.lx            = np.linspace(0,self.batch,self.batch)
      self.opt           = opt
      self.VariablesToOpt= VariablesToOpt
      self.cons          = cons
      self.optword       = optword
      self.vdwcut        = vdwcut
      self.dft           = dft
      self.atoms         = atoms
      self.ro_scale      = ro_scale
      self.conf_vale     = conf_vale
      self.clip_op       = clip_op
      self.InitCheck     = InitCheck
      self.rcut          = rcut
      self.rcuta         = rcuta
      self.hbshort       = hbshort
      self.hblong        = hblong
      self.atomic        = atomic
      self.nn            = nn
      self.nnopt         = nnopt
      self.bo_layer      = bo_layer
      self.spec          = spec
      self.sort          = sort
      self.time          = time.time()
      self.interactive   = interactive
      self.pkl           = pkl
      self.popSize       = popSize
      self.fromPop       = fromPop
      self.board         = board 
      self.to_train      = to_train
      self.maxstep       = maxstep
      self.emse          = emse
      self.optMethod     = optMethod
      self.convergence   = convergence
      self.losFunc       = losFunc
      self.huber_d       = huber_d
      self.ncpu          = ncpu
      self.bore          = bore
      self.m_,self.m     = {},{}

      if self.libfile.endswith('.json'):
         lf = open(self.libfile,'r')
         j = js.load(lf)
         self.p_  = j['p']
         self.m_  = j['m']
         self.zpe_= j['zpe']
         # self.massages = j['massages']
         if 'bo_layer' in j:
            self.bo_layer_ = j['bo_layer']
         else:
            self.bo_layer_ = None
         lf.close()
         self.init_bonds()
      else:
         self.p_,self.zpe_,self.spec,self.bonds,self.offd,self.angs,self.torp,self.hbs= \
              read_lib(libfile=self.libfile,zpe=True)
              
      self.set_rcut(rcut,rcuta,re)
      if self.InitCheck:
         self.ic = Init_Check(re=self.re,nanv=nanv)
         self.p_ = self.ic.check(self.p_)

      self.set_neurons()
      self.bo_penalty    = bo_penalty
      self.logger        = logger('training.log')
      self.initialized   = False  

      if self.VariablesToOpt is None:
         self.set_parameters(self.opt) 
      else:
         self.set_parameters_ud() 


  def init_bonds(self):
      self.bonds,self.offd,self.angs,self.torp,self.hbs = [],[],[],[],[]
      for key in self.p_:
          k = key.split('_')
          if k[0]=='bo1':
             self.bonds.append(k[1])
          elif k[0]=='rosi':
             kk = k[1].split('-')
             if len(kk)==2:
                self.offd.append(k[1])
          elif k[0]=='theta0':
             self.angs.append(k[1])
          elif k[0]=='tor1':
             self.torp.append(k[1])
          elif k[0]=='rohb':
             self.hbs.append(k[1])
          elif k[0]=='val':
             if k[1] not in self.spec:
                self.spec.append(k[1])


  def set_rcut(self,rcut,rcuta,re):
      rcut_,rcuta_,re_ = setRcut(self.bonds)
      if rcut is None:  ## bond order compute cutoff
         self.rcut = rcut_
      if rcuta is None: ## angle term cutoff
         self.rcuta = rcuta_
      if re is None: ## angle term cutoff
         self.re = re_


  def initialize(self): 
      self.nframe = 0
      molecules   = {}
      self.max_e  = {}
      self.cell   = {}
      self.nbe0   = {}
      self.mols   = []
      self.eself,self.evdw_,self.ecoul_ = {},{},{}

      for mol in self.direcs: 
          nindex = []
          for key in molecules:
              if self.direcs[key]==self.direcs[mol]:
                 nindex.extend(molecules[key].indexs) 
          data_ =  get_data(structure=mol,
                                direc=self.direcs[mol],
                                  dft=self.dft,
                                atoms=self.atoms,
                               vdwcut=self.vdwcut,
                                 rcut=self.rcut,
                                rcuta=self.rcuta,
                              hbshort=self.hbshort,
                               hblong=self.hblong,
                                batch=self.batch,
                               sample=self.sample,
                                 sort=self.sort,
                       p=self.p_,spec=self.spec,bonds=self.bonds,
                                  pkl=self.pkl,
                               nindex=nindex)

          if data_.status:
             self.mols.append(mol)
             molecules[mol]  = data_
             self.nframe    += self.batch
             print('-  max energy of %s: %f.' %(mol,molecules[mol].max_e))
             self.max_e[mol] = molecules[mol].max_e

             # self.evdw_[mol]= molecules[mol].evdw
             self.ecoul_[mol] = molecules[mol].ecoul  
                  
             self.eself[mol] = molecules[mol].eself    
             self.nbe0[mol]  = molecules[mol].nbe0   
             self.cell[mol]  = molecules[mol].cell
          else:
             print('-  data status of %s:' %mol,data_.status)
      self.nmol = len(molecules)
      
      print('-  generating links ...')
      self.get_links(molecules)
      
      with tf.compat.v1.name_scope('input'):
           self.memory()
      if not self.atomic:     
         self.set_zpe(molecules=molecules)
         
      self.build_graph()     
      self.feed_dict = self.feed_data()
      self.initialized = True
      return molecules


  def get_links(self,molecules):
      self.lk = links(species=self.spec,
                        bonds=self.bonds,
                         angs=self.angs,
                         tors=self.tors,
                          hbs=self.hbs,
                       vdwcut=self.vdwcut,
                    molecules=molecules)
      self.dlist    = self.lk.dlist
      self.dilink   = self.lk.dilink
      self.djlink   = self.lk.djlink
      self.dalink   = self.lk.dalink
      self.dalist   = self.lk.dalist

      self.bdlink   = self.lk.bdlink
      self.nbd      = self.lk.nbd
      self.nsp      = self.lk.nsp
      self.atomlist = self.lk.atomlist
      self.atomlink = self.lk.atomlink

      self.dglist   = self.lk.dglist
      self.dgilist  = self.lk.dgilist
      self.dgklist  = self.lk.dgklist
      self.anglink  = self.lk.anglink
      self.boaij    = self.lk.boaij
      self.boajk    = self.lk.boajk
      self.nang     = self.lk.nang

      self.tij      = self.lk.tij
      self.tjk      = self.lk.tjk
      self.tkl      = self.lk.tkl
      self.dtj      = self.lk.dtj
      self.dtk      = self.lk.dtk
      self.torlink  = self.lk.torlink
      self.ntor     = self.lk.ntor

      self.nvb      = self.lk.nvb
      self.vlink    = self.lk.vlink

      self.nhb      = self.lk.nhb
      self.hblink   = self.lk.hblink
      self.hij      = self.lk.hij


  def memory(self):
      self.frc = {}
      self.bodiv1,self.bodiv2,self.bodiv3 = {},{},{}
      self.bopow1,self.bopow2,self.bopow3 = {},{},{}
      self.eterm1,self.eterm2,self.eterm3 = {},{},{}
      self.bop_si,self.bop_pi,self.bop_pp,self.bop = {},{},{},{}
      self.bosi,self.bosi_pen = {},{}
      self.bopi,self.bopp,self.bo0,self.bo,self.bso = {},{},{},{},{}

      self.f_1 = {}
      self.dexpf2,self.dexpf2t,self.f_2={},{},{}
      self.dexpf3,self.dexpf3t,self.f_3,self.f3log={},{},{},{}
      self.delta,self.Di,self.Dj,self.Di_boc,self.Dj_boc={},{},{},{},{}

      self.f4r,self.f5r,self.f_4,self.f_5,self.df4,self.df5={},{},{},{},{},{}
      self.F,self.F_11,self.F_12,self.F_45={},{},{},{}
      self.fbot = {}
      self.powb,self.expb,self.EBD,self.ebond,self.ebda = {},{},{},{},{}
      self.sieng,self.pieng,self.ppeng = {},{},{}

      self.Delta_e,self.DE,self.nlp,self.Delta_lp,self.Dlp = {},{},{},{},{}
      self.Dpi,self.BSO,self.BOPI,self.Delta_lpcorr = {},{},{},{}
      self.explp,self.EL,self.elone,self.Elone,self.ELONE = {},{},{},{},{}
      self.EOV,self.Eover,self.eover,self.otrm1,self.otrm2 =  {},{},{},{},{}

      self.expeu1,self.expeu2,self.eu1,self.expeu3,self.eu2,self.EUN = {},{},{},{},{},{}
      self.eunder,self.Eunder = {},{}

      self.EANG,self.Eang,self.eang,self.theta0,self.fijk = {},{},{},{},{}
      self.pbo,self.sbo,self.SBO,self.SBO12,self.SBO3,self.SBO01 = {},{},{},{},{},{}
      self.dang,self.D_ang = {},{}

      self.thet,self.expang,self.f_7,self.f_8,self.rnlp = {},{},{},{},{}
      self.EPEN,self.BOij,self.BOjk,self.Epen,self.epen = {},{},{},{},{}

      self.expcoa1,self.texp0,self.texp1,self.texp2,self.texp3 = {},{},{},{},{}
      self.texp4,self.ETC,self.tconj,self.Etc = {},{},{},{}

      self.cos3w,self.etor,self.Etor,self.ETOR = {},{},{},{}
      self.BOpjk,self.BOtij,self.BOtjk,self.BOtkl,self.fijkl,self.so = {},{},{},{},{},{}
      self.f_9,self.f_10,self.f_11,self.f_12,self.expv2 = {},{},{},{},{}
      self.f11exp3,self.f11exp4 = {},{}
      
      self.v1,self.v2,self.v3 = {},{},{}
      self.Efcon,self.EFC,self.efcon = {},{},{}
      self.expvdw1,self.expvdw2,self.EVDW,self.Evdw,self.f_13={},{},{},{},{}

      self.Ecou,self.ECOU,self.evdw,self.ecoul,self.tpv,self.rth = {},{},{},{},{},{}
 
      self.exphb1,self.exphb2,self.sin4,self.EHB = {},{},{},{}
      self.pc,self.fhb,self.BOhb,self.ehb,self.Ehb = {},{},{},{},{}

      self.dft_energy,self.E,self.zpe,self.eatom,self.loss,self.accur = {},{},{},{},{},{}
      
      for mol in self.mols:
          mol_ = mol.split('-')[0]
          self.dft_energy[mol] = tf.compat.v1.placeholder(tf.float32,shape=[self.batch],
                                                name='DFT_energy_%s' %mol)

      self.rbd,self.rv,self.qij = {},{},{}
      for bd in self.bonds:
          if self.nbd[bd]>0:
             self.rbd[bd] = tf.compat.v1.placeholder(tf.float32,shape=[self.nbd[bd],self.batch],
                                           name='rbd_%s' %bd)
          if self.nvb[bd]>0:
             self.rv[bd]  = tf.compat.v1.placeholder(tf.float32,shape=[self.nvb[bd],self.batch],
                                           name='rvdw_%s' %bd)
             if self.optword.find('nocoul')<0:
                self.qij[bd] = tf.compat.v1.placeholder(tf.float32,shape=[self.nvb[bd],self.batch],
                                              name='qij_%s' %bd)
      self.theta = {}
      for ang in self.angs:
          if self.nang[ang]>0:
             self.theta[ang] = tf.compat.v1.placeholder(tf.float32,shape=[self.nang[ang],self.batch],
                                              name='theta_%s' %ang)

      self.s_ijk,self.s_jkl,self.cos_w,self.cos2w,self.w={},{},{},{},{}
      for tor in self.tors:
          if self.ntor[tor]>0:
             self.s_ijk[tor] = tf.compat.v1.placeholder(tf.float32,shape=[self.ntor[tor],self.batch],
                                              name='sijk_%s' %tor)
             self.s_jkl[tor] = tf.compat.v1.placeholder(tf.float32,shape=[self.ntor[tor],self.batch],
                                              name='sjkl_%s' %tor)
             self.w[tor]     = tf.compat.v1.placeholder(tf.float32,shape=[self.ntor[tor],self.batch],
                                              name='w_%s' %tor)
             self.cos_w[tor] = tf.cos(self.w[tor])
             self.cos2w[tor] = tf.cos(2.0*self.w[tor])

      self.rhb,self.frhb,self.hbthe = {},{},{}
      for hb in self.hbs:
          if self.nhb[hb]>0:
             self.rhb[hb]  = tf.compat.v1.placeholder(tf.float32,shape=[self.nhb[hb],self.batch],
                                            name='rhb_%s' %hb)
             self.frhb[hb] = tf.compat.v1.placeholder(tf.float32,shape=[self.nhb[hb],self.batch],
                                            name='frhb_%s' %hb)
             self.hbthe[hb]= tf.compat.v1.placeholder(tf.float32,shape=[self.nhb[hb],self.batch],
                                            name='hbthe_%s' %hb)


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


  def get_total_energy(self):
      for mol in self.mols:
          mols = mol.split('-')[0] 
          if self.atomic:
             mol_ = mol
             for bd in self.bonds:
                 self.zpe[mol_] += self.p['be0_'+bd]*self.nbe0[mol][bd]
          else:
             mol_ = mols
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
                               self.zpe[mol_],name='E_%s' %mol)   


  def get_loss(self):
      self.Loss = 0.0
      for mol in self.mols:
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
         
          self.Loss     += self.loss[mol]
          self.accuracy += self.accur[mol]

      self.loss_atol = self.supervise()

      self.Loss     +=  self.loss_atol  
      self.accuracy  = self.accuracy/self.nmol


  def supervise(self):
      ''' supervised learning term'''
      l_atol = 0.0
      self.diffa,self.diffb,self.diffe,self.bosip = {},{},{},{}
      for bd in self.bonds: 
          [atomi,atomj] = bd.split('-') 
          if self.nbd[bd]>0:
             fao = tf.where(tf.greater(self.rbd[bd],self.rcuta[bd]),
                            tf.ones_like(self.rbd[bd]),tf.zeros_like(self.rbd[bd]))
             self.diffa[bd]  = tf.reduce_sum(input_tensor=tf.nn.relu(self.bo0[bd]*fao-self.atol))
             l_atol = tf.add(self.diffa[bd],l_atol)

             fbo = tf.where(tf.greater(self.rbd[bd],self.rcut[bd]),
                            tf.ones_like(self.rbd[bd]),tf.zeros_like(self.rbd[bd]))
             self.diffb[bd]  = tf.reduce_sum(input_tensor=tf.nn.relu(self.bo0[bd]*fbo-self.botol))
             l_atol = tf.add(self.diffb[bd],l_atol)

             fe  = tf.where(tf.less_equal(self.rbd[bd],self.re[bd]),
                            tf.ones_like(self.rbd[bd]),tf.zeros_like(self.rbd[bd]))
             self.diffe[bd]  = tf.reduce_sum(input_tensor=tf.nn.relu((self.bore-self.bo0[bd]))*fe)
             
             l_atol = tf.add(self.diffe[bd],l_atol)
             l_atol = tf.add(tf.nn.relu(self.rc_bo[bd]-self.rcut[bd]),l_atol)

             # if not self.nn:
             #    self.bosip[bd]  = tf.reduce_sum(input_tensor=self.bosi_pen[bd])
             #    l_atol = tf.add(self.bosip[bd],l_atol)
      return l_atol*self.bo_penalty


  def get_bond_energy(self):
      self.BOP = tf.zeros([1,self.batch])   # for ghost atom, the value is zero
      for bd in self.bonds:
          if self.nbd[bd]>0:
             with tf.compat.v1.name_scope('BOuc_'+bd):
                self.get_bondorder_uc(bd)
                self.BOP = tf.concat([self.BOP,self.bop[bd]],0)
  
      self.Dp     = tf.gather_nd(self.BOP,self.dlist)  
      self.Deltap = tf.reduce_sum(input_tensor=self.Dp,axis=1,name='Deltap')

      self.BO0    = tf.zeros([1,self.batch])   # for ghost atom, the value is zero
      self.BO     = tf.zeros([1,self.batch])
      self.BOPI   = tf.zeros([1,self.batch])
      self.BSO    = tf.zeros([1,self.batch])
      BPI         = tf.zeros([1,self.batch])

      for bd in self.bonds:
          [atomi,atomj] = bd.split('-') 
          if self.nbd[bd]==0:
             continue

          if self.nn:
             self.get_bondorder_nn(bd,atomi,atomj)
          else:
             self.get_bondorder(bd,atomi,atomj)

          self.BO0 = tf.concat([self.BO0,self.bo0[bd]],0)
          self.BO  = tf.concat([self.BO,self.bo[bd]],0)
          self.BSO = tf.concat([self.BSO,self.bso[bd]],0)
          BPI      = tf.concat([BPI,self.bopi[bd]+self.bopp[bd]],0)
          self.BOPI= tf.concat([self.BOPI,self.bopi[bd]],0)

      self.FBOT   = taper(self.BO0,rmin=self.atol,rmax=2.0*self.atol) 
      self.FHB    = taper(self.BO0,rmin=self.hbtol,rmax=2.0*self.hbtol) 

      D_  = tf.gather_nd(self.BO0,self.dlist,name='D_') 
      SO_ = tf.gather_nd(self.BSO,self.dlist,name='SO_') 
      self.BPI = tf.gather_nd(BPI,self.dlist,name='BPI') 

      self.Delta  = tf.reduce_sum(input_tensor=D_,axis=1,name='Delta')  # without valence i.e. - Val 
      self.SO     = tf.reduce_sum(input_tensor=SO_,axis=1,name='sumover')  
      
      i = 0
      for bd in self.bonds: 
          [atomi,atomj] = bd.split('-') 
          if self.nbd[bd]>0:
             with tf.compat.v1.name_scope('Ebond_'+bd):
                  [atomi,atomj] = bd.split('-') 
                  self.get_ebond(bd)
                  EBDA = self.EBD[bd] if i==0 else tf.concat((EBDA,self.EBD[bd]),0)
             i += 1

      for mol in self.mols:
          with tf.compat.v1.name_scope('Ebond_'+mol):
               self.ebda[mol] = tf.gather_nd(EBDA,self.bdlink[mol])  
               self.ebond[mol]= tf.reduce_sum(input_tensor=self.ebda[mol],axis=0,name='bondenergy')
               

  def get_ebond(self,bd):
      FBO  = tf.where(tf.greater(self.bosi[bd],0.0),
                      tf.ones_like(self.bosi[bd]),tf.zeros_like(self.bosi[bd]))
      FBOR = 1.0 - FBO
      self.powb[bd] = tf.pow(self.bosi[bd]+FBOR,self.p['be2_'+bd])
      self.expb[bd] = tf.exp(tf.multiply(self.p['be1_'+bd],1.0-self.powb[bd]))

      self.sieng[bd]= self.p['Desi_'+bd]*self.bosi[bd]*self.expb[bd]*FBO 
      self.pieng[bd]= tf.multiply(self.p['Depi_'+bd],self.bopi[bd])
      self.ppeng[bd]= tf.multiply(self.p['Depp_'+bd],self.bopp[bd]) 

      self.EBD[bd]  = - self.sieng[bd] - self.pieng[bd] - self.ppeng[bd]


  def get_atom_energy(self):
      i = 0
      for sp in self.spec:
          if self.nsp[sp]==0:
             continue
          self.eatom[sp] = -tf.ones([self.nsp[sp]])*self.p['atomic_'+sp]
          self.delta[sp]     = tf.gather_nd(self.Delta,self.atomlist[sp])
          self.dang[sp]  = self.delta[sp] - self.p['valang_'+sp]

          self.get_elone(sp,self.delta[sp]) 
          self.ELONE  = self.EL[sp] if i==0 else tf.concat((self.ELONE,self.EL[sp]),0)
          self.Dlp[sp]= self.delta[sp] - self.p['val_'+sp] - self.Delta_lp[sp]
          DLP_        = self.Dlp[sp] if i==0 else tf.concat((DLP_,self.Dlp[sp]),0)

          NLP_    = self.nlp[sp]  if i==0 else tf.concat((NLP_,self.nlp[sp]),0)
          DANG_   = self.dang[sp] if i==0 else tf.concat((DANG_,self.dang[sp]),0)
          i          += 1

      self.Dang= tf.gather_nd(DANG_,self.dalink)   
      self.NLP = tf.gather_nd(NLP_,self.dalink) 

      DLPL     = tf.gather_nd(DLP_,self.dalink)     
      self.DLP = tf.gather_nd(DLPL,self.dalist)  # warning: zero pitfal for dalist
      self.DPIL= tf.reduce_sum(input_tensor=self.BPI*self.DLP,axis=1,name='DPI') # 
      self.DPI = tf.reduce_sum(input_tensor=self.BPI,axis=1,name='DPI') # *self.DLP
    
      i = 0
      for sp in self.spec:
          if self.nsp[sp]==0:
             continue
          self.Dpi[sp]   = tf.gather_nd(self.DPIL,self.atomlist[sp])

          self.get_eover(sp,self.delta[sp],self.Dpi[sp])
          self.EOVER = self.EOV[sp] if i==0 else tf.concat((self.EOVER,self.EOV[sp]),0)

          self.get_eunder(sp,self.delta[sp],self.Dpi[sp])
          self.EUNDER = self.EUN[sp] if i==0 else tf.concat((self.EUNDER,self.EUN[sp]),0)
          self.EATOM  = self.eatom[sp] if i==0 else tf.concat((self.EATOM,self.eatom[sp]),0)
          i += 1

      for mol in self.mols:
          self.Elone[mol] = tf.gather_nd(self.ELONE,self.atomlink[mol])  
          self.elone[mol] = tf.reduce_sum(input_tensor=self.Elone[mol],axis=0,name='lonepairenergy')

          self.Eover[mol] = tf.gather_nd(self.EOVER,self.atomlink[mol])  
          self.eover[mol] = tf.reduce_sum(input_tensor=self.Eover[mol],axis=0,name='overenergy')

          self.Eunder[mol] = tf.gather_nd(self.EUNDER,self.atomlink[mol])  
          self.eunder[mol] = tf.reduce_sum(input_tensor=self.Eunder[mol],axis=0,name='underenergy')
    
          if self.atomic:
             zpe_ = tf.gather_nd(self.EATOM,self.atomlink[mol]) 
             self.zpe[mol] = tf.reduce_sum(input_tensor=zpe_,name='zpe') 


  def get_elone(self,atom,D):
      NLPOPT             = 0.5*(self.p['vale_'+atom] - self.p['val_'+atom])
      self.Delta_e[atom] = 0.5*(D - self.p['vale_'+atom])
      self.DE[atom]      = -tf.nn.relu(-tf.math.ceil(self.Delta_e[atom])) 

      self.nlp[atom] = -self.DE[atom] + tf.exp(-self.p['lp1']*4.0*tf.square(1.0+self.Delta_e[atom]-self.DE[atom]))
      
      self.Delta_lp[atom] = NLPOPT-self.nlp[atom]                                        # nan error
      # Delta_lp  = tf.clip_by_value(self.Delta_lp[atom],-1.0,10.0)  # temporary solution
      Delta_lp  = tf.nn.relu(self.Delta_lp[atom]+1) -1

      self.explp[atom]    = 1.0+tf.exp(-75.0*Delta_lp)

      self.EL[atom] = tf.math.divide(self.p['lp2_'+atom]*self.Delta_lp[atom],self.explp[atom],
                                     name='Elone_%s' %atom)


  def get_eover(self,atom,D,DPI):
      self.Delta_lpcorr[atom] = D - self.p['val_'+atom] - tf.math.divide(self.Delta_lp[atom],
                                1.0+self.p['ovun3']*tf.exp(self.p['ovun4']*DPI))

      self.so[atom]     = tf.gather_nd(self.SO,self.atomlist[atom])
      self.otrm1[atom]  = DIV_IF(1.0,self.Delta_lpcorr[atom]+self.p['val_'+atom])
      # self.otrm2[atom]  = tf.math.divide(1.0,1.0+tf.exp(self.p['ovun2_'+atom]*self.Delta_lpcorr[atom]))
      self.otrm2[atom]  = tf.sigmoid(-self.p['ovun2_'+atom]*self.Delta_lpcorr[atom])
      self.EOV[atom]    = self.so[atom]*self.otrm1[atom]*self.Delta_lpcorr[atom]*self.otrm2[atom]
    

  def get_eunder(self,atom,D,DPI):
      self.expeu1[atom] = tf.exp(self.p['ovun6']*self.Delta_lpcorr[atom])
      self.eu1[atom]    = tf.sigmoid(self.p['ovun2_'+atom]*self.Delta_lpcorr[atom])

      self.expeu3[atom] = tf.exp(self.p['ovun8']*DPI)
      self.eu2[atom]    = tf.math.divide(1.0,1.0+self.p['ovun7']*self.expeu3[atom])
      self.EUN[atom]    = -self.p['ovun5_'+atom]*(1.0-self.expeu1[atom])*self.eu1[atom]*self.eu2[atom]  # must positive


  def get_bondorder_uc(self,bd):
      self.frc[bd] = tf.where(tf.logical_or(tf.greater(self.rbd[bd],self.rc_bo[bd]),
                                            tf.less_equal(self.rbd[bd],0.001)),
                              tf.zeros_like(self.rbd[bd]),tf.ones_like(self.rbd[bd]))

      self.bodiv1[bd] = tf.math.divide(self.rbd[bd],self.p['rosi_'+bd],name='bodiv1_'+bd)
      self.bopow1[bd] = tf.pow(self.bodiv1[bd],self.p['bo2_'+bd])
      self.eterm1[bd] = (1.0+self.botol)*tf.exp(tf.multiply(self.p['bo1_'+bd],self.bopow1[bd]))*self.frc[bd] # consist with GULP

      self.bodiv2[bd] = tf.math.divide(self.rbd[bd],self.p['ropi_'+bd],name='bodiv2_'+bd)
      self.bopow2[bd] = tf.pow(self.bodiv2[bd],self.p['bo4_'+bd])
      self.eterm2[bd] = tf.exp(tf.multiply(self.p['bo3_'+bd],self.bopow2[bd]))*self.frc[bd]

      self.bodiv3[bd] = tf.math.divide(self.rbd[bd],self.p['ropp_'+bd],name='bodiv3_'+bd)
      self.bopow3[bd] = tf.pow(self.bodiv3[bd],self.p['bo6_'+bd])
      self.eterm3[bd] = tf.exp(tf.multiply(self.p['bo5_'+bd],self.bopow3[bd]))*self.frc[bd]

      self.bop_si[bd] = taper(self.eterm1[bd],rmin=self.botol,rmax=2.0*self.botol)*(self.eterm1[bd]-self.botol) # consist with GULP
      self.bop_pi[bd] = taper(self.eterm2[bd],rmin=self.botol,rmax=2.0*self.botol)*self.eterm2[bd]
      self.bop_pp[bd] = taper(self.eterm3[bd],rmin=self.botol,rmax=2.0*self.botol)*self.eterm3[bd]
      self.bop[bd]    = tf.add(self.bop_si[bd],self.bop_pi[bd]+self.bop_pp[bd],name='BOp_'+bd)


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
      o.append(tf.sigmoid(tf.matmul(X,self.m[pre+'wi_'+bd],name='bop_input')+self.m[pre+'bi_'+bd]))   # input layer

      for l in range(layer):                                                   # hidden layer      
          o.append(tf.sigmoid(tf.matmul(o[-1],self.m[pre+'w_'+bd][l],name='bop_hide')+self.m[pre+'b_'+bd][l]))

      o_ = tf.sigmoid(tf.matmul(o[-1],self.m[pre+'wo_'+bd],name='bop_output') + self.m[pre+'bo_'+bd])  # output layer
      out= tf.reshape(o_,[nbd,self.batch])
      return out


  def get_bondorder_nn(self,bd,atomi,atomj):
      Di   = tf.gather_nd(self.Deltap,self.dilink[bd])
      Dj   = tf.gather_nd(self.Deltap,self.djlink[bd])
      # Di_  = Di-self.p['val_'+atomi]
      # Dj_  = Dj-self.p['val_'+atomj]
      Dbi  = Di-self.bop[bd]
      Dbj  = Dj-self.bop[bd]
      b             = bd.split('-')
      bdr           = b[1]+'-'+b[0]
      
      Fi   = self.f_nn('f1',bd,self.nbd[bd],[Dbi,Dbj,self.bop[bd]],layer=self.bo_layer[1])
      Fj   = self.f_nn('f1',bdr,self.nbd[bd],[Dbj,Dbi,self.bop[bd]],layer=self.bo_layer[1])
      self.F[bd]    = 4.0*Fi*Fj

      self.bo0[bd]  = self.bop[bd]*self.F[bd]
      self.bo[bd]   = tf.nn.relu(self.bo0[bd] - self.atol)      #bond-order cut-off 0.001 reaxffatol
      self.bopi[bd] = self.bop_pi[bd]*self.F[bd]
      self.bopp[bd] = self.bop_pp[bd]*self.F[bd]
      self.bosi[bd] = self.bo0[bd] - self.bopi[bd] - self.bopp[bd]
      self.bso[bd]  = self.p['ovun1_'+bd]*self.p['Desi_'+bd]*self.bo0[bd]  


  def get_bondorder(self,bd,atomi,atomj):
      Di   = tf.gather_nd(self.Deltap,self.dilink[bd])
      Dj   = tf.gather_nd(self.Deltap,self.djlink[bd])

      self.f1(bd,atomi,atomj,Di,Dj)
      self.f45(bd,atomi,atomj,Di,Dj)

      F1  = lambda:self.f_1[bd]
      F45 = lambda:self.f_4[bd]*self.f_5[bd]
      one = lambda:tf.ones([self.nbd[bd],self.batch])

      case1= tf.logical_and(tf.greater_equal(self.p['ovcorr_'+bd],0.0001),
                            tf.greater_equal(self.p['corr13_'+bd],0.0001))

      self.F_11[bd] = tf.cond(pred=case1,true_fn=F1,false_fn=one)
      self.F_12[bd] = tf.cond(pred=tf.greater_equal(self.p['ovcorr_'+bd],0.0001),true_fn=F1,false_fn=one)
      self.F_45[bd] = tf.cond(pred=tf.greater_equal(self.p['corr13_'+bd],0.0001),true_fn=F45,false_fn=one)
      self.F[bd]    = self.F_11[bd]*self.F_12[bd]*self.F_45[bd]

      self.bo0[bd]  = self.bop[bd]*self.F_11[bd]*self.F_45[bd]  #-0.001        # consistent with GULP
      self.bo[bd]   = tf.nn.relu(self.bo0[bd] - self.atol)      #bond-order cut-off 0.001 reaxffatol
      self.bopi[bd] = self.bop_pi[bd]*self.F[bd]
      self.bopp[bd] = self.bop_pp[bd]*self.F[bd]
      self.bosi[bd] = tf.nn.relu(self.bo0[bd] - self.bopi[bd] - self.bopp[bd])  
      # self.bosi_pen[bd] = tf.nn.relu(self.f_1[bd] - 1.0)  
      # self.bosi_pen[bd] = tf.nn.relu(self.bopi[bd] + self.bopp[bd] - self.bo0[bd])  
      self.bso[bd]  = self.p['ovun1_'+bd]*self.p['Desi_'+bd]*self.bo0[bd]  


  def f1(self,bd,atomi,atomj,Di,Dj):
      Div = Di - self.p['val_'+atomi] # replace val in f1 with valp, 
      Djv = Dj - self.p['val_'+atomj] # different from published ReaxFF model
      self.f2(bd,Div,Djv)
      self.f3(bd,Div,Djv)
      self.f_1[bd] = 0.5*(DIV(self.p['val_'+atomi]+self.f_2[bd],
                          self.p['val_'+atomi]+self.f_2[bd]+self.f_3[bd]) + 
                      DIV(self.p['val_'+atomj]+self.f_2[bd],
                          self.p['val_'+atomj]+self.f_2[bd]+self.f_3[bd]))


  def f2(self,bd,Di,Dj):
      self.dexpf2[bd]  = tf.exp(-self.p['boc1']*Di)
      self.dexpf2t[bd] = tf.exp(-self.p['boc1']*Dj)
      self.f_2[bd]     = tf.add(self.dexpf2[bd],self.dexpf2t[bd])


  def f3(self,bd,Di,Dj):
      self.dexpf3[bd] = tf.exp(-self.p['boc2']*Di)
      self.dexpf3t[bd]= tf.exp(-self.p['boc2']*Dj)

      delta_exp       = self.dexpf3[bd]+self.dexpf3t[bd]
      dexp            = 0.5*delta_exp 

      self.f3log[bd] = tf.math.log(dexp)
      self.f_3[bd]   = tf.math.divide(-1.0,self.p['boc2'])*self.f3log[bd]


  def f45(self,bd,atomi,atomj,Di,Dj):
      self.Di_boc[bd] = Di - self.p['valboc_'+atomi] # + self.p['val_'+atomi]
      self.Dj_boc[bd] = Dj - self.p['valboc_'+atomj] # + self.p['val_'+atomj]
      
      # boc3 boc4 boc5 must positive
      boc3 = tf.sqrt(self.p['boc3_'+atomi]*self.p['boc3_'+atomj])
      boc4 = tf.sqrt(self.p['boc4_'+atomi]*self.p['boc4_'+atomj])
      boc5 = tf.sqrt(self.p['boc5_'+atomi]*self.p['boc5_'+atomj])
      
      self.df4[bd] = boc4*tf.square(self.bop[bd])-self.Di_boc[bd]
      self.f4r[bd] = tf.exp(-boc3*(self.df4[bd])+boc5)

      self.df5[bd] = boc4*tf.square(self.bop[bd])-self.Dj_boc[bd]
      self.f5r[bd] = tf.exp(-boc3*(self.df5[bd])+boc5)

      self.f_4[bd] = tf.math.divide(1.0,1.0+self.f4r[bd])
      self.f_5[bd] = tf.math.divide(1.0,1.0+self.f5r[bd])


  def get_angle_energy(self):
      self.PBOpow = tf.negative(tf.pow(self.BO,8)) # original: self.BO0 
      self.PBOexp = tf.exp(self.PBOpow)
      PBO_        = tf.gather_nd(self.PBOexp,self.dlist)
      self.PBO    = tf.reduce_prod(input_tensor=PBO_,axis=1)

      i = 0
      for ang in self.angs:
          if self.nang[ang]>0:
             atomj = ang.split('-')[1]    
             D     = tf.gather_nd(self.Delta,self.dglist[ang])
             Di    = tf.gather_nd(self.Delta,self.dgilist[ang])
             Dk    = tf.gather_nd(self.Delta,self.dgklist[ang])
             self.D_ang[ang] = tf.gather_nd(self.Dang,self.dglist[ang])
             # self.D_ang[ang] = D - self.p['valang_'+atomj]

             self.get_eangle(ang,atomj,D)
             self.get_epenalty(ang,atomj,D)
             self.get_three_conj(ang,atomj,D,Di,Dk)

      for mol in self.mols:
          i = 0
          for ang in self.angs:
              if self.nang[ang]>0:
                 if len(self.anglink[mol][ang])>0:
                    eang_ = tf.gather_nd(self.EANG[ang],self.anglink[mol][ang])
                    self.Eang[mol] = eang_ if i==0 else tf.concat((self.Eang[mol],eang_),0)

                    epen_ = tf.gather_nd(self.EPEN[ang],self.anglink[mol][ang])
                    self.Epen[mol] = epen_ if i==0 else tf.concat((self.Epen[mol],epen_),0)

                    etc_ = tf.gather_nd(self.ETC[ang],self.anglink[mol][ang])
                    self.Etc[mol] = etc_ if i==0 else tf.concat((self.Etc[mol],etc_),0)
                    i+=1
          if mol in self.Eang:
             self.eang[mol] = tf.reduce_sum(input_tensor=self.Eang[mol],axis=0,name='eang_%s' %mol)
             self.epen[mol] = tf.reduce_sum(input_tensor=self.Epen[mol],axis=0,name='epen_%s' %mol)
             self.tconj[mol]= tf.reduce_sum(input_tensor=self.Etc[mol],axis=0,name='etc_%s' %mol)
          else:
             self.eang[mol] = tf.cast(np.zeros([self.batch]),tf.float32)
             self.epen[mol] = tf.cast(np.zeros([self.batch]),tf.float32)
             self.tconj[mol]= tf.cast(np.zeros([self.batch]),tf.float32)


  def get_eangle(self,ang,atomj,D):
      self.BOij[ang] = tf.gather_nd(self.BO,self.boaij[ang])   ### need to be done
      self.BOjk[ang] = tf.gather_nd(self.BO,self.boajk[ang])   ### need to be done
      fij            = tf.gather_nd(self.FBOT,self.boaij[ang]) 
      fjk            = tf.gather_nd(self.FBOT,self.boajk[ang]) 
      self.fijk[ang] = fij*fjk

      with tf.compat.v1.name_scope('Theta0_%s' %ang):
           self.get_theta0(ang)

      self.thet[ang]    = self.theta0[ang]-self.theta[ang]
      self.expang[ang]  = tf.exp(-self.p['val2_'+ang]*tf.square(self.thet[ang]))
      self.f_7[ang]     = self.f7(ang,atomj)
      self.f_8[ang]     = self.f8(ang,atomj)
      self.EANG[ang] = self.fijk[ang]*self.f_7[ang]*self.f_8[ang]*(self.p['val1_'+ang]-self.p['val1_'+ang]*self.expang[ang]) 


  def get_theta0(self,ang):
      self.sbo[ang] = tf.gather_nd(self.DPI,self.dglist[ang])
      self.pbo[ang] = tf.gather_nd(self.PBO,self.dglist[ang])
      self.rnlp[ang]= tf.gather_nd(self.NLP,self.dglist[ang])
      self.SBO[ang] = self.sbo[ang] - tf.multiply(1.0-self.pbo[ang],self.D_ang[ang]+self.p['val8']*self.rnlp[ang])    
      
      ok         = tf.logical_and(tf.less_equal(self.SBO[ang],1.0),tf.greater(self.SBO[ang],0.0))
      S1         = tf.where(ok,self.SBO[ang],tf.zeros_like(self.SBO[ang]))    #  0< sbo < 1                  
      self.SBO01[ang] = tf.where(ok,tf.pow(S1,self.p['val9']),tf.zeros_like(S1)) 

      ok    = tf.logical_and(tf.less(self.SBO[ang],2.0),tf.greater(self.SBO[ang],1.0))
      S2    = tf.where(ok,self.SBO[ang],tf.zeros_like(self.SBO[ang]))                     
      F2    = tf.where(ok,tf.ones_like(S2),tf.zeros_like(S2))                             #  1< sbo <2
     
      S2    = 2.0*F2-S2  
      self.SBO12[ang] = tf.where(ok,2.0-tf.pow(S2,self.p['val9']),tf.zeros_like(self.SBO[ang]))     #  1< sbo <2
                                                                                          #     sbo >2
      SBO2  = tf.where(tf.greater_equal(self.SBO[ang],2.0),
                       tf.ones_like(self.SBO[ang]),tf.zeros_like(self.SBO[ang]))

      self.SBO3[ang]   = self.SBO01[ang]+self.SBO12[ang]+2.0*SBO2
      theta0 = 180.0 - self.p['theta0_'+ang]*(1.0-tf.exp(-self.p['val10']*(2.0-self.SBO3[ang])))
      self.theta0[ang] = theta0/57.29577951


  def f7(self,ang,atomj): 
      FBOi  = tf.where(tf.greater(self.BOij[ang],0.0),
                       tf.ones_like(self.BOij[ang]),tf.zeros_like(self.BOij[ang]))   
      FBORi = 1.0 - FBOi                                                            # prevent NAN error
      expij = tf.exp(-self.p['val3_'+atomj]*tf.pow(self.BOij[ang]+FBORi,self.p['val4_'+ang])*FBOi)

      FBOk  = tf.where(tf.greater(self.BOjk[ang],0.0),
                        tf.ones_like(self.BOjk[ang]),tf.zeros_like(self.BOjk[ang]))   
      FBORk = 1.0 - FBOk 
      expjk = tf.exp(-self.p['val3_'+atomj]*tf.pow(self.BOjk[ang]+FBORk,self.p['val4_'+ang])*FBOk)
      fi = 1.0 - expij
      fk = 1.0 - expjk
      F  = tf.multiply(fi,fk,name='f7_'+ang)
      return F 


  def f8(self,ang,atomj):
      exp6 = tf.exp( self.p['val6']*self.D_ang[ang])
      exp7 = tf.exp(-self.p['val7_'+ang]*self.D_ang[ang])
      F = self.p['val5_'+atomj] - (self.p['val5_'+atomj]-1.0)*tf.math.divide(2.0+exp6,1.0+exp6+exp7)
      return F


  def get_epenalty(self,ang,atomj,D):
      self.f_9[ang] = self.f9(ang,atomj,D)
      expi = tf.exp(-self.p['pen2']*tf.square(self.BOij[ang]-2.0))
      expk = tf.exp(-self.p['pen2']*tf.square(self.BOjk[ang]-2.0))
      self.EPEN[ang] = self.p['pen1_'+ang]*self.f_9[ang]*expi*expk*self.fijk[ang]


  def f9(self,ang,atomj,D):
      Delta= D - self.p['val_'+atomj]  
      exp3 = tf.exp(-self.p['pen3']*Delta)
      exp4 = tf.exp( self.p['pen4']*Delta)
      F = tf.math.divide(2.0+exp3,1.0+exp3+exp4)
      return F


  def get_three_conj(self,ang,atomj,D,Di,Dk):
      Dcoa = D - self.p['valboc_'+atomj]
      self.expcoa1[ang] = tf.exp(self.p['coa2']*Dcoa)

      self.texp0[ang] = tf.math.divide(self.p['coa1_'+ang],1.0+self.expcoa1[ang])  
      self.texp1[ang] = tf.exp(-self.p['coa3']*tf.square(Di-self.BOij[ang]))
      self.texp2[ang] = tf.exp(-self.p['coa3']*tf.square(Dk-self.BOjk[ang]))
      self.texp3[ang] = tf.exp(-self.p['coa4']*tf.square(self.BOij[ang]-1.5))
      self.texp4[ang] = tf.exp(-self.p['coa4']*tf.square(self.BOjk[ang]-1.5))
      self.ETC[ang] = self.texp0[ang]*self.texp1[ang]*self.texp2[ang]*self.texp3[ang]*self.texp4[ang]*self.fijk[ang] 


  def get_torsion_energy(self):
      for tor in self.tors:
          if self.ntor[tor]>0:
             self.get_etorsion(tor)
             self.get_four_conj(tor)

      for mol in self.mols:
          i = 0
          for tor in self.tors:
              if self.ntor[tor]>0:
                 if len(self.torlink[mol][tor])>0:
                    etor_ = tf.gather_nd(self.ETOR[tor],self.torlink[mol][tor])
                    self.Etor[mol] = etor_ if i==0 else tf.concat((self.Etor[mol],etor_),0)

                    Efcon_ = tf.gather_nd(self.Efcon[tor],self.torlink[mol][tor])
                    self.EFC[mol] = Efcon_ if i==0 else tf.concat((self.EFC[mol],Efcon_),0)
                    i+=1
          if mol in self.Etor:
             self.etor[mol] = tf.reduce_sum(input_tensor=self.Etor[mol],axis=0,name='etor_%s' %mol)
             self.efcon[mol]= tf.reduce_sum(input_tensor=self.EFC[mol],axis=0,name='efcon_%s' %mol)
          else:  # no torsion angle
             self.etor[mol] = tf.zeros([self.batch])
             self.efcon[mol]= tf.zeros([self.batch])


  def get_etorsion(self,tor):
      self.BOtij[tor]  = tf.gather_nd(self.BO,self.tij[tor])
      self.BOtjk[tor]  = tf.gather_nd(self.BO,self.tjk[tor])
      self.BOtkl[tor]  = tf.gather_nd(self.BO,self.tkl[tor])
      fij              = tf.gather_nd(self.FBOT,self.tij[tor])
      fjk              = tf.gather_nd(self.FBOT,self.tjk[tor])
      fkl              = tf.gather_nd(self.FBOT,self.tkl[tor])
      self.fijkl[tor]  = fij*fjk*fkl

      Dj    = tf.gather_nd(self.Dang,self.dtj[tor])
      Dk    = tf.gather_nd(self.Dang,self.dtk[tor])

      self.f_10[tor]   = self.f10(tor)
      self.f_11[tor]   = self.f11(tor,Dj,Dk)

      self.BOpjk[tor]  = tf.gather_nd(self.BOPI,self.tjk[tor]) 
      #   different from reaxff manual
      self.expv2[tor] = tf.exp(self.p['tor1_'+tor]*tf.square(2.0-self.BOpjk[tor]-self.f_11[tor])) 

      self.cos3w[tor] = tf.cos(3.0*self.w[tor])
      self.v1[tor] = 0.5*self.p['V1_'+tor]*(1.0+self.cos_w[tor])   
      self.v2[tor] = 0.5*self.p['V2_'+tor]*self.expv2[tor]*(1.0-self.cos2w[tor])
      self.v3[tor] = 0.5*self.p['V3_'+tor]*(1.0+self.cos3w[tor])

      self.ETOR[tor]=self.fijkl[tor]*self.f_10[tor]*self.s_ijk[tor]*self.s_jkl[tor]*(self.v1[tor]+self.v2[tor]+self.v3[tor])


  def f10(self,tor):
      with tf.compat.v1.name_scope('f10_%s' %tor):
           exp1 = 1.0 - tf.exp(-self.p['tor2']*self.BOtij[tor])
           exp2 = 1.0 - tf.exp(-self.p['tor2']*self.BOtjk[tor])
           exp3 = 1.0 - tf.exp(-self.p['tor2']*self.BOtkl[tor])
      return exp1*exp2*exp3


  def f11(self,tor,Dj,Dk):
      delt = Dj+Dk
      self.f11exp3[tor] = tf.exp(-self.p['tor3']*delt)
      self.f11exp4[tor] = tf.exp( self.p['tor4']*delt)
      f_11 = tf.math.divide(2.0+self.f11exp3[tor],1.0+self.f11exp3[tor]+self.f11exp4[tor])
      return f_11


  def get_four_conj(self,tor):
      exptol= tf.exp(-self.p['cot2']*tf.square(self.atol - 1.5))
      expij = tf.exp(-self.p['cot2']*tf.square(self.BOtij[tor]-1.5))-exptol
      expjk = tf.exp(-self.p['cot2']*tf.square(self.BOtjk[tor]-1.5))-exptol 
      expkl = tf.exp(-self.p['cot2']*tf.square(self.BOtkl[tor]-1.5))-exptol

      self.f_12[tor] = expij*expjk*expkl
      prod = 1.0+(tf.square(tf.cos(self.w[tor]))-1.0)*self.s_ijk[tor]*self.s_jkl[tor]
      self.Efcon[tor] = self.fijkl[tor]*self.f_12[tor]*self.p['cot1_'+tor]*prod  


  def f13(self,r,ai,aj):
      gw = tf.sqrt(self.p['gammaw_'+ai]*self.p['gammaw_'+aj])
      rr = tf.pow(r,self.p['vdw1'])+tf.pow(tf.math.divide(1.0,gw),self.p['vdw1'])
      f  = tf.pow(rr,tf.math.divide(1.0,self.p['vdw1']))  
      return f


  def get_tap(self,r):
      tp = 1.0+tf.math.divide(-35.0,tf.pow(self.vdwcut,4.0))*tf.pow(r,4.0)+ \
           tf.math.divide(84.0,tf.pow(self.vdwcut,5.0))*tf.pow(r,5.0)+ \
           tf.math.divide(-70.0,tf.pow(self.vdwcut,6.0))*tf.pow(r,6.0)+ \
           tf.math.divide(20.0,tf.pow(self.vdwcut,7.0))*tf.pow(r,7.0)
      return tp


  def get_evdw_image(self):
      rv_   = {}
      nc    = 0
      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  for vb in self.bonds:
                      if self.nvb[vb]>0:
                         box = self.pc[vb]*[i,j,k]
                         rv_[vb] = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(self.vr[vb]+box),axis=2))
                  evdw_ = self.get_vdw_energy(rv_)
                  if nc==0:
                     self.evdw = evdw_
                  else:
                     for mol in self.mols:
                         self.evdw[mol] += evdw_[mol]
                  nc += 1


  def get_ev(self,vb,rv):
      [ai,aj] = vb.split('-')
      gm      = tf.sqrt(self.p['gamma_'+ai]*self.p['gamma_'+aj])
      gm3     = tf.pow(tf.math.divide(1.0,gm),3.0)
      r3      = tf.pow(rv,3.0)
      fv      = tf.where(rv>self.vdwcut,tf.zeros_like(rv),tf.ones_like(rv))

      self.f_13[vb] = self.f13(rv,ai,aj)
      self.tpv[vb] = self.get_tap(rv)

      self.expvdw1[vb] = tf.exp(0.5*self.p['alfa_'+vb]*(1.0-tf.math.divide(self.f_13[vb],
                                2.0*self.p['rvdw_'+vb])))
      self.expvdw2[vb] = tf.square(self.expvdw1[vb]) 
      self.EVDW[vb]    = fv*self.tpv[vb]*self.p['Devdw_'+vb]*(self.expvdw2[vb]-2.0*self.expvdw1[vb])

      if self.optword.find('nocoul')<0:
         self.rth[vb]  = tf.pow(r3+gm3,1.0/3.0)
         self.ECOU[vb] = tf.math.divide(fv*self.tpv[vb]*self.qij[vb],self.rth[vb])


  def get_vdw_energy(self):
      evdw = {}
      for vb in self.bonds:
          if self.nvb[vb]>0:
             with tf.compat.v1.name_scope('vdW_%s' %vb):
                  self.get_ev(vb,self.rv[vb])

      for mol in self.mols:
          if self.optword.find('nocoul')>=0:
             self.ecoul[mol]= tf.constant(self.ecoul_[mol],dtype=tf.float32)

          with tf.compat.v1.name_scope('Evdw_'+mol):
             i = 0
             for vb in self.bonds:
                 if self.nvb[vb]>0:
                    if len(self.vlink[mol][vb])>0:
                       evdw_ = tf.gather_nd(self.EVDW[vb],self.vlink[mol][vb])
                       self.Evdw[mol] = evdw_ if i==0 else tf.concat((self.Evdw[mol],evdw_),0)
                       
                       if self.optword.find('nocoul')<0:
                          ecou_ = tf.gather_nd(self.ECOU[vb],self.vlink[mol][vb])
                          self.Ecou[mol] = ecou_ if i==0 else tf.concat((self.Ecou[mol],ecou_),0)
                       i+=1
             self.evdw[mol] = tf.reduce_sum(input_tensor=self.Evdw[mol],axis=0,name='evdw_%s' %mol)
             if self.optword.find('nocoul')<0:
                self.ecoul[mol]= tf.reduce_sum(input_tensor=self.Ecou[mol],axis=0,name='ecoul_%s' %mol)


  def get_hb_energy(self):
      for hb in self.hbs:
          if self.nhb[hb]>0:
             with tf.compat.v1.name_scope('ehb_%s' %hb):
                   self.get_ehb(hb)

      for mol in self.mols:
          with tf.compat.v1.name_scope('Ehb_'+mol):
               i = 0
               for hb in self.hbs:
                   if self.nhb[hb]>0:
                      if len(self.hblink[mol][hb])>0:
                         ehb_ = tf.gather_nd(self.EHB[hb],self.hblink[mol][hb])
                         self.Ehb[mol] = ehb_ if i==0 else tf.concat((self.Ehb[mol],ehb_),0)
                         i+=1
               if mol in self.Ehb: 
                  self.ehb[mol] = tf.reduce_sum(input_tensor=self.Ehb[mol],axis=0,name='ehb_%s' %mol)
               else: 
                  self.ehb[mol] = tf.cast(np.zeros([self.batch]),tf.float32) # case for no hydrogen-bonds in system


  def get_ehb(self,hb):
      self.BOhb[hb]   = tf.gather_nd(self.BO0,self.hij[hb]) 
      self.fhb[hb]    = tf.gather_nd(self.FHB,self.hij[hb]) 
      self.exphb1[hb] = 1.0-tf.exp(-self.p['hb1_'+hb]*self.BOhb[hb])
      sum_            = tf.math.divide(self.p['rohb_'+hb],self.rhb[hb])+tf.math.divide(self.rhb[hb],self.p['rohb_'+hb])-2.0
      self.exphb2[hb] = tf.exp(-self.p['hb2_'+hb]*sum_)
      # self.sin4[hb] = tf.pow(tf.sin(self.hbthe[hb]*0.5),4.0) 
      self.sin4[hb]   = tf.square(self.hbthe[hb])
      self.EHB[hb]    = self.fhb[hb]*self.frhb[hb]*self.p['Dehb_'+hb]*self.exphb1[hb]*self.exphb2[hb]*self.sin4[hb] 


  def set_zpe(self,molecules=None):
      for mol in self.mols:
          mols = mol.split('-')[0] 
          if mols not in self.zpe:
             if mols in self.zpe_:
                self.zpe[mols] = tf.Variable(tf.cast(self.zpe_[mols],tf.float32),name='zpe_'+mols)
             else:
                print('-  molecular %s energy set according atomic ...' %mols) 
                zpe_ = 0.0
                if not molecules is None:
                   for a in molecules[mol].atom_name:
                       zpe_ += self.p_['atomic_'+a]
                self.zpe[mols] = tf.Variable(-zpe_,name='zpe_'+mols)


  def set_neurons(self):
      self.unit = 4.3364432032e-2
      self.p_g  = ['boc1','boc2','coa2','ovun6',
                   'ovun7','ovun8','val6','lp1','val9','val10','tor2',
                   'tor3','tor4','cot2','coa4','ovun4',               # 
                   'ovun3','val8','coa3','pen2','pen3','pen4','vdw1',
                   'cutoff','acut','hbtol'] # #
                   # 'trip2','trip1','trip4','trip3' ,'swa','swb'
                   # tor3,tor4>0

      self.p_spec = ['valang','valboc','ovun5',
                     'lp2','boc4','boc3','boc5','rosi','ropi','ropp',
                     'ovun2','val3','val5','atomic',
                     'gammaw','gamma','mass','chi','mu',
                     'Devdw','rvdw','alfa'] # ,'val','vale','chi','mu','valp', 

      self.p_bond = ['Desi','Depi','Depp','bo5','bo6','ovun1',
                     'be0','be1','be2','bo3','bo4','bo1','bo2','corr13','ovcorr']

      self.p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp'] # 
      self.p_ang  = ['theta0','val1','val2','coa1','val7','val4','pen1'] # 
      self.p_hb   = ['rohb','Dehb','hb1','hb2']
      self.p_tor  = ['V1','V2','V3','tor1','cot1'] # 'tor2','tor3','tor4',

      self.punit  = ['Desi','Depi','Depp','lp2','ovun5','val1',
                     'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb'] # ,'hb1'

      cons = ['mass','corr13','ovcorr', 
              'trip1','trip2','trip3','trip4','swa','swb',
              'chi','mu'] 
              #'val', 'valboc','valang','vale','atomic','gamma'

      self.angopt = ['valang','Theta0','val1','val2','val3','val4','val5',
                     'val6','val7','val8','val9','val10',
                     'pen1','pen2','pen3','pen4',
                     'coa1','coa2','coa3','coa4','atomic',
                     'acut'] 
      self.toropt = ['V1','V2','V3',
                     'cot1','cot2',
                     'tor1','atomic',
                     'acut'] # 'tor2','tor3','tor4',
                     
      self.boopt = ['rosi','ropi','ropp',
                    'Desi','Depi','Depp','be1','be2',
                    'ovun1','ovun2','ovun3','ovun4','ovun5',
                    'ovun6','ovun7','ovun8',
                    'boc1','boc2','boc3','boc4','boc5',
                    'bo1','bo2','bo3','bo4','bo5','bo6',
                    'val','vale','valboc',
                    'lp1','lp2','atomic']
                    
      self.lopt = ['gammaw','vdw1','rvdw','Devdw','alfa',
                   'rohb','Dehb','hb1','hb2','atomic'] # 'gamma',

      # if self.optword.find('novdw')>=0:
      #    cons = cons + ['gamma','gammaw','vdw1','rvdw','Devdw','alfa']

      if self.optword.find('notor')>=0:
         cons = cons + ['tor2','tor3','tor4','V1','V2','V3','tor1','cot1','cot2'] # 

      if self.cons is None:
         self.cons = cons 
      else:
         self.cons += cons

      if self.nn:
         self.cons += ['boc1','boc2','boc3','boc4','boc5','valboc']

      if self.opt is None:
         self.opt = self.p_g+self.p_spec+self.p_bond+self.p_offd+self.p_ang+self.p_tor+self.p_hb
      
      self.nvopt = self.p_g+self.p_spec+self.p_bond+self.p_offd+self.p_ang+self.p_tor+self.p_hb
      for v in ['gammaw','vdw1','rvdw','Devdw','alfa','gamma']:
          self.nvopt.remove(v)

      self.torp = self.checkTors(self.torp)


  def clip_parameters(self):
      ''' clipe operation: clipe values in resonable range 
          exponential parameters (tf.exp) should be small,
          such as: val3,val6,val7,be1,lp1,ovun4,hb2,pen2
      '''   
      self.clip_pe = ['ovun3','ovun4','ovun6','ovun7',
                      'val1','val2','val3','val4','val5',
                      'val6','val8','val9','val10',
                      'pen2','pen3','pen4','coa4',
                      'theta0']  
                      # 'Depi','Depp','ovun1','ovun2','ovun5',
      for k in self.v:
          vn  = k.split('_')
          key = vn[0]
          if key == 'zpe':
             continue
          if key == 'n.u.':
             self.p[k] = self.v[k]
          elif key in self.clip_pe:
             if key in self.punit:
                self.p[k] = tf.clip_by_value(self.v[k],0.0,690.0*self.unit)
             else:
                self.p[k] = tf.clip_by_value(self.v[k],0.0,398.0)
          elif key=='atomic':
             self.p[k] = tf.clip_by_value(self.v[k],-99.9999,999.9999)
          elif key=='acut':
             self.p[k] = tf.clip_by_value(self.v[k],0.0020,0.1000)
          elif key=='hbtol':
             self.p[k] = tf.clip_by_value(self.v[k],0.0001,0.1)
          elif key=='cutoff':
             self.p[k] = tf.clip_by_value(self.v[k],0.01,2.0)
          elif key=='ovun5':
             self.p[k] = tf.clip_by_value(self.v[k],0.0,999.0*self.unit)
          elif key=='ovun1':
             self.p[k] = tf.clip_by_value(self.v[k],0.01,99.0)
          elif key in ['boc4','boc5','vdw1']:
             self.p[k] = tf.clip_by_value(self.v[k],0.00000001,100.0)
          elif key == 'boc3':
             self.p[k] = tf.clip_by_value(self.v[k],0.0,50.0)
          elif key == 'gammaw':
             self.p[k] = tf.clip_by_value(self.v[k],1.00,49.0)
          elif key == 'gamma':
             self.p[k] = tf.clip_by_value(self.v[k],0.001,6.0)
          elif key == 'pen1':
             self.p[k] = tf.clip_by_value(self.v[k],-69.0*self.unit,199.0*self.unit)
          elif key == 'Devdw':
             self.p[k] = tf.clip_by_value(self.v[k],0.01*self.unit,6.0*self.unit)
          elif key == 'Dehb':
             self.p[k] = tf.clip_by_value(self.v[k],-99.0*self.unit,99.0*self.unit)
          elif key == 'Desi':
             self.p[k] = tf.clip_by_value(self.v[k],60.000*self.unit,999.0*self.unit)
          elif key== 'Depi':
             self.p[k] = tf.clip_by_value(self.v[k],0.000*self.unit,999.0*self.unit)
          elif key== 'Depp':
             self.p[k] = tf.clip_by_value(self.v[k], 0.000*self.unit,999.0*self.unit)
          elif key=='tor1':
             self.p[k] = tf.clip_by_value(self.v[k],-99.9900,-0.0001)
          elif key== 'ovun2':
             self.p[k] = tf.clip_by_value(self.v[k],-39.00,-0.01)
          elif key in ['tor3','val7']:
             if key in self.punit:
                self.p[k] = tf.clip_by_value(self.v[k],0.00,48.0*self.unit)
             else:
                self.p[k] = tf.clip_by_value(self.v[k],0.00,48.0)
          elif key in ['tor2','tor4']:
             self.p[k] = tf.clip_by_value(self.v[k],0.00,28.0)
          elif key in ['coa1','cot1']:
             self.p[k] = tf.clip_by_value(self.v[k],-99.9900*self.unit,-0.000001*self.unit)
          elif key in ['V1','V2','V3']:
             self.p[k] = tf.clip_by_value(self.v[k],-99.9900*self.unit,990.0*self.unit)
          elif key in ['bo1','bo3','bo5']:
             self.p[k] = tf.clip_by_value(self.v[k],-50.0,-0.0)
          elif key in ['bo2','bo4','bo6']:
             self.p[k] = tf.clip_by_value(self.v[k],0.0,50.0)
          elif key == 'boc1':
             self.p[k] = tf.clip_by_value(self.v[k],0.001,50.0)
          elif key == 'boc2':
             self.p[k] = tf.clip_by_value(self.v[k],0.001,50.0)
          elif key == 'alfa':
             self.p[k] = tf.clip_by_value(self.v[k],0.001,40.0)
          elif key== 'lp1':
             self.p[k] = tf.clip_by_value(self.v[k],10.0,50.0)
          elif key== 'lp2':
             self.p[k] = tf.clip_by_value(self.v[k],0.0,99.0*self.unit)
          elif key == 'be1':
             self.p[k] = tf.clip_by_value(self.v[k],-6.0,6.0)
          elif key == 'be2':
             self.p[k] = tf.clip_by_value(self.v[k],0.01,19.0)
          elif key in ['coa2','coa3']:
             self.p[k] = tf.clip_by_value(self.v[k],-6.0,10.9)
          elif key == 'rosi':
             bd = vn[1]
             b  = bd.split('-')
             if len(b)==1:
                bd = b[0] +'-' +b[0]
             self.p[k] = tf.clip_by_value(self.v[k],0.80*self.re[bd],1.2*self.re[bd]) # 0.95 1.2
          elif key == 'ropi':
             self.p[k] = tf.clip_by_value(self.v[k],0.80*self.p_['rosi_'+vn[1]],0.9*self.p_['rosi_'+vn[1]])
          elif key == 'ropp':
             self.p[k] = tf.clip_by_value(self.v[k],0.70*self.p_['ropi_'+vn[1]],0.85*self.p_['ropi_'+vn[1]])
          elif key == 'rohb':
             self.p[k] = tf.clip_by_value(self.v[k],1.5,3.6)
          elif key == 'rvdw':
             rvdw_ = max(1.3*self.p_['rosi_'+vn[1]],1.5)
             self.p[k] = tf.clip_by_value(self.v[k],rvdw_,2.5*self.p_['rosi_'+vn[1]])
          elif key=='val':
             sp = vn[1]
             self.p[k] = tf.clip_by_value(self.v[k],0.2,8.0)
          elif key=='vale':
             sp = vn[1]
             self.p[k] = tf.clip_by_value(self.v[k],self.v['val_'+sp],8.0)
          elif key=='valboc':
             sp = vn[1]
             self.p[k] = tf.clip_by_value(self.v[k],0.0,100.0)
          elif key=='valang':
             sp = vn[1]
             self.p[k] = tf.clip_by_value(self.v[k],0.5*self.p_['val_'+sp],2.0*self.p_['val_'+sp])
          else:
             self.p[k] = self.v[k]
  

  def set_parameters_ud(self,opt=[],libfile=None):
      if not libfile is None:
         self.p_,zpe,spec,bonds,offd,angs,torp,hbs = read_lib(libfile=libfile)

      self.p,self.v = {},{}
      for k in self.p_:
          key = k.split('_')[0]
          ktor= ['cot1','V1','V2','V3']

          if self.optword.find('notor')>=0:
             if key in ktor:
                self.p_[k] = 0.0

          if key == 'zpe':
             continue
          if key != 'n.u.':
             if (k in self.VariablesToOpt) and (key in self.opt) and (key not in self.cons):
                if key in self.punit:
                   self.v[k] = tf.Variable(np.float32(self.unit*self.p_[k]),name=k)
                else:
                   self.v[k] = tf.Variable(np.float32(self.p_[k]),name=k)
             else:
                if key in self.punit:
                   self.v[k] = tf.constant(np.float32(self.unit*self.p_[k]),name=k)
                else:
                   self.v[k] = tf.constant(np.float32(self.p_[k]),name=k)

      if self.clip_op:
         self.clip_parameters()
      else:
         for k in self.v:
             key       = k.split('_')[0]
             self.p[k] = self.v[k]
             
      self.botol       = 0.01*self.p['cutoff']
      self.atol        = self.p['acut']
      self.hbtol       = self.p['hbtol']
      self.checkp()
      self.get_rcbo()

      if self.nn:
         self.set_m()


  def set_parameters(self,opt=[],libfile=None):
      if not libfile is None:
         self.p_,zpe,spec,bonds,offd,angs,torp,hbs = read_lib(libfile=libfile)

      self.p,self.v = {},{}
      for k in self.p_:
          key = k.split('_')[0]
          ktor= ['cot1','V1','V2','V3']

          if self.optword.find('notor')>=0:
             if key in ktor:
                self.p_[k] = 0.0

          if key == 'zpe':
             continue
          if key != 'n.u.':
             if key in self.cons:
                if key in self.punit:
                   self.v[k] = tf.constant(np.float32(self.unit*self.p_[k]),name=k)
                else:
                   self.v[k] = tf.constant(np.float32(self.p_[k]),name=k)
             elif key in self.opt:
                # with tf.variable_scope('Neurons'):
                trainable=True if key in opt else False       
                if key in self.punit:
                   self.v[k] = tf.Variable(np.float32(self.unit*self.p_[k]),
                                          trainable=trainable,name=k)
                elif key=='vale':
                   if self.conf_vale is None:
                      self.v[k] = tf.Variable(np.float32(self.p_[k]),
                                             trainable=trainable,name=k)
                   else:
                      e = k.split('_')[0]
                      if e in self.conf_vale:
                         self.v[k] = self.v['val_'+e]
                      else:
                         self.v[k] = tf.Variable(np.float32(self.p_[k]),
                                                trainable=trainable,name=k)
                else:
                   self.v[k] = tf.Variable(np.float32(self.p_[k]),
                                          trainable=trainable,name=k)
             else:
                if key in self.punit:
                   self.v[k] = tf.constant(np.float32(self.unit*self.p_[k]),name=k)
                else:
                   self.v[k] = tf.constant(np.float32(self.p_[k]),name=k)
          else:
             self.v[k] = tf.constant(self.p_[k])

      if self.clip_op:
         self.clip_parameters()
      else:
         for k in self.v:
             key       = k.split('_')[0]
             self.p[k] = self.v[k]
             
      self.botol       = 0.01*self.p['cutoff']
      self.atol        = self.p['acut']
      self.hbtol       = self.p['hbtol']
      self.checkp()
      self.get_rcbo()

      if self.nn:
         self.set_m()


  def set_m(self):
      ''' set variable for neural networks '''
      reuse_m = True if self.bo_layer==self.bo_layer_ else False
      bond = []
      for si in self.spec:
          for sj in self.spec:
              bd = si + '-' + sj
              if bd not in bond:
                 bond.append(bd)
      self.set_wb(pref='f1',reuse_m=reuse_m,nin=3,nout=1,layer=self.bo_layer,vlist=bond)


  def set_wb(self,pref='f',reuse_m=True,nin=8,nout=3,layer=[8,9],vlist=None):
      ''' set matix varibles '''
      for bd in vlist:
          if pref+'wi_'+bd in self.m_ and reuse_m:                   # input layer
              if self.nnopt:
                 # print(self.m_['fwi_'+bd])
                 self.m[pref+'wi_'+bd] = tf.Variable(self.m_[pref+'wi_'+bd],name=pref+'wi_'+bd)
                 self.m[pref+'bi_'+bd] = tf.Variable(self.m_[pref+'bi_'+bd],name=pref+'bi_'+bd)
              else:
                 self.m[pref+'wi_'+bd] = tf.constant(self.m_[pref+'wi_'+bd],name=pref+'wi_'+bd)
                 self.m[pref+'bi_'+bd] = tf.constant(self.m_[pref+'bi_'+bd],name=pref+'bi_'+bd)
          else:
              self.m[pref+'wi_'+bd] = tf.Variable(tf.random.normal([nin,layer[0]],stddev=0.2),name=pref+'wi_'+bd)   
              self.m[pref+'bi_'+bd] = tf.Variable(tf.random.normal([layer[0]],stddev=0.2),name=pref+'bi_'+bd)  
       
          self.m[pref+'w_'+bd] = []                                    # hidden layer
          self.m[pref+'b_'+bd] = []
          if pref+'w_'+bd in self.m_ and reuse_m:     
              if self.nnopt:                            
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
                                                           stddev=0.20),name=pref+'wh'+str(i)+'_'+bd)) 
                  self.m[pref+'b_'+bd].append(tf.Variable(tf.random.normal([layer[0]], 
                                                           stddev=0.20),name=pref+'bh'+str(i)+'_'+bd )) 

          if pref+'wo_'+bd in self.m_ and reuse_m:                                 # output layer
              if self.nnopt:       
                 self.m[pref+'wo_'+bd] = tf.Variable(self.m_[pref+'wo_'+bd],name=pref+'wo_'+bd)
                 self.m[pref+'bo_'+bd] = tf.Variable(self.m_[pref+'bo_'+bd],name=pref+'bo_'+bd)
              else:
                 self.m[pref+'wo_'+bd] = tf.constant(self.m_[pref+'wo_'+bd],name=pref+'wo_'+bd)
                 self.m[pref+'bo_'+bd] = tf.constant(self.m_[pref+'bo_'+bd],name=pref+'bo_'+bd)
          else:
              self.m[pref+'wo_'+bd] = tf.Variable(tf.random.normal([layer[0],nout],stddev=0.10)-1.0, name=pref+'wo_'+bd)   
              self.m[pref+'bo_'+bd] = tf.Variable(tf.random.normal([nout], stddev=0.1)+1.0,name=pref+'bo_'+bd)
         

  def checkp(self):
      for key in self.p_offd:
          for sp in self.spec:
              try:
                 self.p[key+'_'+sp+'-'+sp]  = self.p[key+'_'+sp]  
              except KeyError:
                 print('-  warning: key not in dict') 

      self.tors = []
      for spi in self.spec:
          for spj in self.spec:
              for spk in self.spec:
                  for spl in self.spec:
                      tor = spi+'-'+spj+'-'+spk+'-'+spl
                      torr= spl+'-'+spk+'-'+spj+'-'+spi
                      if (not tor in self.tors) and (not torr in self.tors):
                         if tor in self.torp:
                            self.tors.append(tor)
                         elif torr in self.torp:
                            self.tors.append(torr)
                         else:
                            self.tors.append(tor)

      for key in self.p_tor:
          for tor in self.tors:
              if tor not in self.torp:
                 [t1,t2,t3,t4] = tor.split('-')
                 tor1 = t1+'-'+t3+'-'+t2+'-'+t4
                 tor2 = t4+'-'+t3+'-'+t2+'-'+t1
                 tor3 = t4+'-'+t2+'-'+t3+'-'+t1
                 if tor1 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor1]
                 elif tor2 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor2]
                 elif tor3 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor3]    
                 else:
                    print('-  an error case for %s .........' %tor)


  def checkTors(self,torp):
      tors_ = torp
      for tor in tors_:
          [t1,t2,t3,t4] = tor.split('-')
          tor1 = t1+'-'+t3+'-'+t2+'-'+t4
          tor2 = t4+'-'+t3+'-'+t2+'-'+t1
          tor3 = t4+'-'+t2+'-'+t3+'-'+t1

          if tor1 in torp and tor1!=tor:
             # print('-  dict %s is repeated, delteting ...' %tor1)
             torp.remove(tor1)
          elif tor2 in self.torp and tor2!=tor:
             # print('-  dict %s is repeated, delteting ...' %tor2)
             torp.remove(tor2)
          elif tor3 in self.torp and tor3!=tor:
             # print('-  dict %s is repeated, delteting ...' %tor3)
             torp.remove(tor3)  
      return torp 
      

  def get_rcbo(self):
      self.rc_bo = {}
      for bd in self.bonds:
          b= bd.split('-')
          ofd=bd if b[0]!=b[1] else b[0]

          log_ = tf.math.log((self.botol/(1.0 + self.botol)))
          rr = log_/self.p['bo1_'+bd] 
         
          self.rc_bo[bd]=self.p['rosi_'+ofd]*tf.pow(log_/self.p['bo1_'+bd],1.0/self.p['bo2_'+bd])


  def session(self,learning_rate=3.0-4,method='AdamOptimizer'):
      if self.ncpu is None:
         self.config = tf.compat.v1.ConfigProto()
         # self.config.gpu_options.allow_growth = True
      else:
         self.config = tf.compat.v1.ConfigProto(device_count={'CPU':self.ncpu},
                                   inter_op_parallelism_threads=2,
                                   intra_op_parallelism_threads=self.ncpu,
                                   allow_soft_placement=True,
                                   log_device_placement=False) 
         self.config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1
      if self.interactive:
         self.sess = tf.compat.v1.InteractiveSession(config=self.config) 
      else: 
         self.sess= tf.compat.v1.Session(config=self.config)  

         if self.board:
            writer = tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)
            # see logs using command: tensorboard --logdir logs

         if method=='GradientDescentOptimizer':
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate) 
         elif method=='AdamOptimizer':
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate) 
         elif method=='AdagradOptimizer':
            optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate) 
         elif method=='MomentumOptimizer':
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate,0.9)  #momentum=0.9
         elif method=='RMSPropOptimizer':
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)
         elif method=='NadagradOptimizer':
            optimizer = tf.train.NadagradOptimizer(learning_rate) 

         if self.to_train and self.optMethod=='ADAM':
            self.train_step = optimizer.minimize(self.Loss)

      self.sess.run(tf.compat.v1.global_variables_initializer())  
      self.sess_build = True


  def update(self):
      self.logger.info('-  updating variables ...')
      upop = []
      for key in self.v:
          if key in self.opt:
             if not key in self.cons:
                upop.append(tf.compat.v1.assign(self.v[key],self.p_[key]))
      self.sess.run(upop)


  def reset(self,opt=[],libfile=None):
      if self.InitCheck:
         self.p_ = self.ic.check(self.p_)

      if self.VariablesToOpt is None:
         self.set_parameters(opt=opt,libfile=libfile)
      else:
         self.set_parameters_ud()

      self.memory()
      if not self.atomic:
         self.set_zpe()
         
      self.build_graph() 
      self.feed_dict = self.feed_data()   


  def sa(self,total_step=10000,step=5000,astep=5000,bstep=1000,
          print_step=10,writelib=1000):
      ''' simulated anealing auto train '''
      for i in range(total_step):
          self.nnopt= True
          if i!=0: self.reset(opt=self.opt)
          self.logger.info('-  optiming all variables ...')
          loss,accu,accMax,i = self.run(learning_rate=1.0e-4,method='AdamOptimizer',
                      step=step,print_step=print_step,writelib=writelib)
          if np.isnan(loss):
             print('-  Job continue, the loss is NaN.')
             self.p_  = self.ic.auto(self.p_)
          
          self.nnopt  = False
          self.reset(opt=self.angopt+self.toropt+self.lopt)
          self.logger.info('-  optiming angle and torsion variables ...')
          loss,accu,accMax,i = self.run(learning_rate=1.0e-3,method='AdamOptimizer',
                      step=astep,print_step=print_step,writelib=writelib)
          if np.isnan(loss):
             print('-  Job continue, the loss is NaN.')
             self.p_  = self.ic.auto(self.p_)


  def run(self,learning_rate=1.0e-4,method='AdamOptimizer',
               step=2000,print_step=10,writelib=100):
      if not self.initialized:
         self.initialize()
      self.session(learning_rate=learning_rate,method=method)  

      accs_={}
      libfile = self.libfile.split('.')[0]

      for i in range(step+1):
          loss,latol,accu,accs,_ = self.sess.run([self.Loss,
                                                  self.loss_atol,
                                                  self.accuracy,
                                                  self.accur,
                                                  self.train_step],
                                                  feed_dict=self.feed_dict)
          if i==0:
             accMax = accu
          else:
             if accu>accMax:
                accMax = accu
                
          loss_ = loss - latol
          if np.isnan(loss):
             self.logger.info('NAN error encountered at step %d loss is %f.' %(i,loss/self.nframe))
             # send_msg('NAN error encountered at step %d loss is %f.' %(i,loss/self.nframe))
             tf.compat.v1.reset_default_graph()
             self.sess.close()
             return 0.0,0.0,accMax,i
             
          if i%print_step==0:
             current = time.time()
             elapsed_time = current - self.time

             acc = ''
             for key in accs:
                 acc += key+': %6.4f ' %accs[key]

             self.logger.info('-  step: %d sqe: %6.4f accs: %f %s spv: %6.4f time: %6.4f' %(i,loss_,accu,acc,latol,elapsed_time))
             self.time = current

          if i%writelib==0 or i==step:
             self.lib_bk = libfile+'_'+str(i)
             self.write_lib(libfile=self.lib_bk,loss=loss_)
                
             if i==step:
                self.write_lib(libfile=libfile,loss=loss_)
                E,dfte = self.sess.run([self.E,self.dft_energy],
                                        feed_dict=self.feed_dict)
                self.plot_result(i,E,dfte)

          if accu>self.convergence:
             self.accu = accu
             E,dfte = self.sess.run([self.E,self.dft_energy],
                                     feed_dict=self.feed_dict)
             self.plot_result(None,E,dfte)
             self.write_lib(libfile=libfile,loss=loss_)
             print('-  Convergence Occurred, job compeleted.')
             tf.compat.v1.reset_default_graph()
             self.sess.close()
             return loss_,accu,accMax,i

      tf.compat.v1.reset_default_graph()
      self.sess.close()
      return loss_,accu,accMax,i


  def feed_data(self,indexs=None):
      feed_dict = {}
      for mol in self.mols:
          feed_dict[self.dft_energy[mol]] = self.lk.dft_energy[mol]

      for bd in self.bonds:
          if self.nbd[bd]>0:
             feed_dict[self.rbd[bd]] = self.lk.rbd[bd]
          if self.nvb[bd]>0:
             feed_dict[self.rv[bd]]  = self.lk.rv[bd]
             if self.optword.find('nocoul')<0:
                feed_dict[self.qij[bd]] = self.lk.qij[bd]
                # print(self.lk.vr[bd])
                # feed_dict[self.pc[bd]]  = self.lk.pc[bd]

      for ang in self.angs:
          if self.nang[ang]>0:
             feed_dict[self.theta[ang]] = self.lk.theta[ang]

      for tor in self.tors:
          if self.ntor[tor]>0:
             feed_dict[self.s_ijk[tor]] = self.lk.s_ijk[tor]
             feed_dict[self.s_jkl[tor]] = self.lk.s_jkl[tor]
             feed_dict[self.w[tor]]     = self.lk.w[tor]

      for hb in self.hbs:
          if self.nhb[hb]>0:
             # feed_dict[self.rik[hb]] = self.lk.rik[hb]
             feed_dict[self.rhb[hb]]   = self.lk.rhb[hb]
             feed_dict[self.frhb[hb]]  = self.lk.frhb[hb]
             feed_dict[self.hbthe[hb]] = self.lk.hbthe[hb]
      return feed_dict


  def calculate_energy(self):
      energy = self.get_value(self.E[mol])
      return energy


  def calculate_forces(self):
      forces = 0.0
      return forces


  def get_value(self,var):
      if self.interactive:
         if type(var).__name__=='dict':
            v = {}
            for key in var:
                v[key] = var[key].eval(feed_dict=self.feed_dict)
         else:
            # print(var)
            v = var.eval(feed_dict=self.feed_dict)
      else:
         if type(var).__name__=='dict':
            v = {}
            for key in var:
                v[key] = self.sess.run(var[key],feed_dict=self.feed_dict)
         else:
            v = self.sess.run(var,feed_dict=self.feed_dict)
      return v


  def get_gradient(self,var,gvar):
      gd = tf.gradients(ys=var,xs=gvar)
      if gd[0] is None:
         g = None
      else:
         grad = tf.convert_to_tensor(value=gd)
         if self.interactive:
            g = grad.eval(feed_dict=self.feed_dict)
         else:        
            g = self.sess.run(grad,feed_dict=self.feed_dict)
      return g


  def get_all_gradient(self):
      tlist = tf.compat.v1.trainable_variables()
      grads = tf.gradients(ys=self.Loss, xs=tlist)
      grad,tn,tl  = [],[],[]
      for gg,t in zip(grads,tlist):
          if not gg is None:
             grad.append(gg)
             tl.append(t)
             tn.append(t.name)
      g,t = self.sess.run([grad,tl],feed_dict=self.feed_dict)
      return g,t,tn


  def write_lib(self,libfile='ffield',loss=None):
      self.p_   = self.sess.run(self.p)
      
      if self.atomic:
         self.zpe_ = None
      else:
         self.zpe_ = self.sess.run(self.zpe)
         for key in self.zpe_:
             self.zpe_[key] = np.float32(self.zpe_[key])

      for k in self.p_:
          self.p_[k] = float(self.p_[k])
          key = k.split('_')[0]
          if key in self.punit:
             self.p_[k] = float(self.p_[k]/self.unit)

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

         fj = open(libfile+'.json','w')
         j = {'p':self.p_,'m':self.m_,
              'bo_layer':self.bo_layer,
              'zpe':self.zpe_}
         js.dump(j,fj,sort_keys=True,indent=2)
         fj.close()
      else:
         write_lib(self.p_,self.spec,self.bonds,self.offd,
                   self.angs,self.torp,self.hbs,
                   zpe=self.zpe_,libfile=libfile,
                   loss=loss)


  def plot(self):
      ''' plot out some data'''
      for mol in self.mols:
          E    = self.sess.run(self.ebond[mol],feed_dict=self.feed_dict)
          plt.figure()
          plt.ylabel('Energies')
          plt.xlabel('Step')

          plt.plot(E,label=r'$ReaxFF$', color='blue', linewidth=2, linestyle='--')
          plt.legend()
          plt.savefig('e_%s.eps' %mol) 
          plt.close()


  def plot_result(self,step,E,dfte):
      for mol in self.mols:
          maxe = self.max_e[mol]
          plt.figure()
          plt.ylabel('Energies comparation between DFT and ReaxFF')
          plt.xlabel('Step')
          err      = dfte[mol] - E[mol]

          plt.plot(self.lx,dfte[mol]-maxe,linestyle='-',marker='o',markerfacecolor='snow',
                   markeredgewidth=1,markeredgecolor='k',
                   ms=5,c='k',alpha=0.01,label=r'$DFT$')
          plt.plot(E[mol]-maxe,linestyle='-',marker='^',markerfacecolor='snow',
                   markeredgewidth=1,markeredgecolor='b',
                   ms=5,c='b',alpha=0.01,label=r'$I-ReaxFF$')
          # plt.errorbar(self.lx,E[mol]-maxe,yerr=err,
          #              fmt='-s',ecolor='r',color='r',ms=4,markerfacecolor='none',mec='blue',
          #              elinewidth=2,capsize=2,label='I-ReaxFF')

          plt.legend(loc='best',edgecolor='yellowgreen')
          if step is None:
          	 plt.savefig('result_%s.eps' %mol) 
          else:
             plt.savefig('result_%s_%s.eps' %(mol,step)) # transparent=True
          plt.close()


  def close(self):
      print('-  Job compeleted.')
      # self.sess.close()
      self.lk     = None
      self.ic     = None
      self.m_     = None
      self.m      = None
      self.v      = None
      self.p      = None
      self.p_     = None
      self.frc    = None
      self.bodiv1,self.bodiv2,self.bodiv3 = None,None,None
      self.bopow1,self.bopow2,self.bopow3 = None,None,None
      self.eterm1,self.eterm2,self.eterm3 = None,None,None
      self.bop_si,self.bop_pi,self.bop_pp,self.bop = None,None,None,None
      self.bosi,self.bosi_pen = None,None
      self.bopi,self.bopp,self.bo0,self.bo,self.bso = None,None,None,None,None

      self.f_1 = None
      self.dexpf2,self.dexpf2t,self.f_2=None,None,None
      self.dexpf3,self.dexpf3t,self.f_3,self.f3log=None,None,None,None
      self.Di,self.Dj,self.Di_boc,self.Dj_boc=None,None,None,None

      self.f4r,self.f5r,self.f_4,self.f_5,self.df4,self.df5=None,None,None,None,None,None
      self.F,self.F_11,self.F_12,self.F_45=None,None,None,None
      self.fbot = None
      self.powb,self.expb,self.EBD,self.ebond,self.ebda = None,None,None,None,None
      self.sieng,self.pieng,self.ppeng = None,None,None

      self.D,self.Delta_e,self.DE,self.nlp,self.Delta_lp,self.Dlp = None,None,None,None,None,None
      self.Dpi,self.BSO,self.BOPI,self.Delta_lpcorr = None,None,None,None
      self.explp,self.EL,self.elone,self.Elone,self.ELONE = None,None,None,None,None
      self.EOV,self.Eover,self.eover,self.otrm1,self.otrm2 =  None,None,None,None,None

      self.expeu1,self.expeu2,self.eu1,self.expeu3,self.eu2,self.EUN = None,None,None,None,None,None
      self.eunder,self.Eunder = None,None

      self.EANG,self.Eang,self.eang,self.theta0,self.fijk = None,None,None,None,None
      self.pbo,self.sbo,self.SBO,self.SBO12,self.SBO3,self.SBO01 = None,None,None,None,None,None
      self.dang,self.D_ang = None,None

      self.thet,self.expang,self.f_7,self.f_8,self.rnlp = None,None,None,None,None
      self.EPEN,self.BOij,self.BOjk,self.Epen,self.epen = None,None,None,None,None

      self.expcoa1,self.texp0,self.texp1,self.texp2,self.texp3 = None,None,None,None,None
      self.texp4,self.ETC,self.tconj,self.Etc = None,None,None,None

      self.cos3w,self.etor,self.Etor,self.ETOR = None,None,None,None
      self.BOpjk,self.BOtij,self.BOtjk,self.BOtkl,self.fijkl,self.so = None,None,None,None,None,None
      self.f_9,self.f_10,self.f_11,self.f_12,self.expv2 = None,None,None,None,None
      self.f11exp3,self.f11exp4 = None,None
      
      self.v1,self.v2,self.v3 = None,None,None
      self.Efcon,self.EFC,self.efcon = None,None,None
      self.expvdw1,self.expvdw2,self.EVDW,self.Evdw,self.f_13=None,None,None,None,None

      self.Ecou,self.ECOU,self.evdw,self.ecoul,self.tpv,self.rth = None,None,None,None,None,None
 
      self.exphb1,self.exphb2,self.sin4,self.EHB = None,None,None,None
      self.pc,self.fhb,self.BOhb,self.ehb,self.Ehb = None,None,None,None,None

      self.dft_energy,self.E,self.zpe,self.eatom,self.loss,self.accur = None,None,None,None,None,None



def test_reax(direcs=None,batch_size=200,dft='siesta'):
    ''' test reax with GULP, and run validation-set'''
    RE,GE={},{}
    rn = ReaxFF(libfile='ffield',
                direcs=direcs,dft=dft,
                opt=[],optword='nocoul',
                batch_size=batch_size,
                atomic=True,
                clip_op=False,
                interactive=True,
                to_train=False) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')
    
    
    for mol in direcs:

        RE['Total-Energy'] = rn.get_value(rn.E[mol])
        RE['ebond']        = rn.get_value(rn.ebond[mol])
        RE['elonepair']    = rn.get_value(rn.elone[mol])
        RE['eover']        = rn.get_value(rn.eover[mol])
        RE['eunder']       = rn.get_value(rn.eunder[mol])
        RE['eangle']       = rn.get_value(rn.eang[mol])
        RE['epenalty']     = rn.get_value(rn.epen[mol])
        RE['econjugation'] = rn.get_value(rn.tconj[mol])
        RE['etorsion']     = rn.get_value(rn.etor[mol])
        RE['evdw']         = rn.get_value(rn.evdw[mol])
        RE['ehb']          = rn.get_value(rn.ehb[mol])
        RE['fconj']        = rn.get_value(rn.efcon[mol])
        RE['ecoul']        = rn.get_value(rn.ecoul[mol])
        RE['eself']        = rn.eself[mol] # rn.get_value(rn.eself[mol])

        if dft=='nwchem':
           cell = [(10, 0, 0), (0, 10, 0), (0, 0, 10)]
        else:
           cell = rn.cell[mol]

        e    = rn.get_value(rn.E[mol])
        dfte = rn.get_value(rn.dft_energy[mol]) 
        zpe  = rn.get_value(rn.zpe[mol])

        GE['ebond'],GE['elonepair'],GE['eover'],GE['eunder'],GE['eangle'], \
        GE['econjugation'],GE['evdw'],GE['Total-Energy'] = \
                 [],[],[],[],[],[],[],[]
        GE['epenalty'],GE['etorsion'],GE['fconj'],GE['ecoul'],GE['eself'],GE['ehb']\
               = [],[],[],[],[],[] 

        for nf in range(batch_size):
            A = Atoms(symbols=molecules[mol].atom_name,
                      positions=molecules[mol].x[nf],
                      cell=cell,
                      pbc=(1, 1, 1))
            write_gulp_in(A,runword='gradient nosymmetry conv qite verb')
            system('gulp<inp-gulp>out')
            e_,eb_,el_,eo_,eu_,ea_,ep_,etc_,et_,ef_,ev_,ehb_,ecl_,esl_= \
                get_reax_energy(fo='out')

            GE['ebond'].append(eb_)
            GE['elonepair'].append(el_)
            GE['eover'].append(eo_)
            GE['eunder'].append(eu_)
            GE['eangle'].append(ea_)
            GE['econjugation'].append(etc_)
            GE['epenalty'].append(ep_)
            GE['etorsion'].append(et_)
            GE['fconj'].append(ef_)
            GE['evdw'].append(ev_)
            GE['ecoul'].append(ecl_)
            GE['ehb'].append(ehb_)
            GE['eself'].append(esl_)
            GE['Total-Energy'].append(e_)

        GE['Total-Energy'] = np.array(GE['Total-Energy']) + zpe
        

        x = np.linspace(0,len(RE['Total-Energy']),len(RE['Total-Energy']))

        for key in GE:
            plt.figure()             # test
            plt.ylabel('%s (eV)' %key)
            plt.xlabel('Step')
            
            # err = GE[key] - RE[key]
            plt.plot(x,GE[key],linestyle='-',marker='o',markerfacecolor='none',
                     markeredgewidth=1,markeredgecolor='k',
                     ms=7,c='k',alpha=0.01,label='GULP')
            plt.plot(x,RE[key],linestyle='-',marker='^',markerfacecolor='none',  # snow
                     markeredgewidth=1,markeredgecolor='b',
                     ms=7,c='b',alpha=0.01,label='I-ReaxFF')

            # plt.errorbar(x,RE[key],yerr=err,
            #              fmt='-.s',ecolor='r',color='r',ms=4,mfc='none',mec='blue',
            #              elinewidth=2,capsize=2,label='I-ReaxFF')

            plt.legend(loc='best',edgecolor='yellowgreen')
            plt.savefig('compare_%s_%s.eps' %(mol,key),transparent=True) 
            plt.close() 
        # system('epstopdf compare_%s.eps' %key) 

    rn.sess.close()



