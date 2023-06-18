import matplotlib.pyplot as plt
from os import makedirs
from os.path import exists
import time
import tensorflow as tf
import numpy as np
import json as js
from .reax_data import get_data,Dataset
from .reaxfflib import read_ffield,write_ffield,write_lib
from .intCheck import Intelligent_Check
from .RadiusCutOff import setRcut
from .reax import logger,taper,DIV_IF,clip_parameters,set_variables
from .mpnn import fmessage,fnn,set_matrix
# tf_upgrade_v2 --infile reax.py --outfile reax_v1.py
# tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


def find_torsion_angle(atomi,atomj,atomk,atoml,tors):
    tor1 = atomi+'-'+atomj+'-'+atomk+'-'+atoml
    tor2 = atoml+'-'+atomk+'-'+atomj+'-'+atomi
    if tor1 in tors:
       return tor1
    elif tor2 in tors:
       return tor2  
    else:
       raise RuntimeError('-  Torsion angle {:s} not find in ffield!'.format(tor1))

class ReaxFF_nn(object):
  def __init__(self,libfile='ffield',dataset={},
               dft='ase',atoms=None,
               cons=['val','vale','valang','vale','lp3','cutoff','hbtol'],# 'acut''val','valboc',
               opt=None,optword='nocoul',
               mpopt=None,bdopt=None,mfopt=None,eaopt=[],
               VariablesToOpt=None,
               batch=200,sample='uniform',
               hbshort=6.75,hblong=7.5,
               vdwcut=10.0,
               atol=0.001,
               hbtol=0.001,
               bore={'others':0.0},
               weight={'others':1.0},
               spv_vdw=False,vlo={'others':[(0.0,0.0)]},vup={'others':[(10.0,0.0)]},
               spv_bo=None,                 # e.g. spv_bo={'C-C':(1.3,8.5,8.5,0.0,0.0)}
               interactive=False,
               ro_scale=0.1,
               clip_op=True,
               clip={'others':(0.4,1.0)},   # parameter range
               InitCheck=True,
               resetDeadNeuron=False,
               optmol=True,
               lambda_me=0.1,
               nnopt=True,
               be_universal_nn=None,be_layer=[9,0],
               mf_universal_nn=None,mf_layer=[9,0],
               messages=1,MessageFunction=2,
               bo_layer=None,
               spec=[],
               board=False,
               lambda_bd=100000.0,
               lambda_pi=1.0,
               lambda_reg=0.01,
               lambda_ang=1.0,
               regularize_be=True,
               regularize_mf=True,
               regularize_bias=False,
               spv_ang=False,
               fixrcbo=False,
               to_train=True,
               #optMethod='ADAM',
               maxstep=60000,
               emse=0.9,
               convergence=0.97,
               lossConvergence=1000.0,
               losFunc='n2',
               conf_vale=None,
               huber_d=30.0,
               ncpu=None):
      '''
           ReaxFF-nn: Reactive Force Field with Neural Network for Bond-Order and Bond Energy.
           2022-10-22
      '''
      self.dataset       = dataset 
      self.libfile       = libfile
      self.batch_size    = batch
      self.sample        = sample        # uniform or random
      self.opt           = opt
      self.VariablesToOpt= VariablesToOpt
      self.cons          = cons
      self.optword       = optword
      self.optmol        = optmol
      self.lambda_me     = lambda_me
      self.vdwcut        = vdwcut
      self.dft           = dft
      self.atoms         = atoms
      self.ro_scale      = ro_scale
      self.conf_vale     = conf_vale
      self.clip_op       = clip_op
      self.clip          = clip
      self.InitCheck     = InitCheck
      self.resetDeadNeuron = resetDeadNeuron
      self.hbshort       = hbshort
      self.hblong        = hblong
      self.nnopt         = nnopt
      self.mfopt         = mfopt         # specify the element of message function to be optimized
      if mpopt is None:
         self.mpopt      = [True,True,True,True]
      else:
         self.mpopt      = mpopt
      self.bdopt         = bdopt
      self.eaopt         = eaopt
      self.regularize_be = regularize_be
      self.regularize_mf = regularize_mf
      self.regularize_bias= regularize_bias
      if regularize_mf or regularize_be:
         self.regularize = True
      else:
         self.regularize = False
      self.lambda_reg    = lambda_reg
      self.lambda_pi     = lambda_pi
      self.lambda_ang    = lambda_ang
      self.mf_layer      = mf_layer
      self.be_layer      = be_layer
      self.be_universal_nn = be_universal_nn
      self.mf_universal_nn = mf_universal_nn
      self.messages        = messages
      self.MessageFunction = MessageFunction
      self.spv_vdw       = spv_vdw
      self.spv_ang       = spv_ang
      self.spv_bo        = spv_bo
      self.vlo           = vlo
      self.vup           = vup
      self.bo_layer      = bo_layer
      self.weight        = weight
      self.spec          = spec
      self.time          = time.time()
      self.interactive   = interactive
      self.board         = board 
      self.to_train      = to_train
      self.maxstep       = maxstep
      self.emse          = emse
      #self.optMethod     = optMethod
      self.convergence   = convergence
      self.lossConvergence = lossConvergence
      self.losFunc       = losFunc
      self.huber_d       = huber_d
      self.ncpu          = ncpu
      self.bore          = bore
      #self.atol         = atol     # angle bond-order tolerence
      #self.hbtol        = hbtol    # hydrogen-bond bond-order tolerence
      self.fixrcbo       = fixrcbo
      self.m_,self.m     = None,None

      self.rcut,self.rcuta,self.re = self.read_lib()
      self.set_rcut(self.rcut,self.rcuta,self.re)

      self.set_variable_list()
      self.ic = Intelligent_Check(re=self.re,clip=clip,spec=self.spec,bonds=self.bonds,
                                  offd=self.offd,angs=self.angs,tors=self.torp,ptor=self.p_tor)
      self.p_,self.m_ = self.ic.check(self.p_,self.m_,resetDeadNeuron=self.resetDeadNeuron)

      if not self.libfile.endswith('.json'):
         self.p_['acut']    = atol
         self.p_['hbtol']   = hbtol

      self.torp          = self.checkTors(self.torp)
      
      self.lambda_bd     = lambda_bd
      self.logger        = logger('training.log')
      self.initialized   = False
      self.sess_build    = False

      if self.VariablesToOpt is None:
         self.set_parameters()
      else:
         self.set_parameters_to_opt()

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
      rcut_,rcuta_,re_ = setRcut(self.bonds,rcut,rcuta,re)
      self.rcut        = rcut_
      self.rcuta       = rcuta_
      self.re          = re_

  def initialize(self): 
      self.nframe      = 0
      molecules        = {}
      self.max_e       = {}
      # self.cell      = {}
      self.mols        = []
      self.batch       = {}
      self.eself,self.evdw_,self.ecoul_ = {},{},{}

      for mol in self.dataset: 
          nindex = []
          for key in molecules:
              if self.dataset[key]==self.dataset[mol]:
                 nindex.extend(molecules[key].indexs)
          data_ =  get_data(structure=mol,
                                direc=self.dataset[mol],
                                 # dft=self.dft,
                                atoms=self.atoms,
                               vdwcut=self.vdwcut,
                                 rcut=self.rcut,
                                rcuta=self.rcuta,
                              hbshort=self.hbshort,
                               hblong=self.hblong,
                                batch=self.batch_size,
                       variable_batch=True,
                               sample=self.sample,
                       p=self.p_,spec=self.spec,bonds=self.bonds,
                  angs=self.angs,tors=self.tors,
                                  hbs=self.hbs,
                               nindex=nindex)

          if data_.status:
             self.mols.append(mol)
             molecules[mol]   = data_
             self.batch[mol]  = molecules[mol].batch
             self.nframe     += self.batch[mol]
             print('-  max energy of %s: %f.' %(mol,molecules[mol].max_e))
             self.max_e[mol]  = molecules[mol].max_e
             # self.evdw_[mol]= molecules[mol].evdw
             self.ecoul_[mol] = molecules[mol].ecoul  
             self.eself[mol]  = molecules[mol].eself     
             # self.cell[mol] = molecules[mol].cell
          else:
             print('-  data status of %s:' %mol,data_.status)
      self.nmol = len(molecules)

      self.generate_data(molecules)
      with tf.compat.v1.name_scope('input'):
           self.memory(molecules=molecules)
      print('-  generating dataset ...')
      self.set_zpe(molecules=molecules)

      self.build_graph()    
      self.feed_dict = self.feed_data()

      self.initialized = True
      return molecules

  def generate_data(self,molecules):
      ''' get data '''
      self.blist,self.bdid             = {},{}
      self.dilink,self.djlink          = {},{}
      self.nbd,self.b,self.a,self.t    = {},{},{},{}
      self.ang_i,self.ang_j,self.ang_k = {},{},{}
      self.abij,self.abjk              = {},{}
      self.tij,self.tjk,self.tkl       = {},{},{}
      self.tor_j,self.tor_k            = {},{}
      self.tor_i,self.tor_l            = {},{}
      self.atom_name                   = {}
      self.natom                       = {}
      self.nv                          = {}
      self.na                          = {}
      self.nt                          = {}
      self.nh                          = {}
      self.v                           = {}
      self.h                           = {}
      self.hij                         = {}
      self.data                        = {}
      for m in molecules:
          self.natom[m]    = molecules[m].natom
          self.blist[m]    = molecules[m].blist
          self.dilink[m]   = molecules[m].dilink
          self.djlink[m]   = molecules[m].djlink
          
          self.ang_j[m]    = np.expand_dims(molecules[m].ang_j,axis=1)
          self.ang_i[m]    = np.expand_dims(molecules[m].ang_i,axis=1)
          self.ang_k[m]    = np.expand_dims(molecules[m].ang_k,axis=1)
          self.abij[m]     = molecules[m].abij
          self.abjk[m]     = molecules[m].abjk

          self.tij[m]      = molecules[m].tij
          self.tjk[m]      = molecules[m].tjk
          self.tkl[m]      = molecules[m].tkl
          
          self.tor_i[m]    = np.expand_dims(molecules[m].tor_i,axis=1)
          self.tor_j[m]    = np.expand_dims(molecules[m].tor_j,axis=1)
          self.tor_k[m]    = np.expand_dims(molecules[m].tor_k,axis=1)
          self.tor_l[m]    = np.expand_dims(molecules[m].tor_l,axis=1)

          self.nbd[m]      = molecules[m].nbd
          self.na[m]       = molecules[m].na
          self.nt[m]       = molecules[m].nt
          self.nv[m]       = molecules[m].nv
          self.b[m]        = molecules[m].B
          self.a[m]        = molecules[m].A
          self.t[m]        = molecules[m].T
          self.v[m]        = molecules[m].V
          self.nh[m]       = molecules[m].nh
          self.h[m]        = molecules[m].H
          self.hij[m]      = molecules[m].hij
          self.bdid[m]     = molecules[m].bond  # bond index like pair (i,j).
          self.atom_name[m]= molecules[m].atom_name
          self.data[m]     = Dataset(molecules[m].energy_nw,
                                     molecules[m].forces,
                                     molecules[m].rbd,
                                     molecules[m].rv,
                                     molecules[m].qij,
                                     molecules[m].theta,
                                     molecules[m].s_ijk,
                                     molecules[m].s_jkl,
                                     molecules[m].w,
                                     molecules[m].rhb,
                                     molecules[m].frhb,
                                     molecules[m].hbthe)

  def memory(self,molecules):
      self.frc = {}
      self.bop_si,self.bop_pi,self.bop_pp,self.bop = {},{},{},{}
      self.bosi,self.bosi_pen = {},{}
      self.bopi,self.bopp,self.bo0,self.bo,self.bso = {},{},{},{},{}

      self.Deltap,self.Delta,self.Bp = {},{},{}
      self.delta,self.Di,self.Dj,self.Di_boc,self.Dj_boc={},{},{},{},{}
      self.D,self.H,self.Hsi,self.Hpi,self.Hpp = {},{},{},{},{}

      self.SO,self.fbot,self.fhb = {},{},{}
      self.EBD,self.E = {},{}
      self.powb,self.expb,self.ebond = {},{},{}

      self.esi   = {}
      self.ebd   = {}
      self.rbd   = {}
      self.rbd_  = {}
      self.Dbi   = {}
      self.Dbj   = {}
      #self.bop_ = {}
      #self.bo0_ = {}
      for mol in self.mols:
          self.esi[mol]   = {}
          self.ebd[mol]   = {}
          self.rbd_[mol]  = {}
          self.Dbi[mol]   = {}
          self.Dbj[mol]   = {}
          #self.bop_[mol]  = {}
          #self.bo0_[mol]  = {}

      self.Delta_e,self.DE,self.Delta_lp,self.Dlp,self.Dang  = {},{},{},{},{}
      self.Bpi,self.Dpi,self.BSO,self.BOpi,self.Delta_lpcorr = {},{},{},{},{}
      self.eover,self.eunder,self.elone,self.Elone= {},{},{},{}
      self.EOV,self.EUN = {},{}

      self.BOij,self.BOjk = {},{}
      self.nlp,self.Pbo = {},{}
      self.Eang,self.eang,self.theta0,self.fijk = {},{},{},{}
      self.pbo,self.sbo,self.SBO,self.SBO12,self.SBO3,self.SBO01 = {},{},{},{},{},{}
      self.dang,self.D_ang = {},{}
      self.thet,self.thet2,self.expang,self.f_7,self.f_8,self.rnlp = {},{},{},{},{},{}
      self.Epen,self.epen = {},{}

      self.expcoa1,self.texp0,self.texp1,self.texp2,self.texp3 = {},{},{},{},{}
      self.texp4,self.tconj,self.Etc = {},{},{}

      self.cos3w,self.etor,self.Etor = {},{},{}
      self.BOpjk,self.BOtij,self.BOtjk,self.BOtkl,self.fijkl,self.so = {},{},{},{},{},{}
      self.f_9,self.f_10,self.f_11,self.f_12,self.expv2 = {},{},{},{},{}
      self.f11exp3,self.f11exp4 = {},{}

      self.v1,self.v2,self.v3 = {},{},{}
      self.Efcon,self.efcon = {},{}

      self.Evdw,self.nvb = {},{}
      self.Ecou,self.evdw,self.ecoul,self.tpv,self.rth = {},{},{},{},{}

      self.exphb1,self.exphb2,self.sin4,self.EHB = {},{},{},{}
      self.pc,self.BOhb,self.ehb,self.Ehb = {},{},{},{}

      self.dft_energy,self.E,self.zpe,self.eatom,self.forces = {},{},{},{},{}
      self.loss,self.penalty,self.accur,self.MolEnergy = {},{},{},{}

      self.rv,self.qij = {},{}
      self.theta = {}
      self.s_ijk,self.s_jkl,self.cos_w,self.cos2w,self.w={},{},{},{},{}
      self.rhb,self.frhb,self.hbthe = {},{},{}
      self.nang,self.ntor,self.nhb  = {},{},{}

      for mol in self.mols:
          self.dft_energy[mol] = tf.compat.v1.placeholder(tf.float32,shape=[self.batch[mol]],
                                                name='DFT_energy_%s' %mol)

          self.rbd[mol] = tf.compat.v1.placeholder(tf.float32,shape=[molecules[mol].nbond,self.batch[mol]],
                                                   name='rbd_%s' %mol)
          self.nvb[mol] = molecules[mol].nvb
          self.rv[mol]  = tf.compat.v1.placeholder(tf.float32,shape=[self.nvb[mol],self.batch[mol]],
                                                   name='rvdw_%s' %mol)
          self.qij[mol] = tf.compat.v1.placeholder(tf.float32,shape=[self.nvb[mol],self.batch[mol]],
                                                   name='qij_%s' %mol)
          self.nang[mol] = molecules[mol].nang
          if self.nang[mol]>0:                             
             self.theta[mol] = tf.compat.v1.placeholder(tf.float32,shape=[self.nang[mol],self.batch[mol]],
                                                     name='theta_%s' %mol)
          self.ntor[mol] = molecules[mol].ntor
          if self.ntor[mol]>0:
             self.s_ijk[mol] = tf.compat.v1.placeholder(tf.float32,shape=[self.ntor[mol],self.batch[mol]],
                                                      name='sijk_%s' %mol)
             self.s_jkl[mol] = tf.compat.v1.placeholder(tf.float32,shape=[self.ntor[mol],self.batch[mol]],
                                                      name='sjkl_%s' %mol)
             self.w[mol]     = tf.compat.v1.placeholder(tf.float32,shape=[self.ntor[mol],self.batch[mol]],
                                             name='w_%s' %mol)
             self.cos_w[mol] = tf.cos(self.w[mol])
             self.cos2w[mol] = tf.cos(2.0*self.w[mol])

      
          self.nhb[mol] = molecules[mol].nhb
          if self.nhb[mol]>0:
             self.rhb[mol]  = tf.compat.v1.placeholder(tf.float32,shape=[self.nhb[mol],self.batch[mol]],
                                            name='rhb_%s' %mol)
             self.frhb[mol] = tf.compat.v1.placeholder(tf.float32,shape=[self.nhb[mol],self.batch[mol]],
                                            name='frhb_%s' %mol)
             self.hbthe[mol]= tf.compat.v1.placeholder(tf.float32,shape=[self.nhb[mol],self.batch[mol]],
                                            name='hbthe_%s' %mol)
          if self.data[mol].forces is not None:
             self.forces[mol] = tf.compat.v1.placeholder(tf.float32,shape=[self.natom[mol],3,self.batch[mol]],
                                            name='forces_%s' %mol)

  def build_graph(self):
      print('-  building graph ...')
      self.accuracy   = tf.constant(0.0,name='accuracy')
      self.accuracies = {}
      for mol in self.mols:
          self.get_bond_energy(mol)
          self.get_atom_energy(mol)
          self.get_threebody_energy(mol)
          self.get_fourbody_energy(mol)
          self.get_vdw_energy(mol)
          self.get_hb_energy(mol)
          self.get_total_energy(mol)
      self.get_loss()
      print('-  end of build.')

  def get_total_energy(self,mol):
      ''' compute the total energy of moecule '''
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

  def get_delta(self,mol):
      ''' compute the uncorrected Delta: the sum of BO '''
      BOP = tf.zeros([1,self.batch[mol]])   # for ghost atom, the value is zero
      self.get_bondorder_uc(mol)
      BOP              = tf.concat([BOP,self.bop[mol]],0)
      # BOP            = tf.transpose(BOP,perm=[0,1])
      self.Bp[mol]     = tf.gather_nd(BOP,self.blist[mol])
      self.Deltap[mol] = tf.reduce_sum(input_tensor=self.Bp[mol],axis=1,name='Deltap')

  def get_bond_energy(self,mol):
      self.get_delta(mol)
      self.message_passing(mol)
      self.get_final_sate(mol)
      self.get_ebond(mol)
      self.ebond[mol]= tf.reduce_sum(input_tensor=self.EBD[mol],axis=0,name='bondenergy')

  def get_ebond(self,mol):
      Ebd = []
      
      for bd in self.bonds:
          nbd_ = self.nbd[mol][bd]
          if nbd_==0:
             continue
          b_  = self.b[mol][bd]
          bosi_ = tf.slice(self.bosi[mol],[b_[0],0],[b_[1],self.batch[mol]])
          bopi_ = tf.slice(self.bopi[mol],[b_[0],0],[b_[1],self.batch[mol]])
          bopp_ = tf.slice(self.bopp[mol],[b_[0],0],[b_[1],self.batch[mol]])

          self.esi[mol][bd] = fnn('fe',bd, self.nbd[mol][bd],[bosi_,bopi_,bopp_],
                    self.m,batch=self.batch[mol],layer=self.be_layer[1])
          self.ebd[mol][bd] = -self.p['Desi_'+bd]*self.esi[mol][bd]
          Ebd.append(self.ebd[mol][bd])
      self.EBD[mol] = tf.concat(Ebd,0)

  def get_bondorder_uc(self,mol):
      bop_si,bop_pi,bop_pp = [],[],[]
      for bd in self.bonds:
          nbd_ = self.nbd[mol][bd]
          if nbd_==0:
             continue
          b_  = self.b[mol][bd]
          self.rbd_[mol][bd] = tf.slice(self.rbd[mol],[b_[0],0],[b_[1],self.batch[mol]])
          self.frc[bd] = tf.where(tf.logical_or(tf.greater(self.rbd_[mol][bd],self.rc_bo[bd]),
                                                tf.less_equal(self.rbd_[mol][bd],0.001)), 0.0,1.0)

          bodiv1 = tf.math.divide(self.rbd_[mol][bd],self.p['rosi_'+bd],name='bodiv1_'+bd)
          bopow1 = tf.pow(bodiv1,self.p['bo2_'+bd])
          eterm1 = (1.0+self.botol)*tf.exp(tf.multiply(self.p['bo1_'+bd],bopow1))*self.frc[bd] # consist with GULP

          bodiv2 = tf.math.divide(self.rbd_[mol][bd],self.p['ropi_'+bd],name='bodiv2_'+bd)
          bopow2 = tf.pow(bodiv2,self.p['bo4_'+bd])
          eterm2 = tf.exp(tf.multiply(self.p['bo3_'+bd],bopow2))*self.frc[bd]

          bodiv3 = tf.math.divide(self.rbd_[mol][bd],self.p['ropp_'+bd],name='bodiv3_'+bd)
          bopow3 = tf.pow(bodiv3,self.p['bo6_'+bd])
          eterm3 = tf.exp(tf.multiply(self.p['bo5_'+bd],bopow3))*self.frc[bd]

          bop_si.append(taper(eterm1,rmin=self.botol,rmax=2.0*self.botol)*(eterm1-self.botol)) # consist with GULP
          bop_pi.append(taper(eterm2,rmin=self.botol,rmax=2.0*self.botol)*eterm2)
          bop_pp.append(taper(eterm3,rmin=self.botol,rmax=2.0*self.botol)*eterm3)

      self.bop_si[mol] = tf.concat(bop_si,0)
      self.bop_pi[mol] = tf.concat(bop_pi,0)
      self.bop_pp[mol] = tf.concat(bop_pp,0)
      self.bop[mol]    = self.bop_si[mol] + self.bop_pi[mol] + self.bop_pp[mol]

  def dbop_dr():
      ''' derivetiv of bop/dr '''
      ### TODO
      return None
      
  def get_bondorder(self,mol,t):
      ''' compute bond-order according the message function '''
      flabel  = 'fm'
      bosi_ = []
      bopi_ = []
      bopp_ = []

      for bd in self.bonds:
          nbd_ = self.nbd[mol][bd]
          if nbd_==0:
             continue
          b_   = self.b[mol][bd]
          Di   = tf.gather_nd(self.D[mol][t-1],self.dilink[mol][bd]) 
          Dj   = tf.gather_nd(self.D[mol][t-1],self.djlink[mol][bd])

          h    = tf.slice(self.H[mol][t-1],[b_[0],0],[b_[1],self.batch[mol]],name=bd+'_h_slice')
          hsi  = tf.slice(self.Hsi[mol][t-1],[b_[0],0],[b_[1],self.batch[mol]],name=bd+'_hsi_slice')
          hpi  = tf.slice(self.Hpi[mol][t-1],[b_[0],0],[b_[1],self.batch[mol]],name=bd+'_hpi_slice')
          hpp  = tf.slice(self.Hpp[mol][t-1],[b_[0],0],[b_[1],self.batch[mol]],name=bd+'_hpp_slice')

          b    = bd.split('-')
          self.Dbi[mol][bd]  = Di - h   
          self.Dbj[mol][bd]  = Dj - h   
          
          Fi   = fmessage(flabel,b[0],self.nbd[mol][bd],[self.Dbi[mol][bd],h,self.Dbj[mol][bd]],self.m,
                          batch=self.batch[mol],layer=self.mf_layer[1])
          Fj   = fmessage(flabel,b[1],self.nbd[mol][bd],[self.Dbj[mol][bd],h,self.Dbi[mol][bd]],self.m,
                          batch=self.batch[mol],layer=self.mf_layer[1])
          F    = Fi*Fj
          Fsi,Fpi,Fpp = tf.unstack(F,axis=2)

          bosi_.append(hsi*Fsi)
          bopi_.append(hpi*Fpi)
          bopp_.append(hpp*Fpp)

      bosi = tf.concat(bosi_,0)
      bopi = tf.concat(bopi_,0)
      bopp = tf.concat(bopp_,0)
      bo   = bosi+bopi+bopp
      return bo,bosi,bopi,bopp

  def message_passing(self,mol):
      ''' finding the final Bond-order with a message passing '''
      self.H[mol]   = [self.bop[mol]]                     # 
      self.Hsi[mol] = [self.bop_si[mol]]                  #
      self.Hpi[mol] = [self.bop_pi[mol]]                  #
      self.Hpp[mol] = [self.bop_pp[mol]]                  # 
      self.D[mol]   = [self.Deltap[mol]]                  # get the initial hidden state H[0]

      for t in range(1,self.messages+1):
          print('-  message passing for {:s} t={:d} ...'.format(mol,t))           
          BO    = tf.zeros([1,self.batch[mol]])           # for ghost atom, the value is zero
          bo,bosi,bopi,bopp = self.get_bondorder(mol,t)
          self.H[mol].append(bo)                      # get the hidden state H[t]
          self.Hsi[mol].append(bosi)
          self.Hpi[mol].append(bopi)
          self.Hpp[mol].append(bopp)

          BO      = tf.concat([BO,bo],0)
          D       = tf.gather_nd(BO,self.blist[mol])  
          Delta   = tf.reduce_sum(input_tensor=D,axis=1)
          self.D[mol].append(Delta)                  # degree matrix

  def get_final_sate(self,mol):     
      self.Delta[mol]  = self.D[mol][-1]
      self.bo0[mol]    = self.H[mol][-1]                 # fetch the final state 
      self.bosi[mol]   = self.Hsi[mol][-1]
      self.bopi[mol]   = self.Hpi[mol][-1]
      self.bopp[mol]   = self.Hpp[mol][-1]

      zero             = tf.zeros([1,self.batch[mol]])   # for ghost atom, the value is zero
      self.bo[mol]     = tf.nn.relu(self.bo0[mol] - self.atol)

      bso              = []
      for bd in self.bonds:
          if self.nbd[mol][bd]==0:
             continue
          b_   = self.b[mol][bd]
          bo0  = tf.slice(self.bo0[mol],[b_[0],0],[b_[1],self.batch[mol]],name=bd+'_slice')
          bso.append(self.p['ovun1_'+bd]*self.p['Desi_'+bd]*bo0)
      
      self.bso[mol]    = tf.concat(bso,0)
      BSO              = tf.concat([zero,self.bso[mol]],0)
      BPI              = tf.concat([zero,self.bopi[mol]+self.bopp[mol]],0)

      SO_              = tf.gather_nd(BSO,self.blist[mol],name='SO') 
      self.Bpi[mol]    = tf.gather_nd(BPI,self.blist[mol],name='Bpi') 
      self.Dpi[mol]    = tf.reduce_sum(input_tensor=self.Bpi[mol],axis=1,name='sumover_Bpi') 
      self.SO[mol]     = tf.reduce_sum(input_tensor=SO_,axis=1,name='sumover_bso')  
      self.fbot[mol]   = taper(self.bo0[mol],rmin=self.atol,rmax=2.0*self.atol) 
      self.fhb[mol]    = taper(self.bo0[mol],rmin=self.hbtol,rmax=2.0*self.hbtol) 

  def get_atom_energy(self,mol):
      mol_ = mol.split('-')[0]
      val_     = []
      vale_    = []
      valang_  = []
      lp2_     = []
      eatom_   = []
      ovun2_   = []
      ovun5_   = []
      for sp in self.atom_name[mol]:
          eatom_.append([-self.p['atomic_'+sp]])  
          valang_.append([self.p['valang_'+sp]])
          val_.append([self.p['val_'+sp]])
          vale_.append([self.p['vale_'+sp]])
          lp2_.append([self.p['lp2_'+sp]])
          ovun2_.append([self.p['ovun2_'+sp]])
          ovun5_.append([self.p['ovun5_'+sp]])

      val              = tf.stack(val_)
      vale             = tf.stack(vale_)
      valang           = tf.stack(valang_)
      lp2              = tf.stack(lp2_)
      ovun2            = tf.stack(ovun2_)
      ovun5            = tf.stack(ovun5_)
      self.eatom[mol]  = tf.stack(eatom_)
      self.Dang[mol]   = self.Delta[mol] - valang 
 
      self.get_elone(mol,lp2,val,vale) 
      self.elone[mol]  = tf.reduce_sum(input_tensor=self.Elone[mol],axis=0,name='elone_{:s}'.format(mol))
     
      #if 'ovun1' in self.cons:
      #    self.eover[mol] = 0.0
      #else:
      self.get_eover(mol,val,ovun2) 
      self.eover[mol]  = tf.reduce_sum(input_tensor=self.EOV[mol],axis=0,name='eover_{:s}'.format(mol))
      # if 'ovun5' in self.cons:
      #    self.eunder[mol] = 0.0
      # else:
      self.get_eunder(mol,ovun2,ovun5) 
      self.eunder[mol] = tf.reduce_sum(input_tensor=self.EUN[mol],axis=0,name='eunder_{:s}'.format(mol))
      self.zpe[mol]    = tf.reduce_sum(input_tensor=self.eatom[mol],name='zpe') + self.MolEnergy[mol_]

  def get_elone(self,mol,lp2,val,vale):
      Nlp                = 0.5*(vale - val)
      self.Delta_e[mol]  = 0.5*(self.Delta[mol] - vale)
      self.DE[mol]       = -tf.nn.relu(-tf.math.ceil(self.Delta_e[mol])) 
      self.nlp[mol]      = -self.DE[mol] + tf.exp(-self.p['lp1']*4.0*tf.square(1.0+self.Delta_e[mol]-self.DE[mol]))

      self.Delta_lp[mol] = Nlp - self.nlp[mol]                             # nan error
      # Delta_lp         = tf.clip_by_value(self.Delta_lp[mol],-1.0,10.0)  # temporary solution
      Delta_lp           = tf.nn.relu(self.Delta_lp[mol]+1) -1

      explp              = 1.0+tf.exp(-75.0*Delta_lp) # -self.p['lp3']
      self.Elone[mol]    = tf.math.divide(lp2*self.Delta_lp[mol],explp,
                                          name='Elone_{:s}'.format(mol))
                                          
  def get_eover(self,mol,val,ovun2):
      self.Delta_lpcorr[mol] = self.Delta[mol] - val - tf.math.divide(self.Delta_lp[mol],
                                1.0+self.p['ovun3']*tf.exp(self.p['ovun4']*self.Dpi[mol]))

      #self.so[atom]     = tf.gather_nd(self.SO,self.atomlist[atom])
      otrm1              = DIV_IF(1.0,self.Delta_lpcorr[mol]+val)
      # self.otrm2[atom] = tf.math.divide(1.0,1.0+tf.exp(self.p['ovun2_'+atom]*self.Delta_lpcorr[atom]))
      otrm2             = tf.sigmoid(-ovun2*self.Delta_lpcorr[mol])
      self.EOV[mol]     = self.SO[mol] *otrm1*self.Delta_lpcorr[mol]*otrm2

  def get_eunder(self,mol,ovun2,ovun5):
      expeu1            = tf.exp(self.p['ovun6']*self.Delta_lpcorr[mol])
      eu1               = tf.sigmoid(ovun2*self.Delta_lpcorr[mol])
      expeu3            = tf.exp(self.p['ovun8']*self.Dpi[mol])
      eu2               = tf.math.divide(1.0,1.0+self.p['ovun7']*expeu3)
      self.EUN[mol]     = -ovun5*(1.0-expeu1)*eu1*eu2                          # must positive

  def get_threebody_energy(self,mol):
      bo            = tf.concat([tf.zeros([1,self.batch[mol]]),self.bo[mol]],0)
      PBOpow        = tf.negative(tf.pow(bo,8)) # original: self.BO0 
      PBOexp        = tf.exp(PBOpow)
      PBO_          = tf.gather_nd(PBOexp,self.blist[mol],name=mol+'gather_boexp')
      self.Pbo[mol] = tf.reduce_prod(input_tensor=PBO_,axis=1,name=mol+'_pbo') # BO Product

      if self.nang[mol]==0 or self.optword.find('noang')>=0:
         self.eang[mol] = tf.cast(np.zeros([self.batch[mol]]),tf.float32)
         self.epen[mol] = tf.cast(np.zeros([self.batch[mol]]),tf.float32)
         self.tconj[mol]= tf.cast(np.zeros([self.batch[mol]]),tf.float32)
      else:
         self.D_ang[mol] = tf.gather_nd(self.Dang[mol],self.ang_j[mol])

         val,val1,val2,val3,val4,val5,val7,valang,valboc,theta0,pen1,coa1 = self.stack_threebody_parameters(mol)
         Delta= self.D_ang[mol] + valang - val

         self.get_eangle(mol,val1,val2,val3,val4,val5,val7,theta0)
         self.get_epenalty(mol,pen1,Delta)
         self.get_three_conj(mol,valang,valboc,coa1) 
 
         self.eang[mol] = tf.reduce_sum(input_tensor=self.Eang[mol],axis=0,name='eang_%s' %mol)
         self.epen[mol] = tf.reduce_sum(input_tensor=self.Epen[mol],axis=0,name='epen_%s' %mol)
         self.tconj[mol]= tf.reduce_sum(input_tensor=self.Etc[mol],axis=0,name='etc_%s' %mol)
 
  def get_eangle(self,mol,val1,val2,val3,val4,val5,val7,theta0):
      self.BOij[mol] = tf.gather_nd(self.bo[mol],self.abij[mol])   ### need to be done
      self.BOjk[mol] = tf.gather_nd(self.bo[mol],self.abjk[mol])   ### need to be done
      fij            = tf.gather_nd(self.fbot[mol],self.abij[mol]) 
      fjk            = tf.gather_nd(self.fbot[mol],self.abjk[mol]) 
      self.fijk[mol] = fij*fjk

      with tf.compat.v1.name_scope('Theta0_%s' %mol):
           self.get_theta0(mol,theta0)
      self.thet[mol]  = self.theta0[mol]-self.theta[mol]
      self.thet2[mol] = tf.square(self.thet[mol])

      self.expang[mol]= tf.exp(-val2*self.thet2[mol])
      self.f_7[mol]   = self.f7(mol,val3,val4)
      self.f_8[mol]   = self.f8(mol,val5,val7)
      self.Eang[mol]  = self.fijk[mol]*self.f_7[mol]*self.f_8[mol]*(val1-val1*self.expang[mol]) 

  def get_theta0(self,mol,theta0):
      self.sbo[mol] = tf.gather_nd(self.Dpi[mol],self.ang_j[mol])
      self.pbo[mol] = tf.gather_nd(self.Pbo[mol],self.ang_j[mol])
      self.rnlp[mol]= tf.gather_nd(self.nlp[mol],self.ang_j[mol])
      self.SBO[mol] = self.sbo[mol] - tf.multiply(1.0-self.pbo[mol],self.D_ang[mol]+self.p['val8']*self.rnlp[mol])    
      
      ok         = tf.logical_and(tf.less_equal(self.SBO[mol],1.0),tf.greater(self.SBO[mol],0.0))
      S1         = tf.where(ok,self.SBO[mol],tf.zeros_like(self.SBO[mol]))    #  0< sbo < 1                  
      self.SBO01[mol] = tf.where(ok,tf.pow(S1,self.p['val9']),tf.zeros_like(S1)) 

      ok    = tf.logical_and(tf.less(self.SBO[mol],2.0),tf.greater(self.SBO[mol],1.0))
      S2    = tf.where(ok,self.SBO[mol],tf.zeros_like(self.SBO[mol]))                     
      F2    = tf.where(ok,tf.ones_like(S2),tf.zeros_like(S2))                                    #  1< sbo <2
     
      S2    = 2.0*F2-S2  
      self.SBO12[mol] = tf.where(ok,2.0-tf.pow(S2,self.p['val9']),tf.zeros_like(self.SBO[mol]))  #  1< sbo <2
                                                                                                 #     sbo >2
      SBO2  = tf.where(tf.greater_equal(self.SBO[mol],2.0),
                       tf.ones_like(self.SBO[mol]),tf.zeros_like(self.SBO[mol]))

      self.SBO3[mol]   = self.SBO01[mol]+self.SBO12[mol]+2.0*SBO2
      theta0_ = 180.0 - theta0*(1.0-tf.exp(-self.p['val10']*(2.0-self.SBO3[mol])))
      self.theta0[mol] = theta0_/57.29577951

  def f7(self,mol,val3,val4): 
      FBOi  = tf.where(tf.greater(self.BOij[mol],0.0),
                       tf.ones_like(self.BOij[mol]),tf.zeros_like(self.BOij[mol]))   
      FBORi = 1.0 - FBOi                                                                         # prevent NAN error
      expij = tf.exp(-val3*tf.pow(self.BOij[mol]+FBORi,val4)*FBOi)

      FBOk  = tf.where(tf.greater(self.BOjk[mol],0.0),
                        tf.ones_like(self.BOjk[mol]),tf.zeros_like(self.BOjk[mol]))   
      FBORk = 1.0 - FBOk 
      expjk = tf.exp(-val3*tf.pow(self.BOjk[mol]+FBORk,val4)*FBOk)
      fi = 1.0 - expij
      fk = 1.0 - expjk
      F  = tf.multiply(fi,fk,name='f7_'+mol)
      return F 

  def f8(self,mol,val5,val7):
      exp6 = tf.exp( self.p['val6']*self.D_ang[mol])
      exp7 = tf.exp(-val7*self.D_ang[mol])
      F    = val5 - (val5-1.0)*tf.math.divide(2.0+exp6,1.0+exp6+exp7)
      return F

  def get_epenalty(self,mol,pen1,Delta):
      self.f_9[mol] = self.f9(Delta)
      expi = tf.exp(-self.p['pen2']*tf.square(self.BOij[mol]-2.0))
      expk = tf.exp(-self.p['pen2']*tf.square(self.BOjk[mol]-2.0))
      self.Epen[mol] = pen1*self.f_9[mol]*expi*expk*self.fijk[mol]

  def f9(self,Delta):
      exp3 = tf.exp(-self.p['pen3']*Delta)
      exp4 = tf.exp( self.p['pen4']*Delta)
      F = tf.math.divide(2.0+exp3,1.0+exp3+exp4)
      return F

  def get_three_conj(self,mol,valang,valboc,coa1):
      Dcoa = self.D_ang[mol] + valang - valboc
      self.expcoa1[mol] = tf.exp(self.p['coa2']*Dcoa)

      Di    = tf.gather_nd(self.Delta[mol],self.ang_i[mol])
      Dk    = tf.gather_nd(self.Delta[mol],self.ang_k[mol])

      texp0 = tf.math.divide(coa1,1.0+self.expcoa1[mol])  
      texp1 = tf.exp(-self.p['coa3']*tf.square(Di-self.BOij[mol]))
      texp2 = tf.exp(-self.p['coa3']*tf.square(Dk-self.BOjk[mol]))
      texp3 = tf.exp(-self.p['coa4']*tf.square(self.BOij[mol]-1.5))
      texp4 = tf.exp(-self.p['coa4']*tf.square(self.BOjk[mol]-1.5))
      self.Etc[mol] = texp0*texp1*texp2*texp3*texp4*self.fijk[mol] 

  def get_fourbody_energy(self,mol):
      if self.optword.find('notor')>=0 or self.ntor[mol]==0:
         self.etor[mol] = tf.zeros([self.batch[mol]])
         self.efcon[mol]= tf.zeros([self.batch[mol]])
      else:
         tor1,V1,V2,V3,cot1 = self.stack_fourbody_parameters(mol)
         self.get_etorsion(mol,tor1,V1,V2,V3)
         self.get_four_conj(mol,cot1)

         self.etor[mol] = tf.reduce_sum(input_tensor=self.Etor[mol],axis=0,name='etor_%s' %mol)
         self.efcon[mol]= tf.reduce_sum(input_tensor=self.Efcon[mol],axis=0,name='efcon_%s' %mol)

  def get_etorsion(self,mol,tor1,V1,V2,V3):
      self.BOtij[mol]  = tf.gather_nd(self.bo[mol],self.tij[mol])
      self.BOtjk[mol]  = tf.gather_nd(self.bo[mol],self.tjk[mol])
      self.BOtkl[mol]  = tf.gather_nd(self.bo[mol],self.tkl[mol])
      fij              = tf.gather_nd(self.fbot[mol],self.tij[mol])
      fjk              = tf.gather_nd(self.fbot[mol],self.tjk[mol])
      fkl              = tf.gather_nd(self.fbot[mol],self.tkl[mol])
      self.fijkl[mol]  = fij*fjk*fkl

      Dj    = tf.gather_nd(self.Dang[mol],self.tor_j[mol])
      Dk    = tf.gather_nd(self.Dang[mol],self.tor_k[mol])

      self.f_10[mol]   = self.f10(mol)
      self.f_11[mol]   = self.f11(mol,Dj,Dk)

      self.BOpjk[mol]  = tf.gather_nd(self.bopi[mol],self.tjk[mol]) 
      #   different from reaxff manual
      self.expv2[mol]  = tf.exp(tor1*tf.square(2.0-self.BOpjk[mol]-self.f_11[mol])) 

      self.cos3w[mol]  = tf.cos(3.0*self.w[mol])
      v1 = 0.5*V1*(1.0+self.cos_w[mol])   
      v2 = 0.5*V2*self.expv2[mol]*(1.0-self.cos2w[mol])
      v3 = 0.5*V3*(1.0+self.cos3w[mol])
      self.Etor[mol]=self.fijkl[mol]*self.f_10[mol]*self.s_ijk[mol]*self.s_jkl[mol]*(v1+v2+v3)

  def f10(self,mol):
      with tf.compat.v1.name_scope('f10_%s' %mol):
           exp1 = 1.0 - tf.exp(-self.p['tor2']*self.BOtij[mol])
           exp2 = 1.0 - tf.exp(-self.p['tor2']*self.BOtjk[mol])
           exp3 = 1.0 - tf.exp(-self.p['tor2']*self.BOtkl[mol])
      return exp1*exp2*exp3

  def f11(self,mol,Dj,Dk):
      delt = Dj+Dk
      self.f11exp3[mol] = tf.exp(-self.p['tor3']*delt)
      self.f11exp4[mol] = tf.exp( self.p['tor4']*delt)
      f_11 = tf.math.divide(2.0+self.f11exp3[mol],1.0+self.f11exp3[mol]+self.f11exp4[mol])
      return f_11

  def get_four_conj(self,mol,cot1):
      exptol= tf.exp(-self.p['cot2']*tf.square(self.atol - 1.5))
      expij = tf.exp(-self.p['cot2']*tf.square(self.BOtij[mol]-1.5))-exptol
      expjk = tf.exp(-self.p['cot2']*tf.square(self.BOtjk[mol]-1.5))-exptol 
      expkl = tf.exp(-self.p['cot2']*tf.square(self.BOtkl[mol]-1.5))-exptol

      self.f_12[mol] = expij*expjk*expkl
      prod = 1.0+(tf.square(tf.cos(self.w[mol]))-1.0)*self.s_ijk[mol]*self.s_jkl[mol]
      self.Efcon[mol] = self.fijkl[mol]*self.f_12[mol]*cot1*prod  

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

  def get_ev(self,vb,rv,qij):
      [ai,aj] = vb.split('-')
      gm      = tf.sqrt(self.p['gamma_'+ai]*self.p['gamma_'+aj])
      gm3     = tf.pow(tf.math.divide(1.0,gm),3.0)
      r3      = tf.pow(rv,3.0)
      fv      = tf.where(rv>self.vdwcut,tf.zeros_like(rv),tf.ones_like(rv))

      f_13    = self.f13(rv,ai,aj)
      tpv     = self.get_tap(rv)

      expvdw1 = tf.exp(0.5*self.p['alfa_'+vb]*(1.0-tf.math.divide(f_13,2.0*self.p['rvdw_'+vb])))
      expvdw2 = tf.square(expvdw1) 
      Evdw    = fv*tpv*self.p['Devdw_'+vb]*(expvdw2-2.0*expvdw1)

      if self.optword.find('nocoul')<0:
         rth   = tf.pow(r3+gm3,1.0/3.0)
         Ecoul = tf.math.divide(fv*tpv*qij,rth)
      else:
         Ecoul = 0.0
      return Evdw,Ecoul

  def get_vdw_energy(self,mol):
      Evdw,Ecoul = [],[]
      for vb in self.bonds:
          if self.nv[mol][vb]>0:
             with tf.compat.v1.name_scope('vdW_%s' %vb):
                  v_  = self.v[mol][vb]
                  rv_ = tf.slice(self.rv[mol],[v_[0],0],[v_[1],self.batch[mol]])
                  qij_= tf.slice(self.qij[mol],[v_[0],0],[v_[1],self.batch[mol]])
                  Evdw_,Ecoul_ = self.get_ev(vb,rv_,qij_)
                  Evdw.append(Evdw_)
                  Ecoul.append(Ecoul_)
      self.Evdw[mol] = tf.concat(Evdw,0)
      self.evdw[mol] = tf.reduce_sum(input_tensor=self.Evdw[mol],axis=0,name='evdw_%s' %mol)

      if self.optword.find('nocoul')<0:
         self.Ecoul[mol] = tf.concat(Ecoul,0)
         self.ecoul[mol]= tf.reduce_sum(input_tensor=self.Ecou[mol],axis=0,name='ecoul_%s' %mol)
      else:
         self.ecoul[mol]= tf.constant(self.ecoul_[mol],dtype=tf.float32)

  def get_hb_energy(self,mol):
      Ehb = []
      for hb in self.hbs:
          if self.nh[mol][hb]>0:
             h_  = self.h[mol][hb]
             with tf.compat.v1.name_scope('ehb_%s' %mol):
                  rhb   = tf.slice(self.rhb[mol],[h_[0],0],[h_[1],self.batch[mol]])
                  hbthe = tf.slice(self.hbthe[mol],[h_[0],0],[h_[1],self.batch[mol]])
                  frhb  = tf.slice(self.frhb[mol],[h_[0],0],[h_[1],self.batch[mol]])
                  Ehb_  = self.get_ehb(mol,hb,rhb,hbthe,frhb)
                  Ehb.append(Ehb_)
      if len(Ehb)>0:
         self.Ehb[mol] = tf.concat(Ehb,0,name='Ehb_'+mol)
         self.ehb[mol] = tf.reduce_sum(input_tensor=self.Ehb[mol],axis=0,name='ehb_%s' %mol)
      else: 
         self.ehb[mol] = 0.0 # case for no hydrogen-bonds in system

  def get_ehb(self,mol,hb,rhb,hbthe,frhb):
      ''' compute hydrogen bond energy '''
      bohb   = tf.gather_nd(self.bo0[mol],self.hij[mol][hb]) 
      fhb_   = tf.gather_nd(self.fhb[mol],self.hij[mol][hb]) 
      exphb1 = 1.0-tf.exp(-self.p['hb1_'+hb]*bohb)
      sum_   = tf.math.divide(self.p['rohb_'+hb],rhb)+tf.math.divide(rhb,self.p['rohb_'+hb])-2.0
      exphb2 = tf.exp(-self.p['hb2_'+hb]*sum_)
      # self.sin4[hb] = tf.pow(tf.sin(self.hbthe[hb]*0.5),4.0) 
      sin4   = tf.square(hbthe)
      Ehb    = fhb_*frhb*self.p['Dehb_'+hb]*exphb1*exphb2*sin4
      return Ehb

  def set_zpe(self,molecules=None):
      if self.MolEnergy_ is None:
         self.MolEnergy_ = {}

      for mol in self.mols:
          mols = mol.split('-')[0] 
          if mols not in self.MolEnergy:
             if mols in self.MolEnergy_:
                if self.optmol:
                   self.MolEnergy[mols] = tf.Variable(self.MolEnergy_[mols],name='Molecule-Energy_'+mols)
                else:
                   self.MolEnergy[mols] = tf.constant(self.MolEnergy_[mols])
             else:
                if self.optmol:
                   self.MolEnergy[mols] = tf.Variable(0.0,name='Molecule-Energy_'+mols)
                else:
                   self.MolEnergy[mols] = tf.constant(0.0)

  def get_loss(self):
      ''' return the losses of the model '''
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
             self.loss[mol] =  (-1.0/self.batch[mol])*tf.reduce_sum(y*tf.math.log(y_)+(1-y)*tf.math.log(1.0-y_))
          else:
             raise NotImplementedError('-  This function not supported yet!')

          sum_edft = tf.reduce_sum(input_tensor=tf.abs(self.dft_energy[mol]-self.max_e[mol]))
          self.accur[mol] = 1.0 - tf.reduce_sum(input_tensor=tf.abs(self.E[mol]-self.dft_energy[mol]))/(sum_edft+0.00000001)
         
          self.Loss     += self.loss[mol]*w_
          if mol.find('nomb_')<0:
             self.accuracy += self.accur[mol]
          else:
             self.nmol -= 1

      self.ME   = 0.0
      for mol in self.mols:
          mol_     = mol.split('-')[0] 
          self.ME += tf.square(self.MolEnergy[mol_])

      self.loss_penalty = self.supervise()
      self.Loss        += self.loss_penalty

      if self.optmol:
         self.Loss  += self.ME*self.lambda_me
      self.accuracy  = self.accuracy/self.nmol

  def set_variable_list(self):
      self.unit = 4.3364432032e-2
      self.p_g  = ['boc1','boc2','coa2','ovun6','lp1',#'lp3',
                   'ovun7','ovun8','val6','tor2',
                   'tor3','tor4','cot2','coa4','ovun4',               # 
                   'ovun3','val8','val9','val10',
                   'coa3','pen2','pen3','pen4','vdw1',
                   'cutoff','hbtol','acut'] # # 
                   # 'trip2','trip1','trip4','trip3' ,'swa','swb'
                   # tor3,tor4>0 

      self.p_spec = ['valang','val','valboc','vale','ovun5',
                     'lp2','boc4','boc3','boc5','rosi','ropi','ropp',
                     'ovun2','val3','val5','atomic',
                     'gammaw','gamma','mass','chi','mu',
                     'Devdw','rvdw','alfa'] # ,'chi','mu'

      self.p_bond = ['Desi','Depi','Depp','bo5','bo6','ovun1',
                     'be1','be2','bo3','bo4','bo1','bo2','corr13','ovcorr']

      self.p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp'] # 
      self.p_ang  = ['theta0','val1','val2','coa1','val7','val4','pen1'] # 
      self.p_hb   = ['rohb','Dehb','hb1','hb2']
      self.p_tor  = ['V1','V2','V3','tor1','cot1'] # 'tor2','tor3','tor4',

      self.punit  = ['Desi','Depi','Depp','lp2','ovun5','val1',
                     'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb'] # ,'hb1'

      cons = ['mass','corr13','ovcorr', 
              'trip1','trip2','trip3','trip4','swa','swb',
              #'val', 'valboc','valang','vale',
              'gamma','chi','mu']  

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
                   'rohb','Dehb','hb1','hb2','atomic']  

      if self.optword.find('noover')>=0:
         cons = cons + ['ovun1' ,'ovun2','ovun3','ovun4'] #
      if self.optword.find('nounder')>=0:
         cons = cons + ['ovun5','ovun6','ovun7','ovun8'] 
      # if self.optword.find('noover')>=0 and self.optword.find('nounder')>=0:
      #    cons = cons + ['ovun2','ovun3','ovun4'] 
      if self.optword.find('nolone')>=0:
         cons = cons + ['lp2','lp3', 'lp1'] #
      if self.optword.find('novdw')>=0:
         cons = cons + ['gammaw','vdw1','rvdw','Devdw','alfa'] 
      if self.optword.find('nohb')>=0:
         cons = cons + ['Dehb','rohb','hb1','hb2','hbtol'] 

      self.tor_v = ['tor2','tor3','tor4','V1','V2','V3','tor1','cot1','cot2'] 

      if self.optword.find('notor')>=0:
         cons = cons + self.tor_v
      self.ang_v = ['theta0',
                    'val1','val2','val3','val4','val5','val6','val7',
                    'pen1','pen2','pen3','pen4',
                    'coa1','coa2','coa3','coa4'] 
      if self.optword.find('noang')>=0:
         cons = cons + self.ang_v

      if self.cons is None:
         self.cons = cons 
      else:
         self.cons += cons

      self.cons += ['boc1','boc2','boc3','boc4','boc5'] # 'valboc'

      if self.opt is None:
         self.opt = self.p_g+self.p_spec+self.p_bond+self.p_offd+self.p_ang+self.p_tor+self.p_hb
      
      self.nvopt = self.p_g+self.p_spec+self.p_bond+self.p_offd+self.p_ang+self.p_tor+self.p_hb
      for v in ['gammaw','vdw1','rvdw','Devdw','alfa','gamma']:
          self.nvopt.remove(v)
      

  def set_parameters_to_opt(self,libfile=None):
      if not libfile is None:
         self.p_,zpe,spec,bonds,offd,angs,torp,hbs = read_ffield(libfile=libfile)

      self.p,self.var = {},{}
      for k in self.p_:
          key = k.split('_')[0]
          ktor= ['cot1','V1','V2','V3']

          if self.optword.find('notor')>=0:
             if key in ktor:
                self.p_[k] = 0.0
          if self.optword.find('nolone')>=0:
             if key in 'lp2':
                self.p_[k] = 0.0
          if self.optword.find('noover')>=0:
             if key in 'ovun1':
                self.p_[k] = 0.0
          if self.optword.find('nounder')>=0:
             if key in 'ovun5':
                self.p_[k] = 0.0
          if self.optword.find('noang')>=0:
             if key in ['val1','coa1','pen1']:
                self.p_[k] = 0.0

          if key == 'zpe':
             continue
          if key != 'n.u.':
             if (k in self.VariablesToOpt) and (key in self.opt) and (key not in self.cons):
                if key in self.punit:
                   self.var[k] = tf.Variable(np.float32(self.unit*self.p_[k]),name=k)
                else:
                   self.var[k] = tf.Variable(np.float32(self.p_[k]),name=k)
             else:
                if key in self.punit:
                   self.var[k] = tf.constant(np.float32(self.unit*self.p_[k]),name=k)
                else:
                   self.var[k] = tf.constant(np.float32(self.p_[k]),name=k)

      if self.clip_op:
         self.p = clip_parameters(self.p_,self.var,self.clip)
      else:
         for k in self.var:
             key       = k.split('_')[0]
             self.p[k] = self.var[k]
             
      self.botol       = 0.01*self.p['cutoff']
      self.checkp()
      self.get_rcbo()
      self.m = set_matrix(self.m_,self.spec,self.bonds,
                          self.mfopt,self.mpopt,self.bdopt,self.messages,
                          self.bo_layer,self.bo_layer_,0,0,
                          self.mf_layer,self.mf_layer_,3,3,
                          self.be_layer,self.be_layer_,1,1,
                          (9,0),(9,0),1,1,
                          None,self.be_universal_nn,self.mf_universal_nn,None)

  def set_parameters(self,libfile=None):
      if not libfile is None:
         self.p_,zpe,spec,bonds,offd,angs,torp,hbs = read_ffield(libfile=libfile)
      self.var = set_variables(self.p_, self.optword, self.cons, self.opt,self.eaopt,
                               self.punit, self.unit, self.conf_vale,
                               self.ang_v,self.tor_v)

      self.ea_var = {}        # parameter list to be optimized with evolutional algrithom
      for k in self.var:
            key = k.split('_')[0]
            if key in self.eaopt:
               self.ea_var[k] = self.p_[k]

      if self.clip_op:
         self.p = clip_parameters(self.p_,self.var,self.clip)
      else:
         self.p = {}
         for k in self.var:
             key       = k.split('_')[0]
             self.p[k] = self.var[k]

      self.botol       = 0.01*self.p['cutoff']
      self.atol        = self.p['acut']
      self.hbtol       = self.p['hbtol']
      self.checkp()
      self.get_rcbo()
      self.m = set_matrix(self.m_,self.spec,self.bonds,
                          self.mfopt,self.mpopt,self.bdopt,self.messages,
                          (6,0),(6,0),0,0,
                          self.mf_layer,self.mf_layer_,3,3,
                          self.be_layer,self.be_layer_,1,1,
                          (9,0),(9,0),1,1,
                          None,self.be_universal_nn,self.mf_universal_nn,None)

  def stack_threebody_parameters(self,mol):
      val_,val1_,val2_,val3_,val4_,val5_,val7_= [],[],[],[],[],[],[]
      valboc_,valang_,theta0_,pen1_,coa1_ = [],[],[],[],[]
      for i in range(self.nang[mol]):
          ang = (self.atom_name[mol][self.ang_i[mol][i][0]] + '-' + 
                  self.atom_name[mol][self.ang_j[mol][i][0]] + '-' + 
                  self.atom_name[mol][self.ang_k[mol][i][0]])
          aj  = self.atom_name[mol][self.ang_j[mol][i][0]]
          val_.append([self.p['val_'+aj]])
          valang_.append([self.p['valang_'+aj]])
          valboc_.append([self.p['valboc_'+aj]])
          val1_.append([self.p['val1_'+ang]])
          val2_.append([self.p['val2_'+ang]]) 
          val3_.append([self.p['val3_'+aj]])
          val4_.append([self.p['val4_'+ang]]) 
          val5_.append([self.p['val5_'+aj]])
          val7_.append([self.p['val7_'+ang]]) 
          pen1_.append([self.p['pen1_'+ang]]) 
          coa1_.append([self.p['coa1_'+ang]])
          theta0_.append([self.p['theta0_'+ang]])
      val  = tf.stack(val_)
      val1 = tf.stack(val1_)
      val2 = tf.stack(val2_)
      val3 = tf.stack(val3_)
      val4 = tf.stack(val4_)
      val5 = tf.stack(val5_)
      val7 = tf.stack(val7_)
      valang = tf.stack(valang_)
      valboc = tf.stack(valboc_)
      theta0 = tf.stack(theta0_)
      pen1 = tf.stack(pen1_)
      coa1 = tf.stack(coa1_)
      return val,val1,val2,val3,val4,val5,val7,valang,valboc,theta0,pen1,coa1

  def stack_fourbody_parameters(self,mol):
      tor1_,V1_,V2_,V3_,cot1_ = [],[],[],[],[]
      for i in range(self.ntor[mol]):
          tor = find_torsion_angle(self.atom_name[mol][self.tor_i[mol][i][0]],
                                   self.atom_name[mol][self.tor_j[mol][i][0]],
                                   self.atom_name[mol][self.tor_k[mol][i][0]], 
                                   self.atom_name[mol][self.tor_l[mol][i][0]],self.tors)
          tor1_.append([self.p['tor1_'+tor]])
          V1_.append([self.p['V1_'+tor]])
          V2_.append([self.p['V2_'+tor]])
          V3_.append([self.p['V3_'+tor]])
          cot1_.append([self.p['cot1_'+tor]])
      tor1 = tf.stack(tor1_)
      V1   = tf.stack(V1_)
      V2   = tf.stack(V2_)
      V3   = tf.stack(V3_)
      cot1 = tf.stack(cot1_)
      return tor1,V1,V2,V3,cot1

  def checkp(self):
      for key in self.p_offd:
          for sp in self.spec:
              try:
                 self.p[key+'_'+sp+'-'+sp]  = self.p[key+'_'+sp]  
              except KeyError:
                 print('-  warning: key not in dict') 

      self.tors = []
      fm = open('manybody.log','w')
      print('  The following manybody interaction are not considered, because no parameter in the ffield: ',file=fm)
      print('---------------------------------------------------------------------------------------------',file=fm)
      for spi in self.spec:
          for spj in self.spec:
              for spk in self.spec:
                  ang = spi+'-'+spj+'-'+spk 
                  angr= spk+'-'+spj+'-'+spi
                  if (ang not in self.angs) and (angr not in self.angs):
                     print('                 three-body      {:20s} '.format(ang),file=fm)
                  for spl in self.spec:
                      tor = spi+'-'+spj+'-'+spk+'-'+spl
                      torr= spl+'-'+spk+'-'+spj+'-'+spi
                      tor1= spi+'-'+spk+'-'+spj+'-'+spl
                      tor2= spl+'-'+spj+'-'+spk+'-'+spi
                      tor3= 'X-'+spj+'-'+spk+'-X'
                      tor4= 'X-'+spk+'-'+spj+'-X'
                      if (tor in self.torp) or (torr in self.torp) or (tor1 in self.torp) \
                           or (tor2 in self.torp) or (tor3 in self.torp) or (tor4 in self.torp):
                         if (not tor in self.tors) and (not torr in self.tors):
                            if tor in self.torp:
                               self.tors.append(tor)
                            elif torr in self.torp:
                               self.tors.append(torr)
                            else:
                               self.tors.append(tor)
                      else:
                         print('                 four-body      {:20s} '.format(tor),file=fm)
      
      for key in self.p_tor:
          for tor in self.tors:
              if tor not in self.torp:                 # totally have six variable name share the same value
                 [t1,t2,t3,t4] = tor.split('-')
                 tor1 = t1+'-'+t3+'-'+t2+'-'+t4
                 tor2 = t4+'-'+t3+'-'+t2+'-'+t1
                 tor3 = t4+'-'+t2+'-'+t3+'-'+t1 
                 tor4 = 'X'+'-'+t2+'-'+t3+'-'+'X'
                 tor5 = 'X'+'-'+t3+'-'+t2+'-'+'X'
                 if tor1 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor1]
                 elif tor2 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor2]
                 elif tor3 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor3]    
                 elif tor4 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor4]   
                 elif tor5 in self.torp:
                    self.p[key+'_'+tor] = self.p[key+'_'+tor5]   
                 else:
                    print('-  an error case for {:s},'.format(tor),self.spec,file=fm)
      fm.close()

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
         # elif method=='NadagradOptimizer':
         #    optimizer = tf.compat.v1.train.NadagradOptimizer(learning_rate) 
         else:
            raise RuntimeError('-  This method not implimented!')
            
         if self.to_train: # and self.optMethod=='ADAM':
            self.train_step = optimizer.minimize(self.Loss)

      self.sess.run(tf.compat.v1.global_variables_initializer())  
      self.sess_build = True

  def get_zpe(self):
      e,edft = self.sess.run([self.E,self.dft_energy],feed_dict=self.feed_dict)                                           
      for mol in e:
          mol_ = mol.split('-')[0]
          self.MolEnergy_[mol_] = np.mean(edft[mol]) - np.mean(e[mol]) 
      return self.MolEnergy_

  def update(self,p=None,reset_emol=False):
      # print('-  updating variables ...')
      upop = []
      for key in self.var:
          k_ = key.split('_')
          k  = k_[0]
          hasH = False
          if len(k_)>1:
             kk  = k_[1]
             kkk = kk.split('-')
             if len(kkk)>2:
                if kkk[1]=='H' and (k not in ['rohb','Dehb','hb1','hb2']):
                   hasH=True
             if len(kkk)>3:
                if kkk[2]=='H':
                   hasH=True
          if p is not None:
             if (k in self.opt or key in self.opt) and (key not in self.cons and k not in self.cons):
             #   p_ = self.p_[key]*self.unit if k in self.punit else self.p_[key]
             #   upop.append(tf.compat.v1.assign(self.var[key],p_))
             # else:
                if key in p:
                   p_ = p[key]*self.unit if k in self.punit else p[key]
                   if not hasH: upop.append(tf.compat.v1.assign(self.var[key],p_))
             elif key in self.ea_var:
                if key in p:
                   p_ = p[key]*self.unit if k in self.punit else p[key]
                   self.feed_dict[self.var[key]] = p_
                   self.ea_var[key]              = p[key]
      
      for mol in self.mols:
          mol_ = mol.split('-')[0]
          if reset_emol: self.MolEnergy_[mol_] = 0.0
          emol_ = 0.0 if mol_ not in self.MolEnergy_  else self.MolEnergy_[mol_]
          if self.optmol:
             upop.append(tf.compat.v1.assign(self.MolEnergy[mol_],emol_))     
          else:
             self.MolEnergy[mol_] =  tf.constant(emol_)
      if upop: self.sess.run(upop)

  def reset(self,opt=[],libfile=None):
      if self.InitCheck:
         self.p_,self.m_ = self.ic.check(self.p_)

      if self.VariablesToOpt is None:
         self.set_parameters(libfile=libfile)
      else:
         self.set_parameters_to_opt()

      self.memory()
      self.set_zpe()
         
      self.build_graph() 
      self.feed_dict = self.feed_data()   

  def run(self,learning_rate=1.0e-4,method='AdamOptimizer',
               step=2000,print_step=10,writelib=100,
               close_session=True,saveffield=True):
      if not self.initialized:
         self.initialize()
      if not self.sess_build:
         self.session(learning_rate=learning_rate,method=method)  

      libfile = self.libfile.split('.')[0]

      for i in range(step+1):
          if i==0:
             loss,lpenalty,self.ME_,accu,accs   = self.sess.run([self.Loss,
                                                      self.loss_penalty,
                                                      self.ME,self.accuracy, self.accur],
                                                  feed_dict=self.feed_dict)
          else:
             loss,lpenalty,self.ME_,accu,accs,_ = self.sess.run([self.Loss,
                                                      self.loss_penalty,
                                                      self.ME,
                                                      self.accuracy,
                                                      self.accur,
                                                      self.train_step],
                                                  feed_dict=self.feed_dict)
          if i==0:
             accMax = accu
          else:
             if accu>accMax:
                accMax = accu

          if np.isnan(loss):
             if close_session:
                self.logger.info('NAN error encountered at step %d loss is %f.' %(i,loss/self.nframe))
                loss_ = 99999999999.9 
                accu  = -1.0
                zpe   = {}
                break
                # return 9999999.9,-999.0,accMax,i,0.0
             else:
                break

          if self.optmol:
             los_ = loss - lpenalty - self.ME_*self.lambda_me
          else:
             los_ = loss - lpenalty
          loss_ = los_ if i==0 else min(loss_,los_)

          if i%print_step==0:
             current = time.time()
             elapsed_time = current - self.time

             acc = ''
             for key in accs:
                 acc += key+': %6.4f ' %accs[key]

             self.logger.info('-  step: %d loss: %6.4f accs: %f %s spv: %6.4f me: %6.4f time: %6.4f' %(i,
                              los_,accu,acc,lpenalty,self.ME_,elapsed_time))
             self.time = current

          if i%writelib==0 or i==step:
             self.lib_bk = libfile+'_'+str(i)
             self.write_lib(libfile=self.lib_bk,loss=loss_)

             if i==step:
                if saveffield: self.write_lib(libfile=libfile,loss=loss_)
                # E,dfte,zpe = self.sess.run([self.E,self.dft_energy,self.zpe],
                #                           feed_dict=self.feed_dict)
                # self.plot_result(i,E,dfte)
             if not close_session:
                if i<=200:
                   _loss = loss_
                else:
                   if loss_>=_loss:
                      self.logger.info('-  No other minimum found, optimization compeleted.')
                      break
                   else:
                      _loss = loss_

          if accu>=self.convergence and loss_<=self.lossConvergence:
             self.accu = accu
             E,dfte,zpe = self.sess.run([self.E,self.dft_energy,self.zpe],
                                        feed_dict=self.feed_dict)
             self.plot_result(None,E,dfte)
             self.write_lib(libfile=libfile,loss=loss_)
             print('-  Convergence Occurred, job compeleted.')
             break
      
      self.get_pentalty()
      self.loss_ = loss_ if not (np.isnan(loss) or np.isinf(loss)) else 9999999.9
      if self.loss_ < 9999999.0: self.write_lib(libfile=libfile,loss=loss_)
      if close_session:
         tf.compat.v1.reset_default_graph()
         self.sess.close()
         return loss_,accu,accMax,i

  def feed_data(self):
      feed_dict = {}
      for mol in self.mols:
          feed_dict[self.dft_energy[mol]] = self.data[mol].dft_energy
          feed_dict[self.rbd[mol]] = self.data[mol].rbd
          feed_dict[self.rv[mol]]  = self.data[mol].rv
          # if self.optword.find('nocoul')<0:
          #    feed_dict[self.qij[mol]] = self.data[mol].qij
          if self.nang[mol]>0:
             feed_dict[self.theta[mol]] = self.data[mol].theta

          if self.ntor[mol]>0:
             feed_dict[self.s_ijk[mol]] = self.data[mol].s_ijk
             feed_dict[self.s_jkl[mol]] = self.data[mol].s_jkl
             feed_dict[self.w[mol]]     = self.data[mol].w

          if self.nhb[mol]>0:
             feed_dict[self.rhb[mol]]   = self.data[mol].rhb
             feed_dict[self.frhb[mol]]  = self.data[mol].frhb
             feed_dict[self.hbthe[mol]] = self.data[mol].hbthe
      for k in self.ea_var:
          key = k.split('_')[0]
          p_  = self.p_[k]*self.unit if key in self.punit else self.p_[k]
          feed_dict[self.var[k]] = p_
      return feed_dict

  def calculate_energy(self):
      energy = self.get_value(self.E)
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
          if k in self.ea_var:
             self.p_[k] = self.ea_var[k]
          else:
             if key in self.punit:
                self.p_[k] = float(p_[k]/self.unit)
             else:
                self.p_[k] = float(p_[k])

      score = loss if loss is None else -loss
         
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
              'score':score,
              'BOFunction':0,#self.BOFunction,
              'EnergyFunction':1,# self.EnergyFunction,
              'MessageFunction': self.MessageFunction, 
              'VdwFunction':1,#self.VdwFunction,
              'messages':self.messages,
              'bo_layer':self.bo_layer,
              'mf_layer':self.mf_layer,
              'be_layer':self.be_layer,
              'vdw_layer':None,#self.vdw_layer,
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

  def read_lib(self):
      if self.libfile.endswith('.json'):
         with open(self.libfile,'r') as lf:
              j = js.load(lf)
         self.p_  = j['p']
         self.m_  = j['m']
         # self.BOFunction_      = j['BOFunction']
         # self.EnergyFunction_  = j['EnergyFunction'] 
         self.MessageFunction_ = j['MessageFunction']
         # self.VdwFunction_     = j['VdwFunction']
         self.MolEnergy_         = j['MolEnergy']
         # self.bo_layer_        = j['bo_layer']
         self.mf_layer_          = j['mf_layer']
         self.be_layer_          = j['be_layer']
         # self.vdw_layer_       = j['vdw_layer']
         rcut                    = j['rcut']
         rcuta                   = j['rcutBond']
         re                      = j['rEquilibrium']
         self.init_bonds()
      else:
         (self.p_,self.zpe_,self.spec,self.bonds,self.offd,self.angs,self.torp,
          self.hbs) = read_ffield(libfile=self.libfile,zpe=True)
         self.MolEnergy_ = {}
         rcut,rcuta,re = None,None,None
      return rcut,rcuta,re

  def supervise(self):
      ''' adding some penalty term to accelerate the training '''
      log_    = -9.21044036697651
      penalty = 0.0
      wb_p    = []
      if self.regularize_be:
         wb_p.append('fe')
      # if self.vdwnn and self.regularize_vdw:
      #    wb_p.append('fv')
      w_n     = ['wi','wo',]
      b_n     = ['bi','bo']
      layer   = {'fe':self.be_layer[1]}
      if self.bo_layer is not None:
         layer['fsi'] = layer['fpi'] = layer['fpp'] = self.bo_layer[1]

      wb_message = []
      if self.regularize_mf:
         for t in range(1,self.messages+1):
             wb_message.append('fm')          
             layer['fm'] = self.mf_layer[1]  

      self.penalty_bop     = {}
      self.penalty_bo      = {}
      self.penalty_bo_rcut = {}
      self.penalty_be_cut  = {}
      self.penalty_rcut    = {}
      self.penalty_ang     = {}
      self.penalty_w       = tf.constant(0.0)
      self.penalty_b       = tf.constant(0.0)

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
 
          self.penalty_bop[bd]     = tf.constant(0.0)
          self.penalty_be_cut[bd]  = tf.constant(0.0)
          self.penalty_bo_rcut[bd] = tf.constant(0.0)
          self.penalty_bo[bd]      = tf.constant(0.0)

          for mol in self.mols:
              if self.nbd[mol][bd]>0:       
                 b_   = self.b[mol][bd]
                 #rbd_= tf.slice(self.rbd[mol],[b_[0],0],[b_[1],self.batch[mol]])        
                 bop_ = tf.slice(self.bop[mol],[b_[0],0],[b_[1],self.batch[mol]]) 
                 bo0_ = tf.slice(self.bo0[mol],[b_[0],0],[b_[1],self.batch[mol]]) 

                 fbo  = tf.where(tf.less(self.rbd_[mol][bd],self.rc_bo[bd]),0.0,1.0)     # bop should be zero if r>rcut_bo
                 self.penalty_bop[bd]  +=  tf.reduce_sum(bop_*fbo)                       #####  

                 fao  = tf.where(tf.greater(self.rbd_[mol][bd],self.rcuta[bd]),1.0,0.0)  ##### r> rcuta that bo = 0.0
                 self.penalty_bo_rcut[bd] += tf.reduce_sum(bo0_*fao)

                 fesi = tf.where(tf.less_equal(bo0_,self.botol),1.0,0.0)                 ##### bo <= 0.0 that e = 0.0
                 self.penalty_be_cut[bd]  += tf.reduce_sum(tf.nn.relu(self.esi[mol][bd]*fesi))
                 
                 if self.spv_bo:
                     if (bd in self.spv_bo) or (bdr in self.spv_bo):
                        bd_  = bd if bd in self.spv_bo else bdr
                     for sbo in self.spv_bo[bd_]:
                         r,d_i,d_j,bo_l,bo_u = sbo
                         fe   = tf.where(tf.logical_and(tf.less_equal(self.rbd[bd],r),
                                                         tf.logical_and(tf.greater_equal(self.Dbi[bd],d_i),
                                                                        tf.greater_equal(self.Dbj[bd],d_j))),
                                          1.0,0.0)   ##### r< r_e that bo > bore_
                         self.penalty_bo[bd] += tf.reduce_sum(input_tensor=tf.nn.relu((bo_l-self.bo0[bd])*fe))
                         fe   = tf.where(tf.logical_and(tf.greater_equal(self.rbd[bd],r),
                                                         tf.logical_and(tf.greater_equal(self.Dbi[bd],d_i),
                                                                        tf.greater_equal(self.Dbj[bd],d_j))),
                                          1.0,0.0)  ##### r> r_e that bo < bore_
                         self.penalty_bo[bd] += tf.reduce_sum(input_tensor=tf.nn.relu((self.bo0[bd]-bo_u)*fe))

              if self.spv_ang:
                 self.penalty_ang[mol] = tf.reduce_sum(self.thet2[mol]*self.fijk[mol])
          
          penalty  = tf.add(self.penalty_be_cut[bd]*self.lambda_bd,penalty)
          penalty  = tf.add(self.penalty_bop[bd]*self.lambda_bd,penalty)        
          penalty  = tf.add(self.penalty_bo_rcut[bd]*self.lambda_bd,penalty)
          penalty  = tf.add(self.penalty_bo[bd]*self.lambda_bd,penalty)   

          # penalize term for regularization of the neural networs
          if self.regularize:                             # regularize to avoid overfit
             for k in wb_p:
                 for k_ in w_n:
                     key     = k + k_ + '_' + bd
                     self.penalty_w  += tf.reduce_sum(tf.square(self.m[key]))
                 if self.regularize_bias:
                    for k_ in b_n:
                        key     = k + k_ + '_' + bd
                        self.penalty_b  += tf.reduce_sum(tf.square(self.m[key]))
                 for l in range(layer[k]):                                               
                     self.penalty_w += tf.reduce_sum(tf.square(self.m[k+'w_'+bd][l]))
                     if self.regularize_bias:
                        self.penalty_b += tf.reduce_sum(tf.square(self.m[k+'b_'+bd][l]))

      if self.regularize:                              # regularize
         for sp in self.spec:
             for k in wb_message:
                 for k_ in w_n:
                     key     = k + k_ + '_' + sp
                     self.penalty_w  += tf.reduce_sum(tf.square(self.m[key]))
                 if self.regularize_bias:
                    for k_ in b_n:
                        key     = k + k_ + '_' + sp
                        self.penalty_b  += tf.reduce_sum(tf.square(self.m[key]))
                 for l in range(layer[k]):                                               
                     self.penalty_w += tf.reduce_sum(tf.square(self.m[k+'w_'+sp][l]))
                     if self.regularize_bias:
                        self.penalty_b += tf.reduce_sum(tf.square(self.m[k+'b_'+sp][l]))
         penalty = tf.add(self.lambda_reg*self.penalty_w,penalty)
         penalty = tf.add(self.lambda_reg*self.penalty_b,penalty)
      return penalty

  def get_pentalty(self):
      (penalty_bop,penalty_bo_rcut,
          penalty_bo,penalty_be_cut,
          penalty_rcut,rc_bo,
          penalty_w,penalty_b) = self.sess.run([self.penalty_bop,self.penalty_bo_rcut,
                                         self.penalty_bo,self.penalty_be_cut,
                                         self.penalty_rcut,self.rc_bo,
                                         self.penalty_w,self.penalty_b],
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
             print('Differency between rcut-bo and rcut Penalty of    {:5s}: {:6.4f} {:6.4f} {:6.4f}'.format(bd,penalty_bo_rcut[bd],rc_bo[bd],rcut[bd]))
          # if bd in penalty_esi:
          #    print('Differency between bosi and esi Penalty of      {:5s}: {:6.4f}'.format(bd,penalty_esi[bd]))
          if bd in penalty_be_cut: 
             print('Bond-Energy at radius cutoff penalty of           {:5s}: {:6.4f}'.format(bd,penalty_be_cut[bd]))
          if bd in penalty_rcut:
             print('Bond-Order at radius cutoff penalty of            {:5s}: {:6.4f}'.format(bd,penalty_rcut[bd]))
      print('Sum of square of weight:                          {:6.4f}'.format(penalty_w))
      print('Sum of square of bias:                            {:6.4f}'.format(penalty_b))
      print('\n')
     
      # print('\n------------------------------------------------------------------------')
      # print('-                 -  Energy Components Information  -                  -')
      # print('------------------------------------------------------------------------\n')
      # for mol in self.mols:
      #     print('Max Bond-Order of {:s} {:f}'.format(mol,np.max(bo[mol])))
      #     print('Max Bond Energy of {:s} {:f}'.format(mol,max(ebond[mol])))
      #     print('Max Angle Energy of {:s} {:f}'.format(mol,max(eang[mol])))
          

  def plot_result(self,step,E,dfte):
      if not exists('results'):
         makedirs('results')
      Y,Yp = [],[]
      
      for mol in self.mols:
          maxe = self.max_e[mol]
          x  = np.linspace(0,self.batch[mol],self.batch[mol])
          plt.figure()
          plt.ylabel('Energies comparation between DFT and ReaxFF-nn')
          plt.xlabel('Step')
          #err  = dfte[mol] - E[mol]
          Y.extend(dfte[mol]-maxe)
          Yp.extend(E[mol]-maxe)

          plt.plot(x,dfte[mol]-maxe,linestyle='-',marker='o',markerfacecolor='snow',
                   markeredgewidth=1,markeredgecolor='k',
                   ms=5,c='k',alpha=0.8,label=r'$DFT$')
          plt.plot(E[mol]-maxe,linestyle='-',marker='^',markerfacecolor='snow',
                   markeredgewidth=1,markeredgecolor='b',
                   ms=5,c='b',alpha=0.8,label=r'$ReaxFF-nn$')
          # plt.errorbar(x,E[mol]-maxe,yerr=err,
          #              fmt='-s',ecolor='r',color='r',ms=4,markerfacecolor='none',mec='blue',
          #              elinewidth=2,capsize=2,label='I-ReaxFF')

          plt.legend(loc='best',edgecolor='yellowgreen')
          if step is None:
             plt.savefig('results/result_%s.pdf' %mol) 
          else:
             plt.savefig('results/result_%s_%s.pdf' %(mol,step)) # transparent=True
          plt.close()

      plt.figure()
      plt.xlabel('E(DFT)')
      plt.ylabel('E(ReaxFF-nn)')
      plt.scatter(Y,Yp,
                  marker='o',color='none',edgecolor='r',s=20,
                  alpha=0.8,label=r'$E(DFT) V.S. E(ReaxFF-nn)$')
      plt.savefig('results/Result.svg')
      plt.close()

      with open('results/Results.csv','w') as fcsv:
           print('Edft,Epred',file=fcsv)
           for y,yp in zip(Y,Yp):
               print(y,yp,sep=',',file=fcsv)

   
