import matplotlib.pyplot as plt
from os import makedirs
from os.path import exists
import time
import tensorflow as tf
import numpy as np
import json as js
#from .reax_data import reax_data,Dataset
from .reax_force_data import reax_force_data,Dataset
from .reaxfflib import read_ffield,write_ffield,write_lib
from .intCheck import Intelligent_Check
from .RadiusCutOff import setRcut
from .reax import logger,taper,rtaper,DIV_IF,clip_parameters,set_variables
from .mpnn import fmessage,fnn,set_matrix
# tf_upgrade_v2 --infile reax.py --outfile reax_v1.py
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

def find_torsion_angle(atomi,atomj,atomk,atoml,tors):
    tor1 = atomi+'-'+atomj+'-'+atomk+'-'+atoml
    tor2 = atoml+'-'+atomk+'-'+atomj+'-'+atomi
    if tor1 in tors:
       return tor1
    elif tor2 in tors:
       return tor2  
    else:
       raise RuntimeError('-  Torsion angle {:s} not find in ffield!'.format(tor1))

def fvr(x):
    xi  =  tf.expand_dims(x, axis=1)  
    xj  =  tf.expand_dims(x, axis=2)   
    vr  = xj - xi
    return vr

class ReaxFF_nn(object):
  def __init__(self,libfile='ffield',dataset={},
               dft='ase',atoms=None,
               cons=['val','vale','valang','valboc','lp3','cutoff','hbtol'],# 'acut''val',
               opt=None,energy_term={'etor':True,'eang':True,'eover':True,'eunder':True,
                                  'ecoul':True,'evdw':True,'elone':True,'ehb':True},
               mpopt=None,bdopt=None,mfopt=None,eaopt=[],
               VariablesToOpt=None,
               batch=200,sample='uniform',
               hbshort=6.75,hblong=7.5,
               vdwcut=10.0,
               atol=0.001,
               # hbtol=0.001,
               bore={'others':0.0},
               weight={'others':1.0},
               spv_vdw=False,vlo={'others':[(0.0,0.0)]},vup={'others':[(10.0,0.0)]},
               bo_clip=None,                 # e.g. bo_clip={'C-C':(1.3,8.5,8.5,0.0,0.0)}
               interactive=False,
               ro_scale=0.1,
               clip_op=True,
               clip={'others':(0.4,1.0)},   # parameter range
               InitCheck=True,
               resetDeadNeuron=False,
               optmol=True,
               lambda_me=0.1,
               nnopt=True,
               be_universal_nn=None,be_layer=[3,0],
               mf_universal_nn=None,mf_layer=[3,0],
               messages=1,MessageFunction=3,
               bo_layer=None,
               spec=[],
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
               screen=True,
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
      self.cons          = ['val','vale','valang','valboc','lp3','cutoff']
      self.cons         += cons
      self.energy_term   = {'etor':True,'eang':True,'eover':True,'eunder':True,
                            'ecoul':True,'evdw':True,'elone':True,'ehb':True,
                            'efcon':True,'etcon':True}
      self.energy_term.update(energy_term)
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
      self.screen        = screen
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
      self.bo_clip       = bo_clip
      self.vlo           = vlo
      self.vup           = vup
      self.bo_layer      = bo_layer
      self.weight        = weight
      self.spec          = spec
      self.time          = time.time()
      self.interactive   = interactive
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
      self.safety_value  = 0.00000001

      self.rcut,self.rcuta,self.re = self.read_lib()
      self.set_rcut(self.rcut,self.rcuta,self.re)

      self.set_variable_list()
      self.ic = Intelligent_Check(re=self.re,clip=clip,spec=self.spec,bonds=self.bonds,
                                  offd=self.offd,angs=self.angs,tors=self.torp,ptor=self.p_tor)
      self.p_,self.m_ = self.ic.check(self.p_,self.m_,resetDeadNeuron=self.resetDeadNeuron)

      if not self.libfile.endswith('.json'):
         self.p_['acut']    = atol
         self.p_['hbtol']   = atol

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
      self.natoms      = 0  
      strucs           = {}
      self.max_e       = {}
      # self.cell        = {}
      self.strcs       = []
      self.batch       = {}
      self.eself,self.evdw_,self.ecoul_ = {},{},{}

      for st in self.dataset: 
          nindex = []
          for key in strucs:
              if self.dataset[key]==self.dataset[st]:
                 nindex.extend(strucs[key].indexs)
          data_ = reax_force_data(structure=st,
                                 traj=self.dataset[st],
                               vdwcut=self.vdwcut,
                                 rcut=self.rcut,
                                rcuta=self.rcuta,
                              hbshort=self.hbshort,
                               hblong=self.hblong,
                                batch=self.batch_size,
                       variable_batch=True,
                               sample=self.sample,
                                    m=self.m_,
                             mf_layer=self.mf_layer_,
                       p=self.p_,spec=self.spec,bonds=self.bonds,
                  angs=self.angs,tors=self.tors,
                                  hbs=self.hbs,
                               screen=self.screen,
                               nindex=nindex)

          if data_.status:
             self.strcs.append(st)
             strucs[st]        = data_
             self.batch[st]    = strucs[st].batch
             self.nframe      += self.batch[st]
             self.natoms      += strucs[st].natom*self.batch[st]
             print('-  max energy of %s: %f.' %(st,strucs[st].max_e))
             self.max_e[st]    = strucs[st].max_e
             # self.evdw_[st]  = strucs[st].evdw
             # self.ecoul_[st] = strucs[st].ecoul  
             # self.cell[st]   = strucs[st].cell
             self.eself[st]    = strucs[st].eself  
          else:
             print('-  data status of %s:' %st,data_.status)
      self.nmol = len(strucs)
      
      self.memory()
      self.generate_data(strucs)
           
      self.set_zpe()

      self.build_graph()    
      self.feed_dict = self.feed_data()

      self.initialized = True
      return strucs

  def generate_data(self,strucs):
      ''' get data '''
      print('-  generating dataset ...')
      self.dft_energy                  = {}
      self.dft_forces                  = {}
      self.q                           = {}
      self.bdid                        = {}
      self.bdidr                       = {}
      self.dilink,self.djlink          = {},{}
      self.nbd,self.b,self.a,self.t    = {},{},{},{}
      self.ang_i,self.ang_j,self.ang_k = {},{},{}
      self.abij,self.abjk              = {},{}
      self.tij,self.tjk,self.tkl       = {},{},{}
      self.tor_j,self.tor_k            = {},{}
      self.tor_i,self.tor_l            = {},{}
      self.vb_i                        = {}
      self.vb_j                        = {}
      self.atom_name                   = {}
      self.natom                       = {}
      self.x,self.vr,self.r,self.rr    = {},{},{},{}
      self.vrr                         = {}
      self.nang                        = {}
      self.ntor                        = {}
      self.ns                          = {}
      self.s                           = {s:[] for s in self.spec}
      self.nv                          = {}
      self.na                          = {}
      self.nt                          = {}
      self.nhb                         = {}
      self.v                           = {}
      self.h                           = {}
      self.hbij,self.hbjk              = {},{}
      self.data                        = {}
      self.estruc                      = {}
      self.pmask                       = {}
      self.cell                        = {}
      self.rcell                       = {}
      self.cell0                       = {}
      self.cell1                       = {}
      self.cell2                       = {}
      self.eye                         = {}
      self.P                           = {}
      for s in strucs:
          s_ = s.split('-')[0]
          # self.natom[s]    = strucs[s].natom
          self.nang[s]     = strucs[s].nang
          self.ang_j[s]    = np.expand_dims(strucs[s].ang_j,axis=1)
          self.ang_i[s]    = np.expand_dims(strucs[s].ang_i,axis=1)
          self.ang_k[s]    = np.expand_dims(strucs[s].ang_k,axis=1)

          self.ntor[s]     = strucs[s].ntor
          self.tor_i[s]    = np.expand_dims(strucs[s].tor_i,axis=1)
          self.tor_j[s]    = np.expand_dims(strucs[s].tor_j,axis=1)
          self.tor_k[s]    = np.expand_dims(strucs[s].tor_k,axis=1)
          self.tor_l[s]    = np.expand_dims(strucs[s].tor_l,axis=1)

          self.hbij[s]     = {}
          self.hbjk[s]     = {}
          # print(strucs[s].hb_i)
          for hb in strucs[s].hb_i:
              self.hbij[s][hb] = np.concatenate([strucs[s].hb_i[hb],strucs[s].hb_j[hb]],axis=1)
              self.hbjk[s][hb] = np.concatenate([strucs[s].hb_j[hb],strucs[s].hb_k[hb]],axis=1)

          self.nbd[s]      = strucs[s].nbd
          self.na[s]       = strucs[s].na
          self.nt[s]       = strucs[s].nt
          self.nhb[s]      = strucs[s].nhb
          self.b[s]        = strucs[s].B
          self.a[s]        = strucs[s].A
          self.t[s]        = strucs[s].T

          self.bdid[s]     = strucs[s].bond  # bond index like pair (i,j).
          self.bdidr[s]    = strucs[s].bond[:,[1,0]]  # bond index like pair (i,j).
          self.atom_name[s]= strucs[s].atom_name
          self.dilink[s]   = strucs[s].dilink
          self.djlink[s]   = strucs[s].djlink
          self.P[s]        = {}

          self.s[s]        = {sp:[] for sp in self.spec}
          for i,sp in enumerate(self.atom_name[s]):
              self.s[s][sp].append(i)
          self.ns[s]       = {sp:len(self.s[s][sp]) for sp in self.spec}

          self.data[s]     = Dataset(dft_energy=strucs[s].energy_dft,
                                     x=strucs[s].x,
                                     cell=np.float32(strucs[s].cell),
                                     rcell=np.float32(strucs[s].rcell),
                                     forces=strucs[s].forces,
                                     q=strucs[s].qij)

          self.vb_i[s]     = {bd:[] for bd in self.bonds}
          self.vb_j[s]     = {bd:[] for bd in self.bonds}
          self.natom[s]    = strucs[s].natom
         
          for i in range(self.natom[s]):
              for j in range(self.natom[s]):
                  bd = self.atom_name[s][i]+'-'+self.atom_name[s][j]
                  if bd not in self.bonds:
                     bd = self.atom_name[s][j]+'-'+self.atom_name[s][i]
                  self.vb_i[s][bd].append(i)
                  self.vb_j[s][bd].append(j)
   
          self.pmask[s] = {}
          for sp in self.spec:
              pmask = np.zeros([self.natom[s],1])
              pmask[self.s[s][sp],:] = 1.0
              self.pmask[s][sp] = tf.constant(pmask,dtype=tf.float32,
                                              name='pmask_{:s}_{:s}'.format(s,sp))

          for bd in self.bonds:
             if len(self.vb_i[s][bd])==0:
                continue
             pmask = np.zeros([self.natom[s],self.natom[s],1])
             pmask[self.vb_i[s][bd],self.vb_j[s][bd],:] = 1.0
             self.pmask[s][bd] = tf.constant(pmask,dtype=tf.float32,
                                             name='pmask_{:s}_{:s}'.format(s,bd))

          self.cell[s]  = tf.constant(np.expand_dims(self.data[s].cell,axis=1),name='cell_{:s}'.format(s))
          self.rcell[s] = tf.constant(np.expand_dims(self.data[s].rcell,axis=1),name='rcell_{:s}'.format(s))
          # self.eye[s] = tf.constant(np.expand_dims(1.0 - np.eye(self.natom[s]),axis=0),name='eye_{:s}'.format(s))

          self.dft_energy[s] = tf.compat.v1.placeholder(tf.float32,shape=[self.batch[s]],
                                                name='DFT_energy_{:s}'.format(s))

          self.x[s] = tf.compat.v1.placeholder(tf.float32,shape=[self.batch[s],self.natom[s],3],
                                                    name='x_{:s}'.format(s))
          self.q[s] = tf.compat.v1.placeholder(tf.float32,shape=[self.natom[s],self.natom[s],self.batch[s]],
                                                   name='qij_{:s}'.format(s))
          # self.nang[mol] = molecules[mol].nang
          # self.nhb[mol]  = molecules[mol].nhb
          if strucs[s].forces is not None:
             self.dft_forces[s] = tf.compat.v1.placeholder(tf.float32,shape=[self.batch[s],self.natom[s],3],
                                            name='dftforces_{:s}'.format(s))
          else:
             self.dft_forces[s] = None
             
  def memory(self):
      self.frc = {}
      self.Bsi,self.Bpi,self.Bpp  = {},{},{}
      self.bop_si,self.bop_pi,self.bop_pp,self.bop = {},{},{},{}
      self.bosi,self.bosi_pen = {},{}
      self.bopi,self.bopp,self.bo0,self.bo,self.bso = {},{},{},{},{}

      self.Deltap,self.Delta,self.Bp = {},{},{}
      self.delta,self.Di,self.Dj,self.Di_boc,self.Dj_boc={},{},{},{},{}
      self.D,self.D_si,self.D_pi,self.D_pp = {},{},{},{}
      self.H,self.Hsi,self.Hpi,self.Hpp = {},{},{},{}

      self.So,self.fbot,self.fhb = {},{},{}
      self.EBD,self.E = {},{}
      self.powb,self.expb,self.ebond = {},{},{}

      self.esi   = {}
      self.ebd   = {}
      self.rbd   = {}
      self.rbd_  = {}
      self.Dbi   = {}
      self.Dbj   = {}
      for st in self.strcs:
          self.esi[st]   = {}
          self.ebd[st]   = {}
          self.rbd_[st]  = {}
          self.Dbi[st]   = {}
          self.Dbj[st]   = {}

      self.Delta_e,self.DE,self.Delta_lp,self.Dlp,self.Delta_ang  = {},{},{},{},{}
      self.Bpi,self.Delta_pi,self.Dpil,self.BSO,self.BOpi,self.Delta_lpcorr = {},{},{},{},{},{}
      self.eover,self.eunder,self.elone,self.Elone= {},{},{},{}
      self.EOV,self.EUN = {},{}

      self.BOij,self.BOjk = {},{}
      self.Nlp,self.Pbo = {},{}
      self.Eang,self.eang,self.theta0,self.fijk = {},{},{},{}
      self.pbo,self.sbo,self.SBO,self.SBO12,self.SBO3,self.SBO01 = {},{},{},{},{},{}
      self.dang,self.D_ang = {},{}
      self.thet,self.thet2,self.expang,self.f_7,self.f_8,self.rnlp = {},{},{},{},{},{}
      self.Epen,self.epen = {},{}

      self.expcoa1,self.texp0,self.texp1,self.texp2,self.texp3 = {},{},{},{},{}
      self.texp4,self.etcon,self.Etcon = {},{},{}

      self.cos3w,self.etor,self.Etor = {},{},{}
      self.BOpjk,self.BOtij,self.BOtjk,self.BOtkl,self.fijkl,self.so = {},{},{},{},{},{}
      self.f_9,self.f_10,self.f_11,self.f_12,self.expv2 = {},{},{},{},{}
      self.f11exp3,self.f11exp4 = {},{}

      self.v1,self.v2,self.v3 = {},{},{}
      self.Efcon,self.efcon = {},{}

      self.Evdw,self.nvb = {},{}
      self.Ecoul,self.evdw,self.ecoul,self.tpv,self.rth = {},{},{},{},{}

      self.exphb1,self.exphb2,self.sin4 = {},{},{}
      self.pc,self.BOhb,self.ehb,self.Ehb = {},{},{},{}

      self.E,self.zpe,self.eatom = {},{},{}
      self.forces                = {}
      self.loss,self.penalty,self.accur,self.MolEnergy = {},{},{},{}
      self.loss_force                                  = {}

      self.rv,self.q = {},{}
      self.theta = {}
      self.s_ijk,self.s_jkl,self.cos_w,self.cos2w,self.w={},{},{},{},{}
      # self.rhb,self.frhb,self.hbthe   = {},{},{}

  def build_graph(self):
      print('-  building graph: ')
      self.accuracy   = tf.constant(0.0,name='accuracy')
      self.accuracies = {}
      for mol in self.strcs:
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
                           self.etcon[mol] +
                           self.etor[mol]  +
                           self.efcon[mol] +
                           self.evdw[mol]  +
                           self.ecoul[mol] +
                           self.ehb[mol]   +
                           self.eself[mol], 
                           self.zpe[mol],name='E_%s' %mol)   

  def get_bond_energy(self,st):
      ''' get bond-energy of structure: st '''
      vr          = fvr(self.x[st])
      vrf         = tf.matmul(vr,self.rcell[st])
      vrf         = tf.where(vrf-0.5>0,vrf-1.0,vrf)
      vrf         = tf.where(vrf+0.5<0,vrf+1.0,vrf) 
      
      self.vr[st] = tf.matmul(vrf,self.cell[st])
      self.vrr[st]= tf.transpose(self.vr[st],[1,2,3,0])
      self.r[st]  = tf.sqrt(tf.reduce_sum(self.vr[st]*self.vr[st],axis=3) + self.safety_value) # 
      
      self.get_bondorder_uc(st)
      self.message_passing(st)
      self.get_final_state(st)
      self.get_ebond(st)
      self.ebond[st]= tf.reduce_sum(input_tensor=self.EBD[st],axis=0,name='bondenergy')

  def get_ebond(self,mol):
      Ebd  = []
      bosi = tf.gather_nd(self.bosi[mol],self.bdid[mol],
                          name='bosi_{:s}'.format(mol))
      bopi = tf.gather_nd(self.bopi[mol],self.bdid[mol],
                          name='bopi_{:s}'.format(mol)) 
      bopp = tf.gather_nd(self.bopp[mol],self.bdid[mol],
                          name='bopp_{:s}'.format(mol))
      
      for bd in self.bonds:
          nbd_ = self.nbd[mol][bd]
          if nbd_==0:
             continue
          b_  = self.b[mol][bd]
          bosi_ = tf.slice(bosi,[b_[0],0],[nbd_,self.batch[mol]])
          bopi_ = tf.slice(bopi,[b_[0],0],[nbd_,self.batch[mol]])
          bopp_ = tf.slice(bopp,[b_[0],0],[nbd_,self.batch[mol]])

          self.esi[mol][bd] = fnn('fe',bd, self.nbd[mol][bd],[bosi_,bopi_,bopp_],
                                  self.m,batch=self.batch[mol],layer=self.be_layer[1])
          self.ebd[mol][bd] = -self.p['Desi_'+bd]*self.esi[mol][bd]
          # print(self.ebd[mol],self.p['Desi_'+bd],self.p_['Desi_'+bd])
          Ebd.append(self.ebd[mol][bd])
      self.EBD[mol] = tf.concat(Ebd,0)

  def get_bondorder_uc(self,mol):
      bop_si,bop_pi,bop_pp = [],[],[]
      # print(self.r[st])
      self.rr[mol] = tf.transpose(self.r[mol],perm=(1,2,0))
      self.rbd[mol] = tf.gather_nd(self.rr[mol],self.bdid[mol],
                                   name='rbd_{:s}'.format(mol))

      for bd in self.bonds:
          nbd_ = self.nbd[mol][bd]
          if nbd_==0:
             continue
          b_                 = self.b[mol][bd]
          self.rbd_[mol][bd] = tf.slice(self.rbd[mol],[b_[0],0],[nbd_,self.batch[mol]])
          # print(bd,'r shape: ',self.rbd_[mol][bd].shape)
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

      bosi    = tf.concat(bop_si,0)
      # print('\n after concate \n',bosi.shape)
      bop_sir = tf.scatter_nd(self.bdid[mol],bosi,
                            shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      bop_sil = tf.scatter_nd(self.bdidr[mol],bosi,
                            shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      self.bop_si[mol] = bop_sir + bop_sil

      bopi    = tf.concat(bop_pi,0)
      bop_pir = tf.scatter_nd(self.bdid[mol],bopi,
                            shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      bop_pil = tf.scatter_nd(self.bdidr[mol],bopi,
                            shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      self.bop_pi[mol] = bop_pir + bop_pil

      bopp    = tf.concat(bop_pp,0)
      bop_ppr = tf.scatter_nd(self.bdid[mol],bopp,
                            shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      bop_ppl = tf.scatter_nd(self.bdidr[mol],bopp,
                            shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      self.bop_pp[mol] = bop_ppr + bop_ppl

      self.bop[mol]    = self.bop_si[mol] + self.bop_pi[mol] + self.bop_pp[mol]
      
      self.Deltap[mol] = tf.reduce_sum(self.bop[mol],axis=1,name='Deltap')
      self.D_si[mol]   = [tf.reduce_sum(self.bop_si[mol],axis=1,name='Deltap_si')]
      self.D_pi[mol]   = [tf.reduce_sum(self.bop_pi[mol],axis=1,name='Deltap_pi')]
      self.D_pp[mol]   = [tf.reduce_sum(self.bop_pp[mol],axis=1,name='Deltap_pp')]

  def get_bondorder(self,mol,t):
      ''' compute bond-order according the message function '''
      flabel  = 'fm'
      bosi_ = []
      bopi_ = []
      bopp_ = []

      H    =  tf.gather_nd(self.H[mol][t-1],self.bdid[mol],name=mol+'_h_gather')
      Hsi  =  tf.gather_nd(self.Hsi[mol][t-1],self.bdid[mol],name=mol+'_hsi_gather')
      Hpi  =  tf.gather_nd(self.Hpi[mol][t-1],self.bdid[mol],name=mol+'_hpi_gather')
      Hpp  =  tf.gather_nd(self.Hpp[mol][t-1],self.bdid[mol],name=mol+'_hpp_gather')

      for bd in self.bonds:
          nbd_ = self.nbd[mol][bd]
          if nbd_==0:
             continue
          b_   = self.b[mol][bd]
          bi   = self.dilink[mol][bd]
          bj   = self.djlink[mol][bd]
          Di   = tf.gather_nd(self.D[mol][t-1],bi)
          Dj   = tf.gather_nd(self.D[mol][t-1],bj)

          h    = tf.slice(H,[b_[0],0],[nbd_,self.batch[mol]],name=bd+'_h_slice')
          hsi  = tf.slice(Hsi,[b_[0],0],[nbd_,self.batch[mol]],name=bd+'_hsi_slice')
          hpi  = tf.slice(Hpi,[b_[0],0],[nbd_,self.batch[mol]],name=bd+'_hpi_slice')
          hpp  = tf.slice(Hpp,[b_[0],0],[nbd_,self.batch[mol]],name=bd+'_hpp_slice')

          b    = bd.split('-')

          if self.MessageFunction==1:
             Dsi_i = tf.gather_nd(self.D_si[mol][t-1],self.dilink[mol][bd]) - hsi
             Dpi_i = tf.gather_nd(self.D_pi[mol][t-1],self.dilink[mol][bd]) - hpi
             Dpp_i = tf.gather_nd(self.D_pp[mol][t-1],self.dilink[mol][bd]) - hpp 
             
             Dsi_j = tf.gather_nd(self.D_si[mol][t-1],self.djlink[mol][bd]) - hsi
             Dpi_j = tf.gather_nd(self.D_pi[mol][t-1],self.djlink[mol][bd]) - hpi
             Dpp_j = tf.gather_nd(self.D_pp[mol][t-1],self.djlink[mol][bd]) - hpp
             #Dpii = Dpi_i + Dpp_i
             #Dpij = Dpi_j + Dpp_j
             Fi    = fmessage(flabel,b[0],nbd_,[Dsi_i,Dpi_i,Dpp_i,h,Dpp_j,Dpi_j,Dsi_j],
                              self.m,batch=self.batch[mol],layer=self.mf_layer[1])
             Fj    = fmessage(flabel,b[1],nbd_,[Dsi_j,Dpi_j,Dpp_j,h,Dpp_i,Dpi_i,Dsi_i],
                              self.m,batch=self.batch[mol],layer=self.mf_layer[1])
          elif self.MessageFunction==3:
             self.Dbi[mol][bd]  = Di - h   
             self.Dbj[mol][bd]  = Dj - h   
             Fi   = fmessage(flabel,b[0],nbd_,[self.Dbi[mol][bd],h,self.Dbj[mol][bd]],self.m,
                             batch=self.batch[mol],layer=self.mf_layer[1])
             Fj   = fmessage(flabel,b[1],nbd_,[self.Dbj[mol][bd],h,self.Dbi[mol][bd]],self.m,
                             batch=self.batch[mol],layer=self.mf_layer[1])
          else:
             raise RuntimeError('-  Message funcition not implicited, you can only use 1 or 3.')
          F    = Fi*Fj
          Fsi,Fpi,Fpp = tf.unstack(F,axis=2)

          bosi_.append(hsi*Fsi)
          bopi_.append(hpi*Fpi)
          bopp_.append(hpp*Fpp)

      self.Bsi[mol] = tf.concat(bosi_,0)
      bosir = tf.scatter_nd(self.bdid[mol],self.Bsi[mol],
                           shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      bosil = tf.scatter_nd(self.bdidr[mol],self.Bsi[mol],
                           shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      bosi  = bosir + bosil

      self.Bpi[mol] = tf.concat(bopi_,0)
      bopir = tf.scatter_nd(self.bdid[mol],self.Bpi[mol],
                           shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      bopil = tf.scatter_nd(self.bdidr[mol],self.Bpi[mol],
                           shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      bopi  = bopir + bopil

      self.Bpp[mol] = tf.concat(bopp_,0)
      boppr = tf.scatter_nd(self.bdid[mol],self.Bpp[mol],
                           shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      boppl = tf.scatter_nd(self.bdidr[mol],self.Bpp[mol],
                           shape=(self.natom[mol],self.natom[mol],self.batch[mol]))
      bopp  = boppr + boppl
      bo   = bosi+bopi+bopp
      return bo,bosi,bopi,bopp

  def message_passing(self,mol):
      ''' finding the final Bond-order with a message passing '''
      self.H[mol]    = [self.bop[mol]]                     # 
      self.Hsi[mol]  = [self.bop_si[mol]]                  #
      self.Hpi[mol]  = [self.bop_pi[mol]]                  #
      self.Hpp[mol]  = [self.bop_pp[mol]]                  # 
      self.D[mol]    = [self.Deltap[mol]]                  # get the initial hidden state H[0]

      for t in range(1,self.messages+1):
          print('-  {:s} '.format(mol))           
          bo,bosi,bopi,bopp = self.get_bondorder(mol,t)
          self.H[mol].append(bo)                      # get the hidden state H[t]
          self.Hsi[mol].append(bosi)
          self.Hpi[mol].append(bopi)
          self.Hpp[mol].append(bopp)

          Delta   = tf.reduce_sum(bo,axis=1) 
          Dsi     = tf.reduce_sum(bosi,axis=1) 
          Dpi     = tf.reduce_sum(bopi,axis=1) 
          Dpp     = tf.reduce_sum(bopp,axis=1)  

          self.D[mol].append(Delta)                  # degree matrix
          self.D_si[mol].append(Dsi)
          self.D_pi[mol].append(Dpi)
          self.D_pp[mol].append(Dpp)

  def get_final_state(self,mol):     
      self.Delta[mol]  = self.D[mol][-1]
      self.bo0[mol]    = self.H[mol][-1]                 # fetch the final state 
      self.bosi[mol]   = self.Hsi[mol][-1]
      self.bopi[mol]   = self.Hpi[mol][-1]
      self.bopp[mol]   = self.Hpp[mol][-1]

      self.bo[mol]     = tf.nn.relu(self.bo0[mol] - self.atol)

      ovun             = 0.0
      for bd in self.bonds:
          if self.nbd[mol][bd]==0:
             continue
          ovun = ovun + self.p['ovun1_'+bd]*self.p['Desi_'+bd]*self.pmask[mol][bd]
          # bso.append(self.p['ovun1_'+bd]*self.p['Desi_'+bd]*bo0)

      self.bso[mol]      = ovun*self.bo0[mol]
      self.Bpi[mol]      = self.bopi[mol]+self.bopp[mol]

      self.Delta_pi[mol] = tf.reduce_sum(self.Bpi[mol],axis=1,name='sumover_Bpi')
      self.So[mol]       = tf.reduce_sum(self.bso[mol],axis=1,name='sumover_bso')  
      self.fbot[mol]     = taper(self.bo0[mol],rmin=self.atol,rmax=2.0*self.atol) 
      self.fhb[mol]      = taper(self.bo0[mol],rmin=self.hbtol,rmax=2.0*self.hbtol) 

  def get_atom_energy(self,st):
      ''' atomic energy of structure: st '''
      st_     = st.split('-')[0]
      eatom    = 0.0
      valang   = 0.0 
      val      = 0.0
      vale     = 0.0
      lp2      = 0.0
      ovun2    = 0.0
      ovun5    = 0.0
      for sp in self.spec:
          if self.ns[st][sp]>0:
             eatom  = eatom - self.p['atomic_'+sp]*self.pmask[st][sp]
             valang = valang + self.p['valang_'+sp]*self.pmask[st][sp]
             val    = val    + self.p['val_'+sp]*self.pmask[st][sp]
             vale   = vale   + self.p['vale_'+sp]*self.pmask[st][sp]
             lp2    = lp2    + self.p['lp2_'+sp]*self.pmask[st][sp]
             ovun2  = ovun2  + self.p['ovun2_'+sp]*self.pmask[st][sp]
             ovun5  = ovun5  + self.p['ovun5_'+sp]*self.pmask[st][sp]

      self.eatom[st]  = eatom
      self.Delta_ang[st]   = self.Delta[st] - valang 
 
      self.get_elone(st,lp2,val,vale) 
      self.elone[st]  = tf.reduce_sum(input_tensor=self.Elone[st],axis=0,name='elone_{:s}'.format(st))
     
      dlp  = self.Delta[st] - val - self.Delta_lp[st]
      dlp_ = tf.expand_dims(dlp,axis=0)
      self.Dpil[st] = tf.reduce_sum(dlp_*self.Bpi[st],axis=1)
      self.get_eover(st,val,ovun2) 
      self.eover[st]  = tf.reduce_sum(input_tensor=self.EOV[st],axis=0,name='eover_{:s}'.format(st))
      self.get_eunder(st,ovun2,ovun5) 
      self.eunder[st] = tf.reduce_sum(input_tensor=self.EUN[st],axis=0,name='eunder_{:s}'.format(st))
      self.zpe[st]    = tf.reduce_sum(input_tensor=self.eatom[st],name='zpe') + self.MolEnergy[st_]

  def get_elone(self,st,lp2,val,vale):
      ''' lone pair energy of structure: st '''
      Nlp                = 0.5*(vale - val)
      self.Delta_e[st]  = 0.5*(self.Delta[st] - vale)
      self.DE[st]       = -tf.nn.relu(-tf.math.ceil(self.Delta_e[st])) 
      self.Nlp[st]      = -self.DE[st] + tf.exp(-self.p['lp1']*4.0*tf.square(1.0+self.Delta_e[st]-self.DE[st]))

      self.Delta_lp[st] = Nlp - self.Nlp[st]                             # nan error
      # Delta_lp        = tf.clip_by_value(self.Delta_lp[mol],-1.0,10.0)  # temporary solution
      # Delta_lp        = tf.nn.relu(self.Delta_lp[st]+1) -1

      explp             = 1.0+tf.exp(-75.0*self.Delta_lp[st]) # -self.p['lp3']
      self.Elone[st]    = tf.math.divide(lp2*self.Delta_lp[st],explp,
                                          name='Elone_{:s}'.format(st))
                                          
  def get_eover(self,mol,val,ovun2):
      self.Delta_lpcorr[mol] = self.Delta[mol] - val - tf.math.divide(self.Delta_lp[mol],
                                1.0+self.p['ovun3']*tf.exp(self.p['ovun4']*self.Dpil[mol]))

      #self.so[atom]     = tf.gather_nd(self.SO,self.atomlist[atom])
      otrm1              = DIV_IF(1.0,self.Delta_lpcorr[mol]+val)
      # self.otrm2[atom] = tf.math.divide(1.0,1.0+tf.exp(self.p['ovun2_'+atom]*self.Delta_lpcorr[atom]))
      otrm2              = tf.sigmoid(-ovun2*self.Delta_lpcorr[mol])
      self.EOV[mol]      = self.So[mol] *otrm1*self.Delta_lpcorr[mol]*otrm2

  def get_eunder(self,mol,ovun2,ovun5):
      expeu1            = tf.exp(self.p['ovun6']*self.Delta_lpcorr[mol])
      eu1               = tf.sigmoid(ovun2*self.Delta_lpcorr[mol])
      expeu3            = tf.exp(self.p['ovun8']*self.Dpil[mol])
      eu2               = tf.math.divide(1.0,1.0+self.p['ovun7']*expeu3)
      self.EUN[mol]     = -ovun5*(1.0-expeu1)*eu1*eu2                          # must positive

  def get_threebody_energy(self,st):
      pbopow        = tf.negative(tf.pow(self.bo[st],8)) # original: self.BO0 
      pboexp        = tf.exp(pbopow)
      self.Pbo[st] = tf.reduce_prod(pboexp,axis=1,name=st+'_pbo') # BO Product

      if self.nang[st]==0 or (not self.energy_term['eang']):
         self.eang[st] = tf.cast(np.zeros([self.batch[st]]),tf.float32)
         self.epen[st] = tf.cast(np.zeros([self.batch[st]]),tf.float32)
         self.etcon[st]= tf.cast(np.zeros([self.batch[st]]),tf.float32)
      else:
         Eang  = []
         Epen  = []
         Etcon = []
         for ang in self.angs:
             sp  = ang.split('-')[1]
             # print(ang,self.na[st].get(ang,0))
             if self.na[st].get(ang,0)>0:
                ai        = self.ang_i[st][self.a[st][ang][0]:self.a[st][ang][1]]
                aj        = self.ang_j[st][self.a[st][ang][0]:self.a[st][ang][1]]
                ak        = self.ang_k[st][self.a[st][ang][0]:self.a[st][ang][1]]
                aij       = np.concatenate([ai,aj],axis=1)
                ajk       = np.concatenate([aj,ak],axis=1)
                aik       = np.concatenate([aj,ak],axis=1)
                # print('\n ai \n',self.ang_i[st][self.a[st][ang][0]:self.a[st][ang][1]])  
                boij      = tf.gather_nd(self.bo[st],aij,name='boij_'+ang)
                bojk      = tf.gather_nd(self.bo[st],ajk,name='bojk_'+ang)
                fij       = tf.gather_nd(self.fbot[st],aij,name='fboij_'+ang)  
                fjk       = tf.gather_nd(self.fbot[st],ajk,name='fbojk_'+ang) 

                delta     = tf.gather_nd(self.Delta[st],aj,
                               name='deltaj_{:s}_{:s}'.format(ang,sp)) - self.p['val_'+sp]
                delta_ang = tf.gather_nd(self.Delta_ang[st],aj,
                                         name='delta_ang_{:s}_{:s}'.format(ang,sp))
                delta_i   = tf.gather_nd(self.Delta[st],ai,
                                         name='deltai_{:s}_{:s}'.format(ang,sp))
                delta_k   = tf.gather_nd(self.Delta[st],ak,
                                         name='deltak_{:s}_{:s}'.format(ang,sp))
                sbo       = tf.gather_nd(self.Delta_pi[st],aj,
                                         name='Delta_pi_{:s}_{:s}'.format(ang,sp))
                pbo       = tf.gather_nd(self.Pbo[st],aj,
                                         name='pbo_{:s}_{:s}'.format(ang,sp))
                nlp       = tf.gather_nd(self.Nlp[st],aj,
                                         name='pbo_{:s}_{:s}'.format(ang,sp))

                theta     = self.get_theta(st,aij,ajk,aik)
                Ea,fijk   = self.get_eangle(sp,ang,boij,bojk,fij,fjk,theta,delta_ang,sbo,pbo,nlp)
                Ep        = self.get_epenalty(ang,delta,boij,bojk,fijk)
                Et        = self.get_three_conj(ang,delta_ang,delta_i,delta_k,boij,bojk,fijk) 
                Eang.append(Ea)
                Epen.append(Ep)
                Etcon.append(Et)

         self.Eang[st] = tf.concat(Eang,axis=0)
         self.Epen[st] = tf.concat(Epen,axis=0)
         self.Etcon[st]= tf.concat(Etcon,axis=0)
         self.eang[st] = tf.reduce_sum(self.Eang[st],0)
         self.epen[st] = tf.reduce_sum(self.Epen[st],0)
         self.etcon[st]= tf.reduce_sum(self.Etcon[st],0)

  def get_theta(self,st,aij,ajk,aik):
      # r = tf.transpose(self.r[st],[1,2,0])
      Rij = tf.gather_nd(self.rr[st],aij,name='rij_'+st)        # self.r[st][:,ai,aj]  
      Rjk = tf.gather_nd(self.rr[st],ajk,name='rjk_'+st)        #self.r[st][:,aj,ak]  
      # vr  = tf.transpose(self.vr[st],[1,2,3,0])
      # Rik = self.r[self.angi,self.angk]  
      vik = tf.gather_nd(self.vrr[st],aij) + tf.gather_nd(self.vrr[st],ajk)
      # vik = self.vr[st][:,ai,aj] + self.vr[st][:,aj,ak]
      Rik = tf.sqrt(tf.reduce_sum(tf.square(vik),1))

      Rij2= Rij*Rij
      Rjk2= Rjk*Rjk
      Rik2= Rik*Rik
      # print('\n Rij2 \n',Rij2)
      cos_theta = (Rij2+Rjk2-Rik2)/(2.0*Rij*Rjk)
      cos_theta = tf.where(cos_theta>0.9999999,0.9999999,cos_theta)   
      cos_theta = tf.where(cos_theta<-0.9999999,-0.9999999,cos_theta)
      theta     = tf.acos(cos_theta)
      return theta
 
  def get_eangle(self,sp,ang,boij,bojk,fij,fjk,theta,delta_ang,sbo,pbo,nlp):
      fijk           = fij*fjk

      theta0         = self.get_theta0(ang,delta_ang,sbo,pbo,nlp)
      thet           = theta0 - theta
      thet2          = tf.square(thet)

      expang         = tf.exp(-self.p['val2_'+ang]*thet2)
      f_7            = self.f7(sp,ang,boij,bojk)
      f_8            = self.f8(sp,ang,delta_ang)
      Eang           = fijk*f_7*f_8*(self.p['val1_'+ang]-self.p['val1_'+ang]*expang) 
      return Eang,fijk

  def get_theta0(self,ang,delta_ang,sbo,pbo,nlp):
      Sbo   = sbo - (1.0-pbo)*(delta_ang+self.p['val8']*nlp)    
      
      cond1 = tf.logical_and(tf.less_equal(Sbo,1.0),tf.greater(Sbo,0.0))
      S1    = tf.where(cond1,Sbo,0.0)                                    #  0< sbo < 1                  
      Sbo1  = tf.where(cond1,tf.pow(S1+0.0000001,self.p['val9']),0.0) 

      cond2 = tf.logical_and(tf.less(Sbo,2.0),tf.greater(Sbo,1.0))
      S2    = tf.where(cond2,Sbo,0.0)                     
      F2    = tf.where(cond2,1.0,0.0)                                    #  1< sbo <2
     
      S2    = 2.0*F2-S2  
      Sbo12 = tf.where(cond2,2.0-tf.pow(S2,self.p['val9']),0.0)          #  1< sbo <2
                                                                                                 #     sbo >2
      Sbo2  = tf.where(tf.greater_equal(Sbo,2.0),1.0,0.0)

      Sbo3   = Sbo1 + Sbo12 + 2.0*Sbo2
      theta0_ = 180.0 - self.p['theta0_'+ang]*(1.0-tf.exp(-self.p['val10']*(2.0-Sbo3)))
      theta0 = theta0_/57.29577951
      return theta0

  def f7(self,sp,ang,boij,bojk): 
      Fboi  = tf.where(tf.greater(boij,0.0),1.0,0.0)   
      Fbori = 1.0 - Fboi                                                                         # prevent NAN error
      expij = tf.exp(-self.p['val3_'+sp]*tf.pow(boij+Fbori,self.p['val4_'+ang])*Fboi)

      Fbok  = tf.where(tf.greater(bojk,0.0),1.0,0.0)   
      Fbork = 1.0 - Fbok 
      expjk = tf.exp(-self.p['val3_'+sp]*tf.pow(bojk+Fbork,self.p['val4_'+ang])*Fbok)
      fi = 1.0 - expij
      fk = 1.0 - expjk
      F  = fi*fk
      return F 

  def f8(self,sp,ang,delta_ang):
      exp6 = tf.exp( self.p['val6']*delta_ang)
      exp7 = tf.exp(-self.p['val7_'+ang]*delta_ang)
      F    = self.p['val5_'+sp] - (self.p['val5_'+sp]-1.0)*tf.divide(2.0+exp6,1.0+exp6+exp7)
      return F

  def get_epenalty(self,ang,delta,boij,bojk,fijk):
      f_9  = self.f9(delta)
      expi = tf.exp(-self.p['pen2']*tf.square(boij-2.0))
      expk = tf.exp(-self.p['pen2']*tf.square(bojk-2.0))
      Ep   = self.p['pen1_'+ang]*f_9*expi*expk*fijk
      return Ep

  def f9(self,Delta):
      exp3 = tf.exp(-self.p['pen3']*Delta)
      exp4 = tf.exp( self.p['pen4']*Delta)
      F = tf.divide(2.0+exp3,1.0+exp3+exp4)
      return F

  def get_three_conj(self,ang,delta_ang,delta_i,delta_k,boij,bojk,fijk):
      delta_coa  = delta_ang # self.D_ang[st] + valang - valboc
      expcoa1    = tf.exp(self.p['coa2']*delta_coa)

      texp0 = tf.divide(self.p['coa1_'+ang],1.0 + expcoa1)  
      texp1 = tf.exp(-self.p['coa3']*tf.square(delta_i-boij))
      texp2 = tf.exp(-self.p['coa3']*tf.square(delta_k-bojk))
      texp3 = tf.exp(-self.p['coa4']*tf.square(boij-1.5))
      texp4 = tf.exp(-self.p['coa4']*tf.square(bojk-1.5))
      Etc   = texp0*texp1*texp2*texp3*texp4*fijk 
      return Etc

  def get_fourbody_energy(self,st):
      self.s_ijk[st],self.s_jkl[st]            = {},{}
      self.cos_w[st],self.cos2w[st],self.w[st] = {},{},{}
      self.f_10[st],self.f_11[st]              = {},{}
      if (not self.energy_term['etor'] and not self.energy_term['efcon']) or self.ntor[st]==0:
         self.etor[st] = tf.zeros([self.batch[st]])
         self.efcon[st]= tf.zeros([self.batch[st]])
      else:
         Etor   =    []
         Efcon  =    []
         for tor in self.tors:
             spj  = tor.split('-')[1]
             spk  = tor.split('-')[2]
             if self.nt[st][tor]>0:
                ti        = self.tor_i[st][self.t[st][tor][0]:self.t[st][tor][1]]
                tj        = self.tor_j[st][self.t[st][tor][0]:self.t[st][tor][1]]
                tk        = self.tor_k[st][self.t[st][tor][0]:self.t[st][tor][1]]
                tl        = self.tor_l[st][self.t[st][tor][0]:self.t[st][tor][1]]
                tij       = np.concatenate([ti,tj],axis=1)
                tjk       = np.concatenate([tj,tk],axis=1)
                tkl       = np.concatenate([tk,tl],axis=1)
                # print('\n ai \n',self.ang_i[st][self.a[st][ang][0]:self.a[st][ang][1]])  
                boij      = tf.gather_nd(self.bo[st],tij,name='boij_{:s}_{:s}'.format(st,tor))
                bojk      = tf.gather_nd(self.bo[st],tjk,name='bojk_{:s}_{:s}'.format(st,tor))
                bokl      = tf.gather_nd(self.bo[st],tkl,name='bokl_{:s}_{:s}'.format(st,tor))
                bopjk     = tf.gather_nd(self.bopi[st],tjk,name='bojk_{:s}_{:s}'.format(st,tor))
                fij       = tf.gather_nd(self.fbot[st],tij,name='fij_{:s}_{:s}'.format(st,tor))
                fjk       = tf.gather_nd(self.fbot[st],tjk,name='fjk_{:s}_{:s}'.format(st,tor))
                fkl       = tf.gather_nd(self.fbot[st],tkl,name='fkl_{:s}_{:s}'.format(st,tor))

                delta_j   = tf.gather_nd(self.Delta_ang[st],tj,
                                         name='delta_ang_{:s}_{:s}'.format(tor,spj))
                 
                delta_k   = tf.gather_nd(self.Delta_ang[st],tk,
                                         name='delta_ang_{:s}_{:s}'.format(tor,spk))

                (self.w[st][tor],self.cos_w[st][tor],self.cos2w[st][tor],self.s_ijk[st][tor],
                 self.s_jkl[st][tor]) = self.get_torsion_angle(st,tor,tij,tjk,tkl)
                Et,fijkl  = self.get_etorsion(st,tor,boij,bojk,bokl,fij,fjk,fkl,
                                              bopjk,delta_j,delta_k,
                                              self.w[st][tor],self.cos_w[st][tor],self.cos2w[st][tor],
                                              self.s_ijk[st][tor],self.s_jkl[st][tor])
                Ef        = self.get_four_conj(tor,boij,bojk,bokl,self.w[st][tor],
                                               self.s_ijk[st][tor],self.s_jkl[st][tor],fijkl)
                Etor.append(Et)
                Efcon.append(Ef)

         self.Etor[st]  = tf.concat(Etor,axis=0)
         self.Efcon[st] = tf.concat(Efcon,axis=0)
         self.etor[st]  = tf.reduce_sum(self.Etor[st],0)
         self.efcon[st] = tf.reduce_sum(self.Efcon[st],0)

  def get_etorsion(self,st,tor,boij,bojk,bokl,fij,fjk,fkl,bopjk,delta_j,delta_k,
                        w,cos_w,cos2w,s_ijk,s_jkl):
      fijkl   = fij*fjk*fkl

      self.f_10[st][tor]    = self.f10(boij,bojk,bokl)
      self.f_11[st][tor]    = self.f11(delta_j,delta_k)
      expv2   = tf.exp(self.p['tor1_'+tor]*tf.square(2.0-bopjk-self.f_11[st][tor])) 

      cos3w   = tf.cos(3.0*w)
      v1      = 0.5*self.p['V1_'+tor]*(1.0+cos_w)
      v2      = 0.5*self.p['V2_'+tor]*expv2*(1.0-cos2w)
      v3      = 0.5*self.p['V3_'+tor]*(1.0+cos3w)
      
      Etor    = fijkl*self.f_10[st][tor]*s_ijk*s_jkl*(v1+v2+v3)
      return Etor,fijkl
  
  def get_torsion_angle(self,st,tor,tij,tjk,tkl):
      ''' compute torsion angle '''
      rij = tf.gather_nd(self.rr[st],tij,name='r_{:s}_{:s}'.format(st,tor)) # self.r[st][:,ti,tj]
      rjk = tf.gather_nd(self.rr[st],tjk,name='r_{:s}_{:s}'.format(st,tor))  
      rkl = tf.gather_nd(self.rr[st],tkl,name='r_{:s}_{:s}'.format(st,tor)) # self.r[st][:,tk,tl]

      
      vrjk= tf.gather_nd(self.vrr[st],tjk,name='vr_{:s}_{:s}'.format(st,tor)) # self.vr[st][:,tj,tk]
      vrkl= tf.gather_nd(self.vrr[st],tkl,name='vr_{:s}_{:s}'.format(st,tor))

      vrjl= vrjk + vrkl
      rjl = tf.sqrt(tf.reduce_sum(tf.square(vrjl),1))

      vrij=  tf.gather_nd(self.vrr[st],tij,name='vr_{:s}_{:s}'.format(st,tor))
      vril= vrij + vrjl
      ril = tf.sqrt(tf.reduce_sum(tf.square(vril),1))

      vrik= vrij + vrjk
      rik = tf.sqrt(tf.reduce_sum(tf.square(vrik),1))
      rij2= tf.square(rij)
      rjk2= tf.square(rjk)
      rkl2= tf.square(rkl)
      rjl2= tf.square(rjl)
      ril2= tf.square(ril)
      rik2= tf.square(rik)
      

      c_ijk = (rij2+rjk2-rik2)/(2.0*rij*rjk)
      c2ijk = tf.square(c_ijk)
      # tijk= tf.acos(c_ijk)
      cijk  =  1.0 - c2ijk
      s_ijk = tf.sqrt(tf.where(cijk<0.000000001,0.000000001,cijk))

      c_jkl = (rjk2+rkl2-rjl2)/(2.0*rjk*rkl)
      c2jkl = tf.square(c_jkl)
      cjkl  = 1.0  - c2jkl 
      s_jkl = tf.sqrt(tf.where(cjkl<0.000000001,0.000000001,cjkl))

      # c_ijl = (rij2+rjl2-ril2)/(2.0*rij*rjl)
      c_kjl = (rjk2+rjl2-rkl2)/(2.0*rjk*rjl)

      c2kjl = tf.square(c_kjl)
      ckjl  = 1.0 - c2kjl 
      s_kjl = tf.sqrt(tf.where(ckjl<0.000000001,0.000000001,ckjl))

      fz    = rij2+rjl2-ril2-2.0*rij*rjl*c_ijk*c_kjl
      fm    = rij*rjl*s_ijk*s_kjl

      fm    = tf.where(tf.logical_and(fm<=0.00001,fm>=-0.00001),1.0,fm)
      fac   = tf.where(tf.logical_and(fm<=0.00001,fm>=-0.00001),0.0,1.0)
      cos_w = 0.5*fz*fac/fm
      #cos_w= cos_w*ccijk*ccjkl
      cos_w = tf.where(cos_w>0.9999999,0.9999999,cos_w)   
      cos_w = tf.where(cos_w<-0.9999999,-0.9999999,cos_w)
      w= tf.acos(cos_w)
      cos2w = tf.cos(2.0*w)
      return w,cos_w,cos2w,s_ijk,s_jkl
  
  def f10(self,boij,bojk,bokl):
      exp1 = 1.0 - tf.exp(-self.p['tor2']*boij)
      exp2 = 1.0 - tf.exp(-self.p['tor2']*bojk)
      exp3 = 1.0 - tf.exp(-self.p['tor2']*bokl)
      return exp1*exp2*exp3

  def f11(self,delta_j,delta_k):
      delt = delta_j+delta_k
      f11exp3  = tf.exp(-self.p['tor3']*delt)
      f11exp4  = tf.exp( self.p['tor4']*delt)
      f_11 = tf.math.divide(2.0+f11exp3,1.0+f11exp3+f11exp4)
      return f_11

  def get_four_conj(self,tor,boij,bojk,bokl,w,s_ijk,s_jkl,fijkl):
      exptol= tf.exp(-self.p['cot2']*tf.square(self.p['acut'] - 1.5))
      expij = tf.exp(-self.p['cot2']*tf.square(boij-1.5))-exptol
      expjk = tf.exp(-self.p['cot2']*tf.square(bojk-1.5))-exptol 
      expkl = tf.exp(-self.p['cot2']*tf.square(bokl-1.5))-exptol

      f_12  = expij*expjk*expkl
      prod  = 1.0+(tf.square(tf.cos(w))-1.0)*s_ijk*s_jkl
      Efcon = fijkl*f_12*self.p['cot1_'+tor]*prod  
      return Efcon

  def f13(self,st,r):
      gammaw = tf.sqrt(tf.expand_dims(self.P[st]['gammaw'],0)*tf.expand_dims(self.P[st]['gammaw'],1))
      rr = tf.pow(r,self.p['vdw1'])+tf.pow(tf.math.divide(1.0,gammaw),self.p['vdw1'])
      f_13 = tf.pow(rr,tf.math.divide(1.0,self.p['vdw1']))  
      return f_13

  def get_tap(self,r):
      tp = 1.0+tf.math.divide(-35.0,tf.pow(self.vdwcut,4.0))*tf.pow(r,4.0)+ \
           tf.math.divide(84.0,tf.pow(self.vdwcut,5.0))*tf.pow(r,5.0)+ \
           tf.math.divide(-70.0,tf.pow(self.vdwcut,6.0))*tf.pow(r,6.0)+ \
           tf.math.divide(20.0,tf.pow(self.vdwcut,7.0))*tf.pow(r,7.0)
      return tp

  def get_vdw_energy(self,st):
      self.Evdw[st]     = tf.constant(0.0)
      self.Ecoul[st]    = tf.constant(0.0)
      nc                = 0
      cell0,cell1,cell2 = tf.unstack(self.cell[st],axis=2)
      self.cell0[st]    = tf.transpose(tf.expand_dims(cell0,1),[1,2,3,0])
      self.cell1[st]    = tf.transpose(tf.expand_dims(cell1,1),[1,2,3,0])
      self.cell2[st]    = tf.transpose(tf.expand_dims(cell2,1),[1,2,3,0])
      d1    = tf.constant(np.expand_dims(np.triu(np.ones([self.natom[st],self.natom[st]],
                                                         dtype=np.float32),k=0),axis=2))
      d2    = tf.constant(np.expand_dims(np.triu(np.ones([self.natom[st],self.natom[st]],
                                                         dtype=np.float32),k=1),axis=2))

      for key in ['gamma','gammaw']:
          self.P[st][key] =0.0 
          for sp in self.spec: 
              self.P[st][key] = self.P[st][key] + self.p[key+'_'+sp]*self.pmask[st][sp]

      for key in ['Devdw','alfa','rvdw']:
          self.P[st][key] =0.0 
          for bd in self.bonds:
              if len(self.vb_i[st][bd])==0:
                 continue
              self.P[st][key] = self.P[st][key] + self.p[key+'_'+bd]*self.pmask[st][bd]

      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell = self.cell0[st]*i + self.cell1[st]*j + self.cell2[st]*k
                  vr_  = self.vrr[st] + cell
                  r    = tf.sqrt(tf.reduce_sum(tf.square(vr_),2)+self.safety_value)
                  gamma= tf.sqrt(tf.expand_dims(self.P[st]['gamma'],1)*tf.expand_dims(self.P[st]['gamma'],2))
                  gm3  = tf.pow(tf.math.divide(1.0,gamma),3.0)
                  r3   = tf.pow(r,3.0)
                  fv_  = tf.where(tf.logical_and(r>0.00001,r<=self.vdwcut),1.0,0.0)

                  if nc<13:
                     fv = fv_*d1
                  else:
                     fv = fv_*d2

                  f_13  = self.f13(st,r)
                  tp    = self.get_tap(r)

                  expvdw1 = tf.exp(0.5*self.P[st]['alfa']*(1.0-tf.math.divide(f_13,2.0*self.P[st]['rvdw'])))
                  expvdw2 = tf.square(expvdw1) 
                  self.Evdw[st]  = self.Evdw[st] + fv*tp*self.P[st]['Devdw']*(expvdw2-2.0*expvdw1)
                  rth            = tf.pow(r3+gm3,1.0/3.0)  
                  # print('\n q \n',self.q[st])
                  self.Ecoul[st] = self.Ecoul[st] + tf.math.divide(fv*tp*self.q[st],rth)
                  nc += 1

      self.evdw[st]  = tf.reduce_sum(self.Evdw[st],axis=(0,1))
      self.ecoul[st] = tf.reduce_sum(self.Ecoul[st],axis=(0,1))
      # print('\n evdw \n',self.evdw[st])
      # print('\n ecoul \n',self.ecoul[st])

  def get_hb_energy(self,st):
      self.ehb[st]  = tf.constant(0.0)
      self.Ehb[st]    = tf.constant(0.0)
      Ehb             = []
      for hb in self.hbs:
          # print(hb, self.nhb[st][hb])
          if self.nhb[st][hb]==0:
             continue     
          bo          = tf.gather_nd(self.bo0[st],self.hbij[st][hb],name='r_{:s}_{:s}'.format(st,hb))
          fhb         = tf.gather_nd(self.fhb[st],self.hbij[st][hb],name='r_{:s}_{:s}'.format(st,hb))
          rij         = tf.gather_nd(self.rr[st],self.hbij[st][hb],name='r_{:s}_{:s}'.format(st,hb))

          rij2        = tf.square(rij)
          vrij        = tf.gather_nd(self.vrr[st],self.hbij[st][hb],name='vr_{:s}_{:s}'.format(st,hb)) 
          vrjk_       = tf.gather_nd(self.vrr[st],self.hbjk[st][hb],name='vr_{:s}_{:s}'.format(st,hb)) 
          ehb = 0.0
          for i in range(-1,2):
              for j in range(-1,2):
                  for k in range(-1,2):
                      cell   = tf.squeeze(self.cell0[st]*i + self.cell1[st]*j + self.cell2[st]*k,axis=0)
                      vrjk   = vrjk_ + cell 
                      
                      rjk2   = tf.reduce_sum(tf.square(vrjk),axis=1)
                      rjk    = tf.sqrt(rjk2)

                      vrik   = vrij + vrjk
                      rik2   = tf.reduce_sum(tf.square(vrik),axis=1)
                      rik    = tf.sqrt(rik2)

                      cos_th = (rij2+rjk2-rik2)/(2.0*rij*rjk)
                      hbthe  = 0.5-0.5*cos_th
                      frhb   = rtaper(rik,rmin=self.hbshort,rmax=self.hblong)

                      exphb1 = 1.0-tf.exp(-self.p['hb1_'+hb]*bo)
                      hbsum  = tf.math.divide(self.p['rohb_'+hb],rjk)+tf.math.divide(rjk,self.p['rohb_'+hb])-2.0
                      exphb2 = tf.exp(-self.p['hb2_'+hb]*hbsum)

                      sin4   = tf.square(hbthe)
                      ehb   += fhb*frhb*self.p['Dehb_'+hb]*exphb1*exphb2*sin4 
          Ehb.append(ehb)
      if Ehb:
         self.Ehb[st] = tf.concat(Ehb,axis=0)
         self.ehb[st] = tf.reduce_sum(self.Ehb[st],0)

  def set_zpe(self):
      if self.MolEnergy_ is None:
         self.MolEnergy_ = {}

      for mol in self.strcs:
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

  def get_forces(self,st):
      ''' compute forces with autograd method '''
      # E            = tf.reduce_sum(self.E[st])
      grad           = tf.gradients(ys=self.E[st],xs=self.x[st])
      # print(grad)
      self.forces[st] = -grad[0]

  def get_loss(self):
      ''' return the losses of the model '''
      self.Loss   = tf.constant(0.0)
      self.loss_f = tf.constant(0.0)
      for st in self.strcs:
          st_ = st.split('-')[0]
          if st in self.weight:
             w_ = self.weight[st]
          elif st_ in self.weight:
             w_ = self.weight[st_]
          else:
             w_ = self.weight['others']

          if self.losFunc   == 'n2':
             self.loss[st] = tf.nn.l2_loss(self.E[st]-self.dft_energy[st],
                                 name='loss_%s' %st)
             if self.dft_forces[st] is not None:
                self.get_forces(st) 
                self.loss_force[st] = tf.nn.l2_loss(self.forces[st]-self.dft_forces[st],
                                 name='loss_force_%s' %st)
          elif self.losFunc == 'abs':
             self.loss[st] = tf.compat.v1.losses.absolute_difference(self.dft_energy[st],self.E[st])
             if self.dft_forces[st] is not None:
                self.get_forces(st) 
                self.loss_force[st] = tf.compat.v1.losses.absolute_difference(self.forces[st],
                                                                               self.dft_forces[st],
                                 name='loss_force_%s' %st)
          elif self.losFunc == 'mse':
             self.loss[st] = tf.compat.v1.losses.mean_squared_error(self.dft_energy[st],self.E[st])
             if self.dft_forces[st] is not None:
                self.get_forces(st) 
                self.loss_force[st] = tf.compat.v1.losses.mean_squared_error(self.forces[st],
                                                                              self.dft_forces[st],
                                 name='loss_force_%s' %st)
          elif self.losFunc == 'huber':
             self.loss[st] = tf.compat.v1.losses.huber_loss(self.dft_energy[st],self.E[st],delta=self.huber_d)
             if self.dft_forces[st] is not None:
                self.get_forces(st) 
                self.loss_force[st] = tf.compat.v1.losses.mean_squared_error(self.forces[st],
                                                                              self.dft_forces[st],
                                                                              delta=self.huber_d,
                                                                        name='loss_force_%s' %st)
          else:
             raise NotImplementedError('-  This function not supported yet!')

          sum_edft = tf.reduce_sum(tf.abs(self.dft_energy[st]-self.max_e[st]))
          self.accur[st] = 1.0 - tf.reduce_sum(tf.abs(self.E[st]-self.dft_energy[st]))/(sum_edft+0.00000001)
          if st in self.loss_force:
             self.loss_f    += self.loss_force[st]*w_
          self.Loss      += self.loss[st]*w_ 
          if st.find('nomb')<0:
             self.accuracy += self.accur[st]
          else:
             self.nmol -= 1

      self.Loss   += self.loss_f
      self.ME      = tf.constant(0.0)
      for mol in self.strcs:
          mol_     = mol.split('-')[0] 
          self.ME += tf.square(self.MolEnergy[mol_])

      self.loss_penalty = self.get_penalty()
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
                   'cutoff','acut'] # # 'hbtol',
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

      if not self.energy_term['eover']:
         cons = cons + ['ovun1' ,'ovun2','ovun3','ovun4'] #
      if not self.energy_term['eunder']:
         cons = cons + ['ovun5','ovun6','ovun7','ovun8'] 
      # if self.optword.find('noover')>=0 and self.optword.find('nounder')>=0:
      #    cons = cons + ['ovun2','ovun3','ovun4'] 
      if not self.energy_term['elone']:
         cons = cons + ['lp2','lp3', 'lp1'] #
      if not self.energy_term['evdw']:
         cons = cons + ['gammaw','vdw1','rvdw','Devdw','alfa'] 
      if not self.energy_term['ehb']:
         cons = cons + ['Dehb','rohb','hb1','hb2'] #,'hbtol'

      self.tor_v = ['tor2','tor3','tor4','V1','V2','V3','tor1','cot1','cot2'] 

      if not self.energy_term['etor']:
         cons = cons + self.tor_v
      self.ang_v = ['theta0',
                    'val1','val2','val3','val4','val5','val6','val7',
                    'pen1','pen2','pen3','pen4',
                    'coa1','coa2','coa3','coa4'] 
      if not self.energy_term['eang']:
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

          if not self.energy_term['etor']:
             if key in ktor:
                self.p_[k] = 0.0
          if not self.energy_term['elone']:
             if key in 'lp2':
                self.p_[k] = 0.0
          if not self.energy_term['eover']:
             if key in 'ovun1':
                self.p_[k] = 0.0
          if not self.energy_term['eunder']:
             if key in 'ovun5':
                self.p_[k] = 0.0
          if not self.energy_term['eang']:
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
                          (6,0),(6,0),0,0,
                          self.mf_layer,self.mf_layer_,self.MessageFunction_,self.MessageFunction,
                          self.be_layer,self.be_layer_,1,1,
                          (9,0),(9,0),1,1,
                          None,self.be_universal_nn,self.mf_universal_nn,None)

  def set_parameters(self,libfile=None):
      if not libfile is None:
         self.p_,zpe,spec,bonds,offd,angs,torp,hbs = read_ffield(libfile=libfile)
      self.var = set_variables(self.p_, self.energy_term, self.cons, self.opt,self.eaopt,
                               self.punit, self.unit, self.conf_vale,
                               self.ang_v,self.tor_v)

      self.ea_var = {}        # parameter list to be optimized with evolutional algrithom
      for k in self.var:
            key = k.split('_')[0]
            if key in self.eaopt or k in self.eaopt:
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
                          self.mf_layer,self.mf_layer_,self.MessageFunction_,self.MessageFunction,
                          self.be_layer,self.be_layer_,1,1,
                          (9,0),(9,0),1,1,
                          None,self.be_universal_nn,self.mf_universal_nn,None)

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
         
          self.rc_bo[bd]=self.p['rosi_'+ofd]*tf.pow(rr,1.0/self.p['bo2_'+bd])

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

         # if self.board:
         #    writer = tf.compat.v1.summary.FileWriter("logs/", self.sess.graph)
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
             if key in p:
                p_ = p[key]*self.unit if k in self.punit else p[key]
                if (k in self.opt or key in self.opt) and (key not in self.cons and k not in self.cons):
                   if not hasH: upop.append(tf.compat.v1.assign(self.var[key],p_))
                elif key in self.ea_var:
                   self.feed_dict[self.var[key]] = p_
                   self.ea_var[key]              = p[key]
      
      for mol in self.strcs:
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
      totrain = True
      i       = 0
      zpe     = {}

      while totrain:
          if step==0:
             loss,loss_f,lpenalty,self.ME_,accu,accs   = self.sess.run([self.Loss,
                                                       self.loss_f,
                                                       self.loss_penalty,
                                                       self.ME,self.accuracy, self.accur],
                                                   feed_dict=self.feed_dict)
          else:
             loss,loss_f,lpenalty,self.ME_,accu,accs,_ = self.sess.run([self.Loss,
                                                self.loss_f,
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
                self.logger.info('energy loss is {:f}, forces loss is {:f}.'.format(loss,loss_f))
                loss_ = 99999999999.9 
                self.write_lib(libfile=self.libfile,loss=loss_)
                accu  = -1.0
                break
             else:
                break

          if self.optmol:
             los_ = loss - lpenalty - self.ME_*self.lambda_me  # - loss_f
          else:
             los_ = loss - lpenalty                            # - loss_f
          loss_ = los_ if i==0 else min(loss_,los_)

          if i%print_step==0:
             current = time.time()
             elapsed_time = current - self.time

             acc = ''
             for key in accs:
                 acc += key+': %6.4f ' %accs[key]
             loss_f = loss_f/self.natoms
             loss_e = los_/self.natoms
             self.logger.info('-  step: %d loss: %9.7f accs: %f %s force: %8.6f pen: %6.4f me: %6.4f time: %6.4f' %(i,
                              loss_e,accu,acc,loss_f,lpenalty,self.ME_,elapsed_time))
             self.time = current

          if i%writelib==0 or i==step:
             self.lib_bk = libfile+'_'+str(i)
             self.write_lib(libfile=self.lib_bk,loss=loss_)

             if i==step:
                if saveffield: self.write_lib(libfile=libfile,loss=loss_)
                # E,dfte,zpe = self.sess.run([self.E,self.dft_energy,self.zpe],
                #                           feed_dict=self.feed_dict)
                # self.plot_result(i,E,dfte)
                if not close_session or accu>self.convergence:
                   totrain = False
                else:
                   i = 0
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
          i += 1
      
      self.loss_ = loss_ if not (np.isnan(loss) or np.isinf(loss)) else 9999999.9
      if self.loss_ < 9999999.0: self.write_lib(libfile=libfile,loss=loss_)
      if close_session:
         self.print_penalty()
         tf.compat.v1.reset_default_graph()
         self.sess.close()
         return loss_,accu,accMax,i,zpe

  def feed_data(self):
      feed_dict = {}
      for mol in self.strcs:
          feed_dict[self.dft_energy[mol]] = self.data[mol].dft_energy
          # feed_dict[self.rbd[mol]] = self.data[mol].rbd
          feed_dict[self.x[mol]]     = self.data[mol].x
          feed_dict[self.q[mol]]     = np.transpose(self.data[mol].q,(1,2,0))
          if self.dft_forces[mol] is not None:
             feed_dict[self.dft_forces[mol]] = self.data[mol].forces
         #  if self.nang[mol]>0:
         #     feed_dict[self.theta[mol]] = self.data[mol].theta

         #  if self.ntor[mol]>0:
         #     feed_dict[self.s_ijk[mol]] = self.data[mol].s_ijk
         #     feed_dict[self.s_jkl[mol]] = self.data[mol].s_jkl
         #     feed_dict[self.w[mol]]     = self.data[mol].w

         #  if self.nhb[mol]>0:
         #     feed_dict[self.rhb[mol]]   = self.data[mol].rhb
         #     feed_dict[self.frhb[mol]]  = self.data[mol].frhb
         #     feed_dict[self.hbthe[mol]] = self.data[mol].hbthe
      for k in self.ea_var:
          key = k.split('_')[0]
          p_  = self.p_[k]*self.unit if key in self.punit else self.p_[k]
          feed_dict[self.var[k]] = p_
      return feed_dict

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

  def get_penalty(self):
      ''' adding some penalty term to pretain the physical meaning '''
      log_    = -9.21044036697651
      penalty = tf.constant(0.0)
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

          for mol in self.strcs:
              if self.nbd[mol][bd]>0:       
                 b_    = self.b[mol][bd]
                 bdid  = self.bdid[mol][b_[0]:b_[1]]
                 bo0_  = tf.gather_nd(self.bo0[mol],bdid,
                                      name='bo0_supervize_{:s}'.format(bd)) 
                 bop_  = tf.gather_nd(self.bop[mol],bdid,
                                      name='bop_supervize_{:s}'.format(bd)) 
                 fbo  = tf.where(tf.less(self.rbd_[mol][bd],self.rc_bo[bd]),0.0,1.0)     # bop should be zero if r>rcut_bo
                 self.penalty_bop[bd]  +=  tf.reduce_sum(bop_*fbo)                       #####  

                 fao  = tf.where(tf.greater(self.rbd_[mol][bd],self.rcuta[bd]),1.0,0.0)  ##### r> rcuta that bo = 0.0
                 self.penalty_bo_rcut[bd] += tf.reduce_sum(bo0_*fao)

                 fesi = tf.where(tf.less_equal(bo0_,self.botol),1.0,0.0)                 ##### bo <= 0.0 that e = 0.0
                 self.penalty_be_cut[bd]  += tf.reduce_sum(tf.nn.relu(self.esi[mol][bd]*fesi))
                 
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

  def print_penalty(self):
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
      # for mol in self.strcs:
      #     print('Max Bond-Order of {:s} {:f}'.format(mol,np.max(bo[mol])))
      #     print('Max Bond Energy of {:s} {:f}'.format(mol,max(ebond[mol])))
      #     print('Max Angle Energy of {:s} {:f}'.format(mol,max(eang[mol])))
          

  def plot_result(self,step,E,dfte):
      if not exists('results'):
         makedirs('results')
      Y,Yp = [],[]
      
      for mol in self.strcs:
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

  def close(self):
      print('-  Job compeleted.')
      # self.sess.close()
      self.memory()

