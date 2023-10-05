import json as js
import numpy as np
from ase import Atoms
from ase.io import read,write
from ase.calculators.calculator import Calculator, all_changes
from .qeq import qeq
from .RadiusCutOff import setRcut
from .reaxfflib import read_ffield,write_lib
from .neighbors import get_neighbors,get_pangle,get_ptorsion,get_phb


def rtaper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r<rmin,1.0,0.0) # r > rmax then 1 else 0

    ok    = np.logical_and(r<=rmax,r>rmin)      # rmin < r < rmax  = r else 0
    r2    = np.where(ok,r,0.0)
    r20   = np.where(ok,1.0,0.0)

    rterm = np.divide(1.0,np.power(rmax-rmin,3))
    rm    = rmax*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmin*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3


def taper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r>rmax,1.0,0.0) # r > rmax then 1 else 0

    ok    = np.logical_and(r<=rmax,r>rmin)      # rmin < r < rmax  = r else 0
    r2    = np.where(ok,r,0.0)
    r20   = np.where(ok,1.0,0.0)

    rterm = np.divide(1.0,np.power(rmin-rmax,3)+0.0000001)
    rm    = rmin*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmax*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3
    

def fvr(x):
    xi  = np.expand_dims(x,axis=0)
    xj  = np.expand_dims(x,axis=1) 
    vr  = xj - xi
    return vr


def fr(vr):
    R   = np.sqrt(np.sum(vr*vr,axis=2))
    return R


def sigmoid(x):
    s = 1.0/(1.0+np.exp(-x))
    return s


def relu(x):
    return np.maximum(0, x)


class IRFF(Calculator):
  '''Intelligent Machine-Learning ASE calculator'''
  name = "IRFF"
  implemented_properties = ["energy", "forces", "stress","pressure"]
  def __init__(self,atoms=None,
               mol=None,
               libfile='ffield',
               vdwcut=10.0,
               atol=0.001,
               hbtol=0.001,
               nn=False,# vdwnn=False,
               messages=1,
               hbshort=6.75,hblong=7.5,
               nomb=False,  # this option is used when deal with metal system
               label="IRFF", **kwargs):
      Calculator.__init__(self,label=label, **kwargs)
      self.atoms        = atoms
      self.cell         = atoms.get_cell()
      self.atom_name    = self.atoms.get_chemical_symbols()
      self.natom        = len(self.atom_name)
      self.spec         = []
      self.nn           = nn
      # self.vdwnn      = vdwnn
      self.EnergyFunction = 0
      self.autograd     = autograd
      self.nomb         = nomb # without angle, torsion and hbond manybody term
      self.messages     = messages 
      self.safety_value = 0.000000001
      self.GPa          = 1.60217662*1.0e2
      # self.CalStress    = CalStress

      if libfile.endswith('.json'):
         lf                  = open(libfile,'r')
         j                   = js.load(lf)
         self.p              = j['p']
         m                   = j['m']
         self.MolEnergy_     = j['MolEnergy']
         self.messages       = j['messages']
         self.BOFunction     = j['BOFunction']
         self.EnergyFunction = j['EnergyFunction']
         self.MessageFunction= j['MessageFunction']
         self.VdwFunction    = j['VdwFunction']
         self.bo_layer       = j['bo_layer']
         self.mf_layer       = j['mf_layer']
         self.be_layer       = j['be_layer']
         self.vdw_layer      = j['vdw_layer']
         if not self.vdw_layer is None:
            self.vdwnn       = True
         else:
            self.vdwnn       = False
         rcut                = j['rcut']
         rcuta               = j['rcutBond']
         re                  = j['rEquilibrium']
         lf.close()
         self.init_bonds()
         if mol is None:
            self.emol = 0.0
         else:
            mol_ = mol.split('-')[0]
            if mol_ in self.MolEnergy_:
               self.emol = self.MolEnergy_[mol_]
            else:
               self.emol = 0.0
      else:
         self.p,zpe_,self.spec,self.bonds,self.offd,self.angs,self.torp,self.Hbs= \
                       read_ffield(libfile=libfile,zpe=False)
         m             = None
         self.bo_layer = None
         self.emol     = 0.0
         rcut          = None
         rcuta         = None
         self.vdwnn    = False
         self.EnergyFunction  = 0
         self.MessageFunction = 0
         self.VdwFunction     = 0
         self.p['acut']   = 0.0001
         self.p['hbtol']  = 0.0001
      if m is None:
         self.nn=False
         
      for sp in self.atom_name:
          if sp not in self.spec:
             self.spec.append(sp)

      self.hbshort   = hbshort
      self.hblong    = hblong
      self.set_rcut(rcut,rcuta,re)
      self.vdwcut    = vdwcut
      self.botol     = 0.01*self.p['cutoff']
      self.atol      = self.p['acut']   # atol
      self.hbtol     = self.p['hbtol']  # hbtol
      self.check_offd()
      self.check_hb()
      self.get_rcbo()
      self.set_p(m,self.bo_layer)
      # self.Qe= qeq(p=self.p,atoms=self.atoms)

