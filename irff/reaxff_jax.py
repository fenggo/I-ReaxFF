import json as js
import numpy as np
from ase import Atoms
from ase.io import read,write
from ase.calculators.calculator import Calculator, all_changes
from .qeq import qeq
from .RadiusCutOff import setRcut
from .reaxfflib import read_ffield,write_lib
from .intCheck import Intelligent_Check
from .reax_force_data import reax_force_data,Dataset
from .neighbors import get_neighbors,get_pangle,get_ptorsion,get_phb
from .set_matrix_jax import set_matrix
from .intCheck import check_tors as check_torsion
import jax
import jax.numpy as jnp
import optax

try:
   from .neighbor import get_neighbors,get_pangle,get_ptorsion,get_phb
except ImportError:
   from .neighbors import get_neighbors,get_pangle,get_ptorsion,get_phb

# Enable float64
jax.config.update('jax_enable_x64', True)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def rtaper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r<rmin,np.full_like(r,1.0),np.full_like(r,0.0)) # r > rmax then 1 else 0

    ok    = np.logical_and(r<=rmax,r>rmin)     # rmin < r < rmax  = r else 0
    r2    = np.where(ok,r,np.full_like(r,0.0))
    r20   = np.where(ok,np.full_like(r,1.0),np.full_like(r,0.0))

    rterm = 1.0/(rmax-rmin)**3.0
    rm    = rmax*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmin*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3

def taper(r,rmin=0.001,rmax=0.002):
    ''' taper function for bond-order '''
    r3    = np.where(r>rmax,np.full_like(r,1.0),np.full_like(r,0.0)) # r > rmax then 1 else 0

    ok    = np.logical_and(r<=rmax,r>rmin)      # rmin < r < rmax  = r else 0
    r2    = np.where(ok,r,np.full_like(r,0.0))
    r20   = np.where(ok,np.full_like(r,1.0),np.full_like(r,0.0))

    rterm = 1.0/(rmin-rmax)**3.0
    rm    = rmin*r20
    rd    = rm - r2
    trm1  = rm + 2.0*r2 - 3.0*rmax*r20
    r22   = rterm*rd*rd*trm1
    return r22+r3
    
def fvr(x):
    xi  = np.expand_dims(x, 1)
    xj  = np.expand_dims(x, 2) 
    vr  = xj - xi
    # vr = torch.cdist(x, x, p=2)
    return vr

def fr(vr):
    R   = np.sqrt(np.sum(vr*vr,2))
    return R

def DIV(y,x):
    xok = (x!=0.0)
    f = lambda x: y/x
    safe_x = np.where(xok,x,np.full_like(x,1.0))
    return np.where(xok, f(safe_x), np.full_like(x,0.0))

def DIV_IF(y,x):
    xok = (x!=0.0)
    f = lambda x: y/x
    safe_x = np.where(xok,x,np.full_like(x,0.00000001))
    return np.where(xok, f(safe_x), f(safe_x))

def relu(x):
    return np.where(x>0.0,x,np.full_like(x,0.0))  

def fmessage(pre,bd,x,m,layer=5):
    ''' Dimention: (nbatch,3) input = 3
                Wi:  (3,9) 
                Wh:  (9,9)
                Wo:  (9,3)  output = 3
    '''
    X   = np.expand_dims(np.stack(x,axis=2),axis=2)
    # print('\n X \n',X,X.shape)
    o   =  []                        
    o.append(sigmoid(np.matmul(X,m[pre+'wi_'+bd])+m[pre+'bi_'+bd]))   # input layer
    # print('\n ai \n',o[-1])
    for l in range(layer):                                                   # hidden layer      
        o.append(sigmoid(np.matmul(o[-1],m[pre+'w_'+bd][l])+m[pre+'b_'+bd][l]))
    out = sigmoid(np.matmul(o[-1],m[pre+'wo_'+bd]) + m[pre+'bo_'+bd])  # output layer
    return  np.squeeze(out, axis=2) 

def fnn(pre,bd,x,m,layer=5):
    ''' Dimention: (nbatch,3) input = 3
                Wi:  (3,8) 
                Wh:  (8,8)
                Wo:  (8,1)  output = 3
    '''
    X   = np.expand_dims(np.stack(x,axis=2),axis=2)
    #                                 #        Wi:  (3,8) 
    o   =  []
    o.append(sigmoid(np.matmul(X,m[pre+'wi_'+bd])+m[pre+'bi_'+bd]))   # input layer

    for l in range(layer):                                     # hidden layer      
        o.append(sigmoid(np.matmul(o[-1],m[pre+'w_'+bd][l])+m[pre+'b_'+bd][l]))

    out = sigmoid(np.matmul(o[-1],m[pre+'wo_'+bd]) + m[pre+'bo_'+bd])  # output layer
    # print(out.shape)
    return  np.squeeze(out, axis=(2, 3)) 



# ── JAX-based force computation ──
# These JAX functions compute energy from positions, allowing
# jax.grad to efficiently compute forces.

def taper_jnp(r, rmin=0.001, rmax=0.002):
    r3 = jnp.where(r > rmax, jnp.ones_like(r), jnp.zeros_like(r))
    ok = jnp.logical_and(r <= rmax, r > rmin)
    r2 = jnp.where(ok, r, jnp.zeros_like(r))
    r20 = jnp.where(ok, jnp.ones_like(r), jnp.zeros_like(r))
    rterm = 1.0 / (rmin - rmax) ** 3.0
    rm = rmin * r20
    rd = rm - r2
    trm1 = rm + 2.0 * r2 - 3.0 * rmax * r20
    return rterm * rd * rd * trm1 + r3

def fvr_jnp(x):
    xi = jnp.expand_dims(x, 1)
    xj = jnp.expand_dims(x, 2)
    return xj - xi

def fmessage_jnp(pre, bd, x, m, layer=5):
    X = jnp.expand_dims(jnp.stack(x, axis=2), axis=2)
    o = [jax.nn.sigmoid(jnp.matmul(X, m[pre + 'wi_' + bd]) + m[pre + 'bi_' + bd])]
    for l in range(layer):
        o.append(jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'w_' + bd][l]) + m[pre + 'b_' + bd][l]))
    out = jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'wo_' + bd]) + m[pre + 'bo_' + bd])
    return jnp.squeeze(out, axis=2)

def fnn_jnp(pre, bd, x, m, layer=5):
    X = jnp.expand_dims(jnp.stack(x, axis=2), axis=2)
    o = [jax.nn.sigmoid(jnp.matmul(X, m[pre + 'wi_' + bd]) + m[pre + 'bi_' + bd])]
    for l in range(layer):
        o.append(jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'w_' + bd][l]) + m[pre + 'b_' + bd][l]))
    out = jax.nn.sigmoid(jnp.matmul(o[-1], m[pre + 'wo_' + bd]) + m[pre + 'bo_' + bd])
    return jnp.squeeze(out, axis=(2, 3))


class ReaxFF_nn:

  ''' Force Learning '''
  name = "ReaxFF_nn"
  implemented_properties = ["energy", "forces"]
  def __init__(self,dataset={},data={},
               # batch={'others':500},
               sample='uniform',
               libfile='ffield.json',
               vdwcut=10.0,
               messages=1,
               hbshort=6.75,hblong=7.5,
               EnergyFunction=1,MessageFunction=3,
               mf_layer=None,be_layer=None,
               be_universal=None,mf_universal=None,
               cons=['val','vale','valang','vale','valboc','lp3','gamma',
                     'cutoff','hbtol'],# 'acut''val',
               opt=None,
               opt_term={'etcon':0,'efcon':0,'etor':1,
                         'elone':0,'eover':0,'eunder':0,'epen':0},
               clip={},
               bdopt=None,mfopt=None,beopt=None,
               weight_force={'others':1.0},weight_energy={'others':1.0},
               bo_clip={},                 # e.g. bo_clip={'C-C':(1.3,8.5,8.5,0.0,0.0)}
               lambda_bd=1000.0,
               lambda_pi=0.0,
               lambda_reg=0.01,
               lambda_ang=0.0,
               fixrcbo=False,
               eaopt=[],
               nomb=False,              # this option is used when deal with metal system
               screen=False,
               tors=[],
               device={'all':'cpu'}):
      
      self.dataset      = dataset 
      self.data         = data
      # self.batch_size   = batch
      # if 'others' not in self.batch_size:
      #    self.batch_size['others'] = 200
      self.sample       = sample        # uniform or random
      self.opt          = opt
      self.opt_term     = opt_term
      self.bdopt        = bdopt
      self.mfopt        = mfopt
      self.beopt        = beopt
      self.eaopt        = eaopt
      self.cons         = ['val','vale','valang','vale','valboc','lp3','gamma',
                           'cutoff','hbtol']
      self.cons.extend(cons)
      self.clip         = clip
      self.fixrcbo      = fixrcbo
      self.weight_force = weight_force
      self.weight_energy= weight_energy
      self.mf_layer     = mf_layer
      self.be_layer     = be_layer
      self.mf_universal = mf_universal
      self.be_universal = be_universal
      self.lambda_bd    = lambda_bd
      self.lambda_reg   = lambda_reg
      self.lambda_pi    = lambda_pi
      self.lambda_ang   = lambda_ang
      self.hbshort      = hbshort
      self.hblong       = hblong
      self.vdwcut       = vdwcut
      self.screen       = screen
      self.EnergyFunction = EnergyFunction
      self.MessageFunction= MessageFunction
      self.bo_clip      = bo_clip
      self.m_,self.rcut,self.rcuta,self.re  = self.read_ffield(libfile)

      if self.m_ is not None:
         self.nn        = True          # whether use neural network
      
      self._device      = device
      if 'others' not in self._device:
         self._device['others'] = 'cpu'
      if 'diff' not in self._device:
         if 'others' in self._device:
            self._device['diff'] = self._device['others']
      self.device       = {'others':(self._device['others']),
                           'diff':(self._device['diff'])}
      self.tors         = tors

      self.pp           = {}   # training parameter
      self.set_p() 
      self.stack_tensor()

      self.results        = {}
      self.nomb           = nomb # without angle, torsion and hbond manybody term
      self.messages       = messages 
      # self.safety_value   = np.array(0.00000001)
      # for dev in self.devices:
      #     self.safety_value
      self.set_memory()
      # self.params = np.random.rand(3, 3))
      # self.Qe= qeq(p=self.p,atoms=self.atoms)

  def get_total_energy(self,st):
      ''' compute the total energy of moecule '''
      if self.zpe[st].device != self.device[st]:
         self.zpe[st] = self.zpe[st]

      self.E[st] = (self.ebond[st] + 
                    self.eover[st] + self.eunder[st]+ self.elone[st] +
                    self.eang[st]  + self.epen[st]  + self.etcon[st] +
                    self.etor[st]  + self.efcon[st] +
                    self.ecoul[st] + self.evdw[st]  + 
                    self.ehb[st]   +
                    self.eself[st] + self.zpe[st]     )
      
  def forward(self, st):
      # for st in self.strcs:
      self.get_bond_energy(st)      # get bond energy for every structure
      self.get_atomic_energy(st)
      self.get_threebody_energy(st)
      self.get_fourbody_energy(st)
      self.get_vdw_energy(st)
      self.get_hb_energy(st)
      self.get_total_energy(st)
      if self.weight_force[st]!=0:
         self.get_forces(st)
      return self.E,self.force

  def get_loss(self,st):
      ''' compute loss '''
      self.loss_e = 0.0
      self.loss_f = 0.0
      weight_e = self.weight_energy['others'] if st not in self.weight_energy else self.weight_energy[st]
      self.loss_e = self.loss_e + np.sum(np.square(self.E[st] - self.dft_energy[st]))*weight_e
      if self.dft_forces[st] is not None:
         weight_f = self.weight_force['others'] if st not in self.weight_force else self.weight_force[st]
         self.loss_f = self.loss_f + np.sum(np.square(self.force[st] - self.dft_forces[st]))*weight_f
      self.loss_penalty = self.get_penalty(st)
      return self.loss_e + self.loss_f + self.loss_penalty
  def get_forces(self, st):
      ''' compute forces using jax.grad '''
      try:
          # Convert necessary arrays to jnp
          x_jnp = jnp.array(self.x[st])
          rcell_jnp = jnp.array(self.rcell[st])
          cell_jnp = jnp.array(self.cell[st])
          
          # Define energy function
          def energy_fn(x):
              return self._compute_energy_jax(st, x, rcell_jnp, cell_jnp)
          
          # Compute forces = -dE/dx
          grad_fn = jax.grad(energy_fn)
          forces_jnp = grad_fn(x_jnp)
          self.force[st] = -np.array(forces_jnp)
      except Exception as e:
          # Fallback to finite difference if JAX fails
          print(f"  Warning: jax.grad failed ({e}), using finite diff")
          self._forces_finite_diff(st)

  def _forces_finite_diff(self, st):
      ''' Finite difference fallback for forces. '''
      eps = 1e-5
      nbatch = self.x[st].shape[0]
      natom = self.x[st].shape[1]
      forces = np.zeros((nbatch, natom, 3))
      for ib in range(nbatch):
          for ia in range(natom):
              for idim in range(3):
                  x_orig = self.x[st][ib, ia, idim]
                  self.x[st][ib, ia, idim] = x_orig + eps
                  self._recompute_energy(st)
                  E_plus = self.E[st][ib]
                  self.x[st][ib, ia, idim] = x_orig - eps
                  self._recompute_energy(st)
                  E_minus = self.E[st][ib]
                  self.x[st][ib, ia, idim] = x_orig
                  forces[ib, ia, idim] = -(E_plus - E_minus) / (2 * eps)
      self.force[st] = forces

  def _compute_energy_jax(self, st, x, rcell, cell):
      """Pure JAX function: positions -> bond energy (scalar sum)."""
      natom = x.shape[1]
      bdid_full = self.bdid[st]
      
      # Convert numpy scalar params to Python floats for JAX tracing
      p = {k: float(v) for k, v in self.p.items()}
      _log = float(self.log_)
      _botol = float(self.botol)
      
      # 1. Pairwise vectors and distances
      vr = fvr_jnp(x)
      vrf = jnp.matmul(vr, rcell)
      vrf = jnp.where(vrf - 0.5 > 0, vrf - 1.0, vrf)
      vrf = jnp.where(vrf + 0.5 < 0, vrf + 1.0, vrf)
      vr = jnp.matmul(vrf, cell)
      r = jnp.sqrt(jnp.sum(vr * vr, axis=3) + 1e-8)
      
      # 2. Bond orders from distances
      r_bond = r[:, bdid_full[:, 0], bdid_full[:, 1]]
      bop_si = jnp.zeros_like(r)
      bop_pi = jnp.zeros_like(r)
      bop_pp = jnp.zeros_like(r)
      
      for bd_name in self.bonds:
          if self.nbd[st].get(bd_name, 0) == 0:
              continue
          b_start, b_end = self.b[st][bd_name]
          ndx = bdid_full[b_start:b_end]
          bi = ndx[:, 0]
          bj = ndx[:, 1]
          r_bd = r[:, bi, bj]
          
          p_key = 'bo1_' + bd_name
          rr = _log / p[p_key]
          rc_bo = p['rosi_' + bd_name] * jnp.power(rr, 1.0 / p['bo2_' + bd_name])
          frc = jnp.where(jnp.logical_or(r_bd > rc_bo, r_bd <= 0.001), 0.0, 1.0)
          
          bodiv1 = r_bd / p['rosi_' + bd_name]
          bopow1 = jnp.power(bodiv1, p['bo2_' + bd_name])
          eterm1 = (1.0 + _botol) * jnp.exp(p['bo1_' + bd_name] * bopow1) * frc
          
          bodiv2 = r_bd / p['ropi_' + bd_name]
          bopow2 = jnp.power(bodiv2, p['bo4_' + bd_name])
          eterm2 = jnp.exp(p['bo3_' + bd_name] * bopow2) * frc
          
          bodiv3 = r_bd / p['ropp_' + bd_name]
          bopow3 = jnp.power(bodiv3, p['bo6_' + bd_name])
          eterm3 = jnp.exp(p['bo5_' + bd_name] * bopow3) * frc
          
          bsi = taper_jnp(eterm1, rmin=_botol, rmax=2.0 * _botol) * (eterm1 - _botol)
          bpi = taper_jnp(eterm2, rmin=_botol, rmax=2.0 * _botol) * eterm2
          bpp = taper_jnp(eterm3, rmin=_botol, rmax=2.0 * _botol) * eterm3
          
          bop_si = bop_si.at[:, bi, bj].set(bsi)
          bop_si = bop_si.at[:, bj, bi].set(bsi)
          bop_pi = bop_pi.at[:, bi, bj].set(bpi)
          bop_pi = bop_pi.at[:, bj, bi].set(bpi)
          bop_pp = bop_pp.at[:, bi, bj].set(bpp)
          bop_pp = bop_pp.at[:, bj, bi].set(bpp)
      
      bop = bop_si + bop_pi + bop_pp
      
      # 3. Message passing
      eye = jnp.expand_dims(1.0 - jnp.eye(natom), axis=0)
      H = [bop]
      Hsi = [bop_si]
      Hpi = [bop_pi]
      Hpp = [bop_pp]
      D = [jnp.sum(bop, axis=2)]
      
      for t in range(1, self.messages + 1):
          Di = jnp.expand_dims(D[t - 1], 2) * eye
          Dj = jnp.expand_dims(D[t - 1], 1) * eye
          Dbi = Di - H[t - 1]
          Dbj = Dj - H[t - 1]
          
          Dbi_ = Dbi[:, bdid_full[:, 0], bdid_full[:, 1]]
          Dbj_ = Dbj[:, bdid_full[:, 0], bdid_full[:, 1]]
          H_ = H[t - 1][:, bdid_full[:, 0], bdid_full[:, 1]]
          Hsi_ = Hsi[t - 1][:, bdid_full[:, 0], bdid_full[:, 1]]
          Hpi_ = Hpi[t - 1][:, bdid_full[:, 0], bdid_full[:, 1]]
          Hpp_ = Hpp[t - 1][:, bdid_full[:, 0], bdid_full[:, 1]]
          
          bo_new = jnp.zeros_like(r)
          bosi_new = jnp.zeros_like(r)
          bopi_new = jnp.zeros_like(r)
          bopp_new = jnp.zeros_like(r)
          
          for bd_name in self.bonds:
              if self.nbd[st].get(bd_name, 0) == 0:
                  continue
              b_start, b_end = self.b[st][bd_name]
              ndx = bdid_full[b_start:b_end]
              bi = ndx[:, 0]
              bj = ndx[:, 1]
              bd_parts = bd_name.split('-')
              
              h = H_[:, b_start:b_end]
              hsi = Hsi_[:, b_start:b_end]
              hpi = Hpi_[:, b_start:b_end]
              hpp = Hpp_[:, b_start:b_end]
              dbi = Dbi_[:, b_start:b_end]
              dbj = Dbj_[:, b_start:b_end]
              
              Fi = fmessage_jnp('fm', bd_parts[0], [dbi, h, dbj], self.m, layer=self.mf_layer[1])
              Fj = fmessage_jnp('fm', bd_parts[1], [dbj, h, dbi], self.m, layer=self.mf_layer[1])
              F = Fi * Fj
              
              Fsi = F[..., 0]
              Fpi = F[..., 1]
              Fpp = F[..., 2]
              
              bo_new = bo_new.at[:, bi, bj].set(hsi * Fsi)
              bo_new = bo_new.at[:, bj, bi].set(hsi * Fsi)
              bosi_new = bosi_new.at[:, bi, bj].set(hsi * Fsi)
              bosi_new = bosi_new.at[:, bj, bi].set(hsi * Fsi)
              bopi_new = bopi_new.at[:, bi, bj].set(hpi * Fpi)
              bopi_new = bopi_new.at[:, bj, bi].set(hpi * Fpi)
              bopp_new = bopp_new.at[:, bi, bj].set(hpp * Fpp)
              bopp_new = bopp_new.at[:, bj, bi].set(hpp * Fpp)
          
          H.append(bo_new)
          Hsi.append(bosi_new)
          Hpi.append(bopi_new)
          Hpp.append(bopp_new)
          D.append(jnp.sum(bo_new, axis=2))
      
      # 4. Bond energy
      bo0 = H[-1]
      bosi = Hsi[-1]
      bopi = Hpi[-1]
      bopp = Hpp[-1]
      
      bosi_bond = bosi[:, bdid_full[:, 0], bdid_full[:, 1]]
      bopi_bond = bopi[:, bdid_full[:, 0], bdid_full[:, 1]]
      bopp_bond = bopp[:, bdid_full[:, 0], bdid_full[:, 1]]
      
      ebd = jnp.zeros_like(r)
      for bd_name in self.bonds:
          if self.nbd[st].get(bd_name, 0) == 0:
              continue
          b_start, b_end = self.b[st][bd_name]
          ndx = bdid_full[b_start:b_end]
          bi = ndx[:, 0]
          bj = ndx[:, 1]
          
          bosi_ = bosi_bond[:, b_start:b_end]
          bopi_ = bopi_bond[:, b_start:b_end]
          bopp_ = bopp_bond[:, b_start:b_end]
          
          if self.EnergyFunction == 0:
              FBO = jnp.where(bosi_ > 0.0, 1.0, 0.0)
              FBOR = 1.0 - FBO
              powb = jnp.power(bosi_ + FBOR, p['be2_' + bd_name])
              expb = jnp.exp(p['be1_' + bd_name] * (1.0 - powb))
              sieng = p['Desi_' + bd_name] * bosi_ * expb * FBO
              pieng = p['Depi_' + bd_name] * bopi_
              ppeng = p['Depp_' + bd_name] * bopp_
              esi = sieng + pieng + ppeng
          else:
              esi = fnn_jnp('fe', bd_name, [bosi_, bopi_, bopp_], self.m, layer=self.be_layer[1])
              esi = p['Desi_' + bd_name] * esi
          
          ebd = ebd.at[:, bi, bj].set(-esi)
          ebd = ebd.at[:, bj, bi].set(-esi)
      
      ebond = jnp.sum(ebd, axis=(1, 2))
      return jnp.sum(ebond)

  def get_bond_energy(self,st):
      vr          = fvr(self.x[st])
      vrf         = np.matmul(vr,self.rcell[st])
      vrf         = np.where(vrf-0.5>0,vrf-1.0,vrf)
      vrf         = np.where(vrf+0.5<0,vrf+1.0,vrf) 
      self.vr[st] = np.matmul(vrf,self.cell[st])
      self.r[st]  = np.sqrt(np.sum(self.vr[st]*self.vr[st],axis=3) + 0.00000001) # 
      
      self.get_bondorder_uc(st)
      self.message_passing(st)
      self.get_final_state(st)
      
      self.ebd[st] = np.zeros_like(self.bosi[st])
      self.esi[st] = {}
      bosi = self.bosi[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]
      bopi = self.bopi[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]
      bopp = self.bopp[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]

      for bd in self.bonds:
          nbd_ = self.nbd[st][bd]
          if nbd_==0:
             continue
          b_  = self.b[st][bd]
          bi   = self.bdid[st][b_[0]:b_[1],0]
          bj   = self.bdid[st][b_[0]:b_[1],1]

          bosi_ = bosi[:,b_[0]:b_[1]]
          bopi_ = bopi[:,b_[0]:b_[1]]
          bopp_ = bopp[:,b_[0]:b_[1]]
          if self.EnergyFunction==0:
             FBO  = np.where(np.greater(bosi_,0.0),1.0,0.0)
             FBOR = 1.0 - FBO
             powb = np.power(bosi_+FBOR,self.p['be2_'+bd])
             expb = np.exp(np.multiply(self.p['be1_'+bd],1.0-powb))

             sieng = self.p['Desi_'+bd]*bosi_*expb*FBO 
             pieng = np.multiply(self.p['Depi_'+bd],bopi_)
             ppeng = np.multiply(self.p['Depp_'+bd],bopp_) 
             self.esi[st][bd]  =  sieng + pieng + ppeng
             self.ebd[st][:,bi,bj] = -self.esi[st][bd]
          else:
            self.esi[st][bd] = fnn('fe',bd,[bosi_,bopi_,bopp_],self.m,layer=self.be_layer[1])
            self.ebd[st][:,bi,bj] = -self.p['Desi_'+bd]*self.esi[st][bd]
          # Ebd.append(self.ebd[mol][bd])
      # self.ebd[st][:,self.bdid[st][:,0],self.bdid[st][:,1]] = np.concatenate(ebd,axis=1)
      self.ebond[st]= np.sum(self.ebd[st],axis=(1, 2),keepdims=False)
      # self.ebond[st]= np.squeeze(self.ebond[st],2)
  
  def get_bondorder_uc(self,st):
      bop_si,bop_pi,bop_pp = [],[],[]
      # print(self.r[st])
      r = self.r[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]
      self.bop_si[st] = np.zeros_like(self.r[st])
      self.bop_pi[st] = np.zeros_like(self.r[st])
      self.bop_pp[st] = np.zeros_like(self.r[st])
      self.rbd[st]    = {}
      for bd in self.bonds:
          nbd_ = self.nbd[st][bd]
          b_   = self.b[st][bd]
          if nbd_==0:
             continue
              
          rr   = self.log_/self.p['bo1_'+bd] 
          self.rc_bo[bd]=self.p['rosi_'+bd]*np.power(rr,1.0/self.p['bo2_'+bd])
              
          self.rbd[st][bd] = r[:,b_[0]:b_[1]]
          self.frc[bd] = np.where(np.logical_or(np.greater(self.rbd[st][bd],self.rc_bo[bd]),
                                                np.less_equal(self.rbd[st][bd],0.001)), 0.0,1.0)

          bodiv1 = np.divide(self.rbd[st][bd],self.p['rosi_'+bd])
          bopow1 = np.power(bodiv1,self.p['bo2_'+bd])
          eterm1 = (1.0+self.botol)*np.exp(np.multiply(self.p['bo1_'+bd],bopow1))*self.frc[bd] 

          bodiv2 = np.divide(self.rbd[st][bd],self.p['ropi_'+bd])
          bopow2 = np.power(bodiv2,self.p['bo4_'+bd])
          eterm2 = np.exp(np.multiply(self.p['bo3_'+bd],bopow2))*self.frc[bd]

          bodiv3 = np.divide(self.rbd[st][bd],self.p['ropp_'+bd])
          bopow3 = np.power(bodiv3,self.p['bo6_'+bd])
          eterm3 = np.exp(np.multiply(self.p['bo5_'+bd],bopow3))*self.frc[bd]

          bop_si.append(taper(eterm1,rmin=self.botol,rmax=2.0*self.botol)*(eterm1-self.botol)) # consist with GULP
          bop_pi.append(taper(eterm2,rmin=self.botol,rmax=2.0*self.botol)*eterm2)
          bop_pp.append(taper(eterm3,rmin=self.botol,rmax=2.0*self.botol)*eterm3)
      
      bosi_ = np.concatenate(bop_si,axis=1)
      bopi_ = np.concatenate(bop_pi,axis=1)
      bopp_ = np.concatenate(bop_pp,axis=1)

      self.bop_si[st][:,self.bdid[st][:,0],self.bdid[st][:,1]] = self.bop_si[st][:,self.bdid[st][:,1],self.bdid[st][:,0]] = bosi_
      self.bop_pi[st][:,self.bdid[st][:,0],self.bdid[st][:,1]] = self.bop_pi[st][:,self.bdid[st][:,1],self.bdid[st][:,0]] = bopi_
      self.bop_pp[st][:,self.bdid[st][:,0],self.bdid[st][:,1]] = self.bop_pp[st][:,self.bdid[st][:,1],self.bdid[st][:,0]] = bopp_
      self.bop[st]    = self.bop_si[st] + self.bop_pi[st] + self.bop_pp[st]
      # print(self.bop[st].size)
      self.Deltap[st] = np.sum(self.bop[st],2)
      self.D_si[st]   = np.sum(self.bop_si[st],2)
      self.D_pi[st]   = np.sum(self.bop_pi[st],2)
      self.D_pp[st]   = np.sum(self.bop_pp[st],2)

  def get_bondorder(self,st,Dbi,H,Dbj,Hsi,Hpi,Hpp):
      ''' compute bond-order according the message function'''
      flabel  = 'fm'
      bosi = np.zeros_like(self.r[st])
      bopi = np.zeros_like(self.r[st])
      bopp = np.zeros_like(self.r[st])

      bosi_ = []
      bopi_ = []
      bopp_ = []
      for bd in self.bonds:
          nbd_ = self.nbd[st][bd]
          if nbd_==0:
             continue
          b_   = self.b[st][bd]

          bi   = self.bdid[st][b_[0]:b_[1],0]
          bj   = self.bdid[st][b_[0]:b_[1],1]

          h    = H[:,b_[0]:b_[1]]
          hsi  = Hsi[:,b_[0]:b_[1]]
          hpi  = Hpi[:,b_[0]:b_[1]]
          hpp  = Hpp[:,b_[0]:b_[1]]
          b    = bd.split('-')

          self.Dbi[st][bd]  = Dbi[:,b_[0]:b_[1]] 
          self.Dbj[st][bd]  = Dbj[:,b_[0]:b_[1]]

          Fi   = fmessage(flabel,b[0],[self.Dbi[st][bd],h,self.Dbj[st][bd]],self.m,layer=self.mf_layer[1])
          Fj   = fmessage(flabel,b[1],[self.Dbj[st][bd],h,self.Dbi[st][bd]],self.m,layer=self.mf_layer[1])
          F    = Fi*Fj

          Fsi,Fpi,Fpp = [F[..., i] for i in range(3)]

          bosi_.append(hsi*Fsi)
          bopi_.append(hpi*Fpi)
          bopp_.append(hpp*Fpp)

          bosi[:,bi,bj] = bosi[:,bj,bi] = hsi*Fsi
          bopi[:,bi,bj] = bopi[:,bj,bi] = hpi*Fpi
          bopp[:,bi,bj] = bopp[:,bj,bi] = hpp*Fpp

      bo   = bosi+bopi+bopp
      return bo,bosi,bopi,bopp
  
  def message_passing(self,st):
      self.H[st]    = [self.bop[st]]                     #
      self.Hsi[st]  = [self.bop_si[st]]                  #
      self.Hpi[st]  = [self.bop_pi[st]]                  #
      self.Hpp[st]  = [self.bop_pp[st]]                  #
      self.D[st]    = [self.Deltap[st]]    
      
      for t in range(1,self.messages+1):
          Di   = np.expand_dims(self.D[st][t-1],2)*self.eye[st]
          Dj   = np.expand_dims(self.D[st][t-1],1)*self.eye[st]

          Dbi  = Di  - self.H[st][t-1] 
          Dbj  = Dj  - self.H[st][t-1]

          Dbi_ = Dbi[:,self.bdid[st][:,0],self.bdid[st][:,1]]
          Dbj_ = Dbj[:,self.bdid[st][:,0],self.bdid[st][:,1]]
          H    = self.H[st][t-1][:,self.bdid[st][:,0],self.bdid[st][:,1]]
          Hsi  = self.Hsi[st][t-1][:,self.bdid[st][:,0],self.bdid[st][:,1]]
          Hpi  = self.Hpi[st][t-1][:,self.bdid[st][:,0],self.bdid[st][:,1]]
          Hpp  = self.Hpp[st][t-1][:,self.bdid[st][:,0],self.bdid[st][:,1]]

          bo,bosi,bopi,bopp = self.get_bondorder(st,Dbi_,H,Dbj_,Hsi,Hpi,Hpp)
          
          self.H[st].append(bo)                     # get the hidden state H[t]
          self.Hsi[st].append(bosi)
          self.Hpi[st].append(bopi)
          self.Hpp[st].append(bopp)

          Delta = np.sum(bo,2)
          self.D[st].append(Delta)                  # degree matrix

  def get_final_state(self,st):     
      self.Delta[st]  = self.D[st][-1]
      self.bo0[st]    = self.H[st][-1]              # fetch the final state 
      self.bosi[st]   = self.Hsi[st][-1]
      self.bopi[st]   = self.Hpi[st][-1]
      self.bopp[st]   = self.Hpp[st][-1]

      self.bo[st]     = np.maximum(0, self.bo0[st] - self.p['acut'])

      bso             = []
      bo0             = self.bo0[st][:,self.bdid[st][:,0],self.bdid[st][:,1]]

      for bd in self.bonds:
          if self.nbd[st][bd]==0:
             continue
          b_   = self.b[st][bd]
          bo0_ = bo0[:,b_[0]:b_[1]]
          bso.append(self.p['ovun1_'+bd]*self.p['Desi_'+bd]*bo0_)
      
      bso_   = np.zeros_like(self.bo0[st])
      bso_[:,self.bdid[st][:,0],self.bdid[st][:,1]]  = np.concatenate(bso,1)
      bso_[:,self.bdid[st][:,1],self.bdid[st][:,0]]  = np.concatenate(bso,1)

      self.SO[st]      = np.sum(bso_,2)  
      self.Delta_pi[st]= self.bopi[st]+self.bopp[st]
      self.delta_pi[st]= np.sum(self.Delta_pi[st],2) 

      self.fbot[st]   = taper(self.bo0[st],rmin=self.p['acut'],rmax=2.0*self.p['acut']) 
      self.fhb[st]    = taper(self.bo0[st],rmin=self.hbtol,rmax=2.0*self.hbtol) 
  
  def get_atomic_energy(self,st):
      ''' compute atomic energy of structure (st): elone, eover,eunder'''
      # st_ = st.split('-')[0]
      self.Elone[st]     = np.zeros_like(self.Delta[st])
      self.Eover[st]     = np.zeros_like(self.Delta[st])
      self.Eunder[st]    = np.zeros_like(self.Delta[st])
      self.Nlp[st]       = np.zeros_like(self.Delta[st])
      self.Delta_ang[st] = np.zeros_like(self.Delta[st])
      Dlp                = np.zeros_like(self.Delta[st])

      delta       = {}
      delta_pi    = {}
      delta_lp    = {}
      so          = {}

      for sp in self.spec:
          delta[sp]    = self.Delta[st][:,self.s[st][sp]]
          delta_pi[sp] = self.Delta_pi[st][:,self.s[st][sp]]
          so[sp]       = self.SO[st][:,self.s[st][sp]]
          
          delta_lp[sp],nlp,dlp,Elone        = self.get_elone(sp,delta[sp]) 
          self.Nlp[st][:,self.s[st][sp]]    = nlp
          self.Elone[st][:,self.s[st][sp]]  = Elone
          Dlp[:,self.s[st][sp]]             = dlp

      for sp in self.spec:
          # print(delta_pi.shape,dlp.shape)
          dpi                 = np.sum(delta_pi[sp]*np.expand_dims(Dlp,1), 2)
          # print(dpi)
          delta_lpcorr,Eover  = self.get_eover(sp,delta[sp],delta_lp[sp],dpi,so[sp]) 
          Eunder              = self.get_eunder(sp,delta_lpcorr,dpi) 
          delta_ang           = delta[sp] - self.p['valang_'+sp]
          
          self.Delta_ang[st][:,self.s[st][sp]]  = delta_ang
          self.Eover[st][:,self.s[st][sp]]      = Eover
          self.Eunder[st][:,self.s[st][sp]]     = Eunder

      self.elone[st]  = np.sum(self.Elone[st],1)
      self.eover[st]  = np.sum(self.Eover[st],1)
      self.eunder[st] = np.sum(self.Eunder[st],1)

      self.eatomic[st] = np.array(0.0)
      for sp in self.spec:
          if self.ns[st][sp]>0:
             self.eatomic[st] -= self.p['atomic_'+sp]*self.ns[st][sp]
      self.zpe[st]    = self.eatomic[st] + self.estruc[st]
  
  def get_eover(self,sp,delta,delta_lp,dpi,so):
      delta_val    = delta - self.p['val_'+sp]
      delta_lpcorr = delta_val - np.divide(delta_lp,
                     1.0+self.p['ovun3']*np.exp(self.p['ovun4']*dpi))
      otrm1              = DIV_IF(1.0,delta_lpcorr + self.p['val_'+sp])
      otrm2              = sigmoid(-self.p['ovun2_'+sp]*delta_lpcorr)
      Eover              = so*otrm1*delta_lpcorr*otrm2
      return delta_lpcorr,Eover 
  
  def get_eunder(self,sp,delta_lpcorr,dpi):
      expeu1            = np.exp(self.p['ovun6']*delta_lpcorr)
      eu1               = sigmoid(self.p['ovun2_'+sp]*delta_lpcorr)
      expeu3            = np.exp(self.p['ovun8']*dpi)
      eu2               = np.divide(1.0,1.0+self.p['ovun7']*expeu3)
      Eunder            = -self.p['ovun5_'+sp]*(1.0-expeu1)*eu1*eu2                          # must positive
      return Eunder 
  
  def get_elone(self,sp,delta):
      Nlp            = 0.5*(self.p['vale_'+sp] - self.p['val_'+sp])
      delta_e        = 0.5*(delta - self.p['vale_'+sp])
      De             = -np.maximum(0, -np.ceil(delta_e)) 
      nlp            = -De + np.exp(-self.p['lp1']*4.0*np.square(1.0+delta_e-De))

      delta_lp       = Nlp - nlp           
      # Delta_lp     = np.maximum(0, Delta_lp+1) -1
      dlp            = delta -self.p['val_'+sp] - delta_lp

      explp          = 1.0+np.exp(-75.0*delta_lp) # -self.p['lp3']
      Elone          = self.p['lp2_'+sp]*delta_lp/explp
      return delta_lp,nlp,dlp,Elone
  
  def get_threebody_energy(self,st):
      ''' compute three-body term interaction '''
      PBOpow        = -np.power(self.bo[st]+0.00000001,8)        # original: self.BO0 
      PBOexp        =  np.exp(PBOpow)
      self.Pbo[st]  =  np.prod(PBOexp,2)     # BO Product

      if self.nang[st]==0:
         self.eang[st] = np.zeros(self.batch[st]) 
         self.epen[st] = np.zeros(self.batch[st]) 
         self.etcon[st]= np.zeros(self.batch[st]) 
      else:
         Eang  = []
         Epen  = []
         Etcon = []
         for ang in self.angs:
             sp  = ang.split('-')[1]
             # print(ang,self.na[st].get(ang,0))
             if self.na[st].get(ang,0)>0:
                ai        = np.ravel(self.ang_i[st][self.a[st][ang][0]:self.a[st][ang][1]])
                aj        = np.ravel(self.ang_j[st][self.a[st][ang][0]:self.a[st][ang][1]])
                ak        = np.ravel(self.ang_k[st][self.a[st][ang][0]:self.a[st][ang][1]])
                # print('\n ai \n',ai)  
                boij      = self.bo[st][:,ai,aj]
                bojk      = self.bo[st][:,aj,ak]
                fij       = self.fbot[st][:,ai,aj]
                fjk       = self.fbot[st][:,aj,ak]
                delta     = self.Delta[st][:,aj] - self.p['val_'+sp]
                delta_ang = self.Delta_ang[st][:,aj]
                delta_i   = self.Delta[st][:,ai]
                delta_k   = self.Delta[st][:,ak]
                sbo       = self.delta_pi[st][:,aj]
                pbo       = self.Pbo[st][:,aj]
                nlp       = self.Nlp[st][:,aj]
                # theta   = self.theta[st][:,self.a[st][ang][0]:self.a[st][ang][1]]
                theta     = self.get_theta(st,ai,aj,ak)
                Ea,fijk   = self.get_eangle(sp,ang,boij,bojk,fij,fjk,theta,delta_ang,sbo,pbo,nlp)
                Ep        = self.get_epenalty(ang,delta,boij,bojk,fijk)
                Et        = self.get_three_conj(ang,delta_ang,delta_i,delta_k,boij,bojk,fijk) 
                Eang.append(Ea)
                Epen.append(Ep)
                Etcon.append(Et)

         self.Eang[st] = np.concatenate(Eang,axis=1)
         self.Epen[st] = np.concatenate(Epen,axis=1)
         self.Etcon[st]= np.concatenate(Etcon,axis=1)
         self.eang[st] = np.sum(self.Eang[st],1)
         self.epen[st] = np.sum(self.Epen[st],1)
         self.etcon[st]= np.sum(self.Etcon[st],1)

  def get_theta(self,st,ai,aj,ak):
      Rij = self.r[st][:,ai,aj]  
      Rjk = self.r[st][:,aj,ak]  
      # Rik = self.r[self.angi,self.angk]  
      vik = self.vr[st][:,ai,aj] + self.vr[st][:,aj,ak]
      # print(vik.shape)
      Rik = np.sqrt(np.sum(np.square(vik),2))

      Rij2= Rij*Rij
      Rjk2= Rjk*Rjk
      Rik2= Rik*Rik

      cos_theta = (Rij2+Rjk2-Rik2)/(2.0*Rij*Rjk)
      cos_theta = np.where(cos_theta>0.9999999,0.9999999,cos_theta)   
      cos_theta = np.where(cos_theta<-0.9999999,-0.9999999,cos_theta)
      theta     = np.arccos(cos_theta)
      return theta

  def get_eangle(self,sp,ang,boij,bojk,fij,fjk,theta,delta_ang,sbo,pbo,nlp):
      fijk           = fij*fjk

      theta0         = self.get_theta0(ang,delta_ang,sbo,pbo,nlp)
      thet           = theta0 - theta
      thet2          = np.square(thet)

      expang         = np.exp(-self.p['val2_'+ang]*thet2)
      f_7            = self.f7(sp,ang,boij,bojk)
      f_8            = self.f8(sp,ang,delta_ang)
      Eang           = fijk*f_7*f_8*(self.p['val1_'+ang]-self.p['val1_'+ang]*expang) 
      return Eang,fijk

  def get_theta0(self,ang,delta_ang,sbo,pbo,nlp):
      Sbo   = sbo - (1.0-pbo)*(delta_ang+self.p['val8']*nlp)    
      
      ok    = np.logical_and(np.less_equal(Sbo,1.0),np.greater(Sbo,0.0))
      S1    = np.where(ok,Sbo,0.0)    #  0< sbo < 1                  
      Sbo1  = np.where(ok,np.power(S1+0.00000001,self.p['val9']),0.0) 

      ok    = np.logical_and(np.less(Sbo,2.0),np.greater(Sbo,1.0))
      S2    = np.where(ok,Sbo,0.0)                     
      F2    = np.where(ok,1.0,0.0)                 #  1< sbo <2
     
      S2    = 2.0*F2-S2  
      Sbo12 = np.where(ok,2.0-np.power(S2+0.00000001,self.p['val9']),0.0)  #  1< sbo <2
                                                                                      #     sbo >2
      Sbo2  = np.where(np.greater_equal(Sbo,2.0),1.0,0.0)

      Sbo3   = Sbo1 + Sbo12 + 2.0*Sbo2
      theta0_ = 180.0 - self.p['theta0_'+ang]*(1.0-np.exp(-self.p['val10']*(2.0-Sbo3)))
      theta0 = theta0_/57.29577951
      return theta0

  def f7(self,sp,ang,boij,bojk): 
      Fboi  = np.where(np.greater(boij,0.0),1.0,0.0)   
      Fbori = 1.0 - Fboi                                                                         # prevent NAN error
      expij = np.exp(-self.p['val3_'+sp]*np.power(boij+Fbori,self.p['val4_'+ang])*Fboi)

      Fbok  = np.where(np.greater(bojk,0.0),1.0,0.0)   
      Fbork = 1.0 - Fbok 
      expjk = np.exp(-self.p['val3_'+sp]*np.power(bojk+Fbork,self.p['val4_'+ang])*Fbok)
      fi = 1.0 - expij
      fk = 1.0 - expjk
      F  = fi*fk
      return F 

  def f8(self,sp,ang,delta_ang):
      exp6 = np.exp( self.p['val6']*delta_ang)
      exp7 = np.exp(-self.p['val7_'+ang]*delta_ang)
      F    = self.p['val5_'+sp] - (self.p['val5_'+sp]-1.0)*np.divide(2.0+exp6,1.0+exp6+exp7)
      return F

  def get_epenalty(self,ang,delta,boij,bojk,fijk):
      f_9  = self.f9(delta)
      expi = np.exp(-self.p['pen2']*np.square(boij-2.0))
      expk = np.exp(-self.p['pen2']*np.square(bojk-2.0))
      Ep   = self.p['pen1_'+ang]*f_9*expi*expk*fijk
      return Ep

  def f9(self,Delta):
      exp3 = np.exp(-self.p['pen3']*Delta)
      exp4 = np.exp( self.p['pen4']*Delta)
      F = np.divide(2.0+exp3,1.0+exp3+exp4)
      return F

  def get_three_conj(self,ang,delta_ang,delta_i,delta_k,boij,bojk,fijk):
      delta_coa  = delta_ang # self.D_ang[st] + valang - valboc
      expcoa1    = np.exp(self.p['coa2']*delta_coa)

      texp0 = np.divide(self.p['coa1_'+ang],1.0 + expcoa1)  
      texp1 = np.exp(-self.p['coa3']*np.square(delta_i-boij))
      texp2 = np.exp(-self.p['coa3']*np.square(delta_k-bojk))
      texp3 = np.exp(-self.p['coa4']*np.square(boij-1.5))
      texp4 = np.exp(-self.p['coa4']*np.square(bojk-1.5))
      Etc   = texp0*texp1*texp2*texp3*texp4*fijk 
      return Etc

  def get_fourbody_energy(self,st):
      if (not self.opt_term['etor'] and not self.opt_term['efcon']) or self.ntor[st]==0:
         self.etor[st] = np.zeros([self.batch[st]])
         self.efcon[st]= np.zeros([self.batch[st]])
      else:
         Etor   =    []
         Efcon  =    []
         for tor in self.tors:
             if self.nt[st][tor]>0:
                ti        = np.squeeze(self.tor_i[st][self.t[st][tor][0]:self.t[st][tor][1]],axis=1)
                tj        = np.squeeze(self.tor_j[st][self.t[st][tor][0]:self.t[st][tor][1]],axis=1)
                tk        = np.squeeze(self.tor_k[st][self.t[st][tor][0]:self.t[st][tor][1]],axis=1)
                tl        = np.squeeze(self.tor_l[st][self.t[st][tor][0]:self.t[st][tor][1]],axis=1)
                boij      = self.bo[st][:,ti,tj]
                bojk      = self.bo[st][:,tj,tk]
                bokl      = self.bo[st][:,tk,tl]
                bopjk     = self.bopi[st][:,tj,tk]
                fij       = self.fbot[st][:,ti,tj]
                fjk       = self.fbot[st][:,tj,tk]
                fkl       = self.fbot[st][:,tk,tl]
                
                delta_j   = self.Delta_ang[st][:,tj]
                delta_k   = self.Delta_ang[st][:,tk]

                w,cos_w,cos2w,s_ijk,s_jkl = self.get_torsion_angle(st,ti,tj,tk,tl)
                
                Et,fijkl  = self.get_etorsion(tor,boij,bojk,bokl,fij,fjk,fkl,
                                       bopjk,delta_j,delta_k,
                                       w,cos_w,cos2w,
                                       s_ijk,s_jkl)
                Ef        = self.get_four_conj(tor,boij,bojk,bokl,w,s_ijk,s_jkl,fijkl)
                
                Etor.append(Et)
                Efcon.append(Ef)
                # self.etor[st] = self.etor[st]  + np.sum(Et,1)
                # self.efcon[st]= self.efcon[st] + np.sum(Ef,1)
         self.Etor[st] = np.concatenate(Etor,axis=1)
         self.Efcon[st] = np.concatenate(Efcon,axis=1)
         self.etor[st] = np.sum(self.Etor[st],1)
         self.efcon[st]= np.sum(self.Efcon[st],1)

  def get_torsion_angle(self,st,ti,tj,tk,tl):
      ''' compute torsion angle '''
      rij = self.r[st][:,ti,tj]
      rjk = self.r[st][:,tj,tk]
      rkl = self.r[st][:,tk,tl]

      
      vrjk= self.vr[st][:,tj,tk]
      vrkl= self.vr[st][:,tk,tl]

      vrjl= vrjk + vrkl
      rjl = np.sqrt(np.sum(np.square(vrjl),2))

      vrij= self.vr[st][:,ti,tj]
      vril= vrij + vrjl
      ril = np.sqrt(np.sum(np.square(vril),2))

      vrik= vrij + vrjk
      rik = np.sqrt(np.sum(np.square(vrik),2))
      rij2= np.square(rij)
      rjk2= np.square(rjk)
      rkl2= np.square(rkl)
      rjl2= np.square(rjl)
      ril2= np.square(ril)
      rik2= np.square(rik)
      
      c_ijk = (rij2+rjk2-rik2)/(2.0*rij*rjk)
      c2ijk = np.square(c_ijk)
      # tijk  = tf.acos(c_ijk)
      cijk  =  1.00000001 - c2ijk
      s_ijk = np.sqrt(cijk)

      c_jkl = (rjk2+rkl2-rjl2)/(2.0*rjk*rkl)
      c2jkl = np.square(c_jkl)
      cjkl  = 1.00000001  - c2jkl 
      s_jkl = np.sqrt(cjkl)

      # c_ijl = (rij2+rjl2-ril2)/(2.0*rij*rjl)
      c_kjl = (rjk2+rjl2-rkl2)/(2.0*rjk*rjl)

      c2kjl = np.square(c_kjl)
      ckjl  = 1.00000001 - c2kjl 
      s_kjl = np.sqrt(ckjl)

      fz    = rij2+rjl2-ril2-2.0*rij*rjl*c_ijk*c_kjl
      fm    = rij*rjl*s_ijk*s_kjl

      fm    = np.where(np.logical_and(fm<=0.000001,fm>=-0.000001),np.full_like(fm,1.0),fm)
      fac   = np.where(np.logical_and(fm<=0.000001,fm>=-0.000001),np.full_like(fm,0.0),
                                                                        np.full_like(fm,1.0))
      cos_w = 0.5*fz*fac/fm
      #cos_w= cos_w*ccijk*ccjkl
      cos_w = np.where(cos_w>0.9999999,np.full_like(cos_w,0.999999),cos_w)   
      cos_w = np.where(cos_w<-0.9999999,np.full_like(cos_w,-0.999999),cos_w)
      w= np.arccos(cos_w)
      cos2w = np.cos(2.0*w)
      return w,cos_w,cos2w,s_ijk,s_jkl
  
  def get_etorsion(self,tor,boij,bojk,bokl,fij,fjk,fkl,bopjk,delta_j,delta_k,
                        w,cos_w,cos2w,s_ijk,s_jkl):
      fijkl   = fij*fjk*fkl

      f_10    = self.f10(boij,bojk,bokl)
      f_11    = self.f11(delta_j,delta_k)
      expv2   = np.exp(self.p['tor1_'+tor]*np.square(2.0-bopjk-f_11)) 

      cos3w   = np.cos(3.0*w)
      v1      = 0.5*self.p['V1_'+tor]*(1.0+cos_w)
      v2      = 0.5*self.p['V2_'+tor]*expv2*(1.0-cos2w)
      v3      = 0.5*self.p['V3_'+tor]*(1.0+cos3w)
      
      Etor    = fijkl*f_10*s_ijk*s_jkl*(v1+v2+v3)
      return Etor,fijkl
  
  def f10(self,boij,bojk,bokl):
      exp1 = 1.0 - np.exp(-self.p['tor2']*boij)
      exp2 = 1.0 - np.exp(-self.p['tor2']*bojk)
      exp3 = 1.0 - np.exp(-self.p['tor2']*bokl)
      return exp1*exp2*exp3

  def f11(self,delta_j,delta_k):
      delt = delta_j+delta_k
      f11exp3  = np.exp(-self.p['tor3']*delt)
      f11exp4  = np.exp( self.p['tor4']*delt)
      f_11 = np.divide(2.0+f11exp3,1.0+f11exp3+f11exp4)
      return f_11

  def get_four_conj(self,tor,boij,bojk,bokl,w,s_ijk,s_jkl,fijkl):
      exptol= np.exp(-self.p['cot2']*np.square(self.p['acut'] - 1.5))
      expij = np.exp(-self.p['cot2']*np.square(boij-1.5))-exptol
      expjk = np.exp(-self.p['cot2']*np.square(bojk-1.5))-exptol 
      expkl = np.exp(-self.p['cot2']*np.square(bokl-1.5))-exptol

      f_12  = expij*expjk*expkl
      prod  = 1.0+(np.square(np.cos(w))-1.0)*s_ijk*s_jkl
      Efcon = fijkl*f_12*self.p['cot1_'+tor]*prod  
      return Efcon

  def f13(self,st,r):
      # print(self.p['vdw1'].device)
      gammaw = np.sqrt(np.expand_dims(self.P[st]['gammaw'],1)*np.expand_dims(self.P[st]['gammaw'],2))
      rr = np.power(r,self.p['vdw1'])+np.power(np.divide(1.0,gammaw),self.p['vdw1'])
      f_13 = np.power(rr,np.divide(1.0,self.p['vdw1']))  
      return f_13

  def get_tap(self,r):
      tpc = 1.0+np.divide(-35.0,self.vdwcut**4.0)*np.power(r,4.0)+ \
            np.divide(84.0,self.vdwcut**5.0)*np.power(r,5.0)+ \
            np.divide(-70.0,self.vdwcut**6.0)*np.power(r,6.0)+ \
            np.divide(20.0,self.vdwcut**7.0)*np.power(r,7.0)
      return tpc

  def get_vdw_energy(self,st):
      self.Evdw[st]   = np.array(0.0)
      self.Ecoul[st]  = np.array(0.0)
      nc = 0
      cell0 = self.cell[st][:, :, 0]
      cell1 = self.cell[st][:, :, 1]
      cell2 = self.cell[st][:, :, 2]
      self.cell0[st] = np.expand_dims(cell0,1)
      self.cell1[st] = np.expand_dims(cell1,1)
      self.cell2[st] = np.expand_dims(cell2,1)

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
              
      gamma= np.sqrt(np.expand_dims(self.P[st]['gamma'],1)*np.expand_dims(self.P[st]['gamma'],2))
      gm3  = np.power(np.divide(1.0,gamma),3.0)

      for i in range(-1,2):
          for j in range(-1,2):
              for k in range(-1,2):
                  cell = self.cell0[st]*i + self.cell1[st]*j + self.cell2[st]*k
                  vr_  = self.vr[st] + cell
                  r    = np.sqrt(np.sum(np.square(vr_),3)+0.00000001)
                  r3   = np.power(r+0.00000001,3.0)
                  fv_  = np.where(np.logical_and(r>0.0000001,r<=self.vdwcut),1.0,0.0)
                  if nc<13:
                     fv = np.triu(fv_, k=0)
                  else:
                     fv = np.triu(fv_, k=1)

                  f_13  = self.f13(st,r)
                  tp    = self.get_tap(r)

                  expvdw1 = np.exp(0.5*self.P[st]['alfa']*(1.0-np.divide(f_13,2.0*self.P[st]['rvdw'])))
                  expvdw2 = np.square(expvdw1) 
                  self.Evdw[st]  = self.Evdw[st] + fv*tp*self.P[st]['Devdw']*(expvdw2-2.0*expvdw1)
                  rth            = np.power(r3+gm3,1.0/3.0)                                      # ecoul
                  self.Ecoul[st] = self.Ecoul[st] + np.divide(fv*tp*self.q[st],rth)
                  nc += 1
      self.evdw[st]  = np.sum(self.Evdw[st],axis=(1, 2))
      self.ecoul[st] = np.sum(self.Ecoul[st],axis=(1, 2))
  
  def get_hb_energy(self,st):
      self.ehb[st]  = np.array(0.0)
      self.Ehb[st]  = np.array(0.0)
      Ehb           = []
      for hb in self.hbs:
          if self.nhb[st][hb]==0:
             continue     
          bo          = self.bo0[st][:,self.hb_i[st][hb],self.hb_j[st][hb]]
          fhb         = self.fhb[st][:,self.hb_i[st][hb],self.hb_j[st][hb]]

          rij         = self.r[st][:,self.hb_i[st][hb],self.hb_j[st][hb]]
          rij2        = np.square(rij)
          vrij        = self.vr[st][:,self.hb_i[st][hb],self.hb_j[st][hb]]
          vrjk_       = self.vr[st][:,self.hb_j[st][hb],self.hb_k[st][hb]]
          ehb         = 0.0
          for i in range(-1,2):
              for j in range(-1,2):
                  for k in range(-1,2):
                      cell   = self.cell0[st]*i + self.cell1[st]*j + self.cell2[st]*k
                      vrjk   = vrjk_ + cell 
  
                      rjk2   = np.sum(np.square(vrjk),axis=3)
                      rjk    = np.sqrt(rjk2+0.00000001)

                      vrik   = vrij + vrjk
                      rik2   = np.sum(np.square(vrik),axis=3)
                      rik    = np.sqrt(rik2+0.00000001)

                      cos_th = (rij2+rjk2-rik2)/(2.0*rij*rjk)
                      hbthe  = 0.5-0.5*cos_th
                      frhb   = rtaper(rik,rmin=self.hbshort,rmax=self.hblong)

                      exphb1 = 1.0-np.exp(-self.p['hb1_'+hb]*bo)
                      hbsum  = np.divide(self.p['rohb_'+hb],rjk)+np.divide(rjk,self.p['rohb_'+hb])-2.0
                      exphb2 = np.exp(-self.p['hb2_'+hb]*hbsum)
                     
                      sin4   = np.square(hbthe)
                      ehb    = ehb + fhb*frhb*self.p['Dehb_'+hb]*exphb1*exphb2*sin4 
                      # ehb_   = np.squeeze(np.sum(ehb,1),1)
                      #   print('ehb: ',ehb_)  
          Ehb.append(ehb)

      if Ehb:
         self.Ehb[st] = np.squeeze(np.concatenate(Ehb,axis=1),2)
         self.ehb[st] = np.sum(self.Ehb[st],1)

  def get_rcbo(self):
      self.rc_bo = {}
      for bd in self.bonds:
          b= bd.split('-')
          ofd=bd if b[0]!=b[1] else '{:s}-{:s}'.format(b[0],b[0])
          log_ = np.log((self.botol/(1.0 + self.botol)))
          rr = log_/self.p['bo1_'+bd] 
          self.rc_bo[bd]=self.p['rosi_'+ofd]*np.power(rr,1.0/self.p['bo2_'+bd])

  def get_eself(self):
      chi    = np.expand_dims(self.P['chi'],axis=0)
      mu     = np.expand_dims(self.P['mu'],axis=0)
      self.eself = self.q*(chi+self.q*mu)
      self.Eself = np.array(np.sum(self.eself,axis=1))

  def check_hb(self):
      if 'H' in self.spec:
         for sp1 in self.spec:
             if sp1 != 'H':
                for sp2 in self.spec:
                    if sp2 != 'H':
                       hb = sp1+'-H-'+sp2
                       if hb not in self.hbs:
                          self.hbs.append(hb) # 'rohb','Dehb','hb1','hb2'
                          self.p_['rohb_'+hb] = 1.9
                          self.p_['Dehb_'+hb] = 0.0
                          self.p_['hb1_'+hb]  = 2.0
                          self.p_['hb2_'+hb]  = 19.0

  def check_offd(self):
      p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
      for key in p_offd:
          for sp in self.spec:
              self.p_[key+'_'+sp+'-'+sp]  = self.p_[key+'_'+sp]  

      for bd in self.bonds:             # check offd parameters
          b= bd.split('-')
          for key in p_offd:        # set offd parameters according combine rules
              if key+'_'+bd not in self.p_:
                 if self.p_[key+'_'+b[0]]>0.0 and self.p_[key+'_'+b[1]]>0.0:
                    self.p_[key+'_'+bd] = np.sqrt(self.p_[key+'_'+b[0]]*self.p_[key+'_'+b[1]])
                 else:
                    self.p_[key+'_'+bd] = -1.0

      for bd in self.bonds:             # check minus ropi ropp parameters
          if self.p_['ropi_'+bd]<0.0:
             self.p_['ropi_'+bd] = 0.3*self.p_['rosi_'+bd]
             self.p_['bo3_'+bd]  = -50.0
             self.p_['bo4_'+bd]  = 0.0
          if self.p_['ropp_'+bd]<0.0:
             self.p_['ropp_'+bd] = 0.2*self.p_['rosi_'+bd]
             self.p_['bo5_'+bd]  = -50.0
             self.p_['bo6_'+bd]  = 0.0

  def set_p(self):
      ''' setting up parameters '''
      self.unit   = 4.3364432032e-2
      self.punit  = ['Desi','Depi','Depp','lp2','ovun5','val1',
                     'coa1','V1','V2','V3','cot1','pen1','Devdw','Dehb']
      ##  All Parameters
      self.p_bond = ['Desi','ovun1','Depi','Depp',
                     'bo3','bo4','bo1','bo2','bo5','bo6','be2','be1',
                     'Devdw','rvdw','alfa','rosi','ropi','ropp'] # 'corr13','ovcorr'
                     
      self.p_offd = ['Devdw','rvdw','alfa','rosi','ropi','ropp']
      self.p_g    = ['coa2','ovun6','lp1','lp3',                 # 'boc1','boc2',
                     'ovun7','ovun8','val6','tor2',
                     'val8','val9','val10',
                     'tor3','tor4','cot2','coa4','ovun4',
                     'ovun3','val8','coa3','pen2','pen3','pen4',
                     'acut',
                     'vdw1']
      self.p_spec = ['valang','valboc','val','vale',
                     'lp2','ovun5','val3','val5',        # ,'boc3','boc4','boc5'
                     'ovun2','atomic',
                     'mass','chi','mu','gamma','gammaw'] # 'Devdw','rvdw','alfa'
      self.p_ang  = ['theta0','val1','val2','coa1','val7','val4','pen1'] 
      self.p_hb   = ['rohb','Dehb','hb1','hb2']
      self.p_tor  = ['V1','V2','V3','tor1','cot1']

      self.ic = Intelligent_Check(re=self.re,clip=self.clip,spec=self.spec,bonds=self.bonds,
                                  offd=self.offd,angs=self.angs,tors=self.torp,ptor=self.p_tor)
      self.p_,self.m_,cons = self.ic.check(self.p_,self.m_)
      # self.cons.extend(cons)

      self.botol        = 0.01*self.p_['cutoff'] 
      self.log_         = np.array(-9.21044036697651)
      self.hbtol        = self.p_['hbtol'] # np.array(self.p_['hbtol']) 
      self.check_offd()
      # self.check_hb()
      self.check_tors()
      self.get_data()

      sp_opt = set()
      bd_opt = set()
      ang_opt= set()
      tor_opt= set()
      hb_opt = set()
      for bd in self.spec:
          for st in self.strcs:
              if self.ns[st].get(bd,0)>0:        
                 sp_opt.add(bd)
                 break
      for bd in self.bonds:
          for st in self.strcs:
              if self.nbd[st].get(bd,0)>0:        
                 bd_opt.add(bd)
                 break
      for bd in self.angs:
          for st in self.strcs:
              if self.na[st].get(bd,0)>0:        
                 ang_opt.add(bd)
                 break
      for bd in self.tors:
          for st in self.strcs:
              if self.nt[st].get(bd,0)>0:        
                 tor_opt.add(bd)
                 break
      for bd in self.hbs:
          for st in self.strcs:
              if self.nhb[st].get(bd,0)>0:        
                 hb_opt.add(bd)
                 break

      if self.opt is None:
         self.opt = []
         for key in self.p_g:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_spec:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_bond:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_ang:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_tor:
             if key not in self.cons:
                self.opt.append(key)
         for key in self.p_hb:
             if key not in self.cons:
                self.opt.append(key)

      self.p            = {}
      for key in self.p_g:
          unit_ = self.unit if key in self.punit else 1.0
          if key in self.opt:
             self.pp[key]= np.array(self.p_[key]*unit_)
             self.p[key] = self.pp[key]
          else:
             self.p[key] = np.array(self.p_[key]*unit_)

      # self.atol         = np.clip(self.p['acut'],min=self.p_['acut']*0.96)        # atol
      self.p['acut'] = np.clip(self.p['acut'],min=self.p_['acut']*0.95,max=self.p_['acut']*1.05)

      for key in self.p_spec:
          unit_ = self.unit if key in self.punit else 1.0
          for sp in self.spec:
              key_ = key+'_'+sp
              if (key in self.opt or key_ in self.opt) and sp in sp_opt:
                 self.pp[key_]= np.array(self.p_[key_]*unit_)
                 self.p[key_]  = self.pp[key_]
              else:
                 self.p[key_] = np.array(self.p_[key_]*unit_)
      
      for key in self.p_bond:
          unit_ = self.unit if key in self.punit else 1.0
          for bd in self.bonds:
              key_ = key+'_'+bd
              if (key in self.opt or key_ in self.opt) and bd in bd_opt:
                 self.pp[key_]= np.array(self.p_[key+'_'+bd]*unit_)
                 self.p[key_] = self.pp[key_]
              else:
                 self.p[key_] = np.array(self.p_[key+'_'+bd]*unit_)

      for key in self.p_ang:
          unit_ = self.unit if key in self.punit else 1.0
          for a in self.angs:
              key_ = key + '_' + a
              if (key in self.opt or key_ in self.opt) and a in ang_opt:
                 self.pp[key_]= np.array(self.p_[key_]*unit_)
                 self.p[key_] = self.pp[key_]
              else:
                 self.p[key_] = np.array(self.p_[key_]*unit_)

      for key in self.p_tor:
          unit_ = self.unit if key in self.punit else 1.0
          for t in self.tors:
              key_ = key + '_' + t
              if (key in self.opt or key_ in self.opt) and t in tor_opt:
                 self.pp[key_] = np.array(self.p_[key_]*unit_)
                 self.p[key_] = self.pp[key_]
              else:
                 self.p[key_] = np.array(self.p_[key_]*unit_)  

      for key in self.p_hb:
          unit_ = self.unit if key in self.punit else 1.0
          for h in self.hbs:
              key_ = key + '_' + h
              if (key in self.opt or key_ in self.opt) and h in hb_opt:
                 self.pp[key_]= np.array(self.p_[key_]*unit_)
                 self.p[key_]  = self.pp[key_]
              else:
                 self.p[key_] = np.array(self.p_[key_]*unit_)
      # self.get_rcbo()
      if self.nn:
         self.set_m()

  def clamp(self):
      ''' clamp parameters '''
      for k in self.pp:
          key = k.split('_')[0]
          unit_ = self.unit if key in self.punit else 1.0 
          if k in self.ic.clip:
             self.pp[k] = np.clip(self.pp[k],min=self.ic.clip[k][0]*unit_,max=self.ic.clip[k][1]*unit_)
          elif key in self.ic.clip:
             self.pp[k] = np.clip(self.pp[k],min=self.ic.clip[key][0]*unit_,max=self.ic.clip[key][1]*unit_)

  def init_bonds(self):
      self.bonds,self.offd,self.angs,self.torp,self.hbs = [],[],[],[],[]
      self.spec = []
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
             self.spec.append(k[1])
      self.torp = self.checkTors(self.torp)  

  def checkTors(self,torp):
      tors_ = torp
      for tor in tors_:
          [t1,t2,t3,t4] = tor.split('-')
          tor1 = t1+'-'+t3+'-'+t2+'-'+t4
          tor2 = t4+'-'+t3+'-'+t2+'-'+t1
          tor3 = t4+'-'+t2+'-'+t3+'-'+t1

          if tor1 in torp and tor1!=tor:
             # print('-  dict %s is repeated, deleteting ...' %tor1)
             torp.remove(tor1)
          elif tor2 in self.torp and tor2!=tor:
             # print('-  dict %s is repeated, deleteting ...' %tor2)
             torp.remove(tor2)
          elif tor3 in self.torp and tor3!=tor:
             # print('-  dict %s is repeated, deleteting ...' %tor3)
             torp.remove(tor3)  
      return torp 

  def check_tors(self):
      '''  check torsion parameter  '''
      fm = open('manybody.log','w')
      print('  The following manybody interaction are not considered, because no parameter in the ffield: ',file=fm)
      print('---------------------------------------------------------------------------------------------',file=fm)
      if not self.tors:
         self.tors = check_torsion(self.spec,self.torp)
      
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
                    self.p_[key+'_'+tor] = self.p_[key+'_'+tor1] # consistent with lammps
                 elif tor2 in self.torp:
                    self.p_[key+'_'+tor] = self.p_[key+'_'+tor2]
                 elif tor3 in self.torp:
                    self.p_[key+'_'+tor] = self.p_[key+'_'+tor3]    
                 elif tor4 in self.torp:
                    self.p_[key+'_'+tor] = self.p_[key+'_'+tor4]  
                 elif tor5 in self.torp:
                    self.p_[key+'_'+tor] = self.p_[key+'_'+tor5]     
                 else:
                    print('-  an error case for {:s},'.format(tor),self.spec,file=fm)
      fm.close()

  def stack_tensor(self):
      self.x     = {}
      self.rcell = {}
      self.cell  = {}
      self.cell0 = {}
      self.cell1 = {}
      self.cell2 = {}
      self.eye   = {}
      self.P     = {}
      self.pmask = {}
      self.vb_i  = {}
      self.vb_j  = {}
      for st in self.strcs:
          self.x[st]     = np.array(self._data[st].x)
          self.cell[st]  = np.array(np.expand_dims(self._data[st].cell,axis=1))
          self.rcell[st] = np.array(np.expand_dims(self._data[st].rcell,axis=1))
          self.eye[st]   = np.array(np.expand_dims(1.0 - np.eye(self.natom[st]),axis=0))
          self.P[st]     = {}
          self.vb_i[st]  = {bd:[] for bd in self.bonds}
          self.vb_j[st]  = {bd:[] for bd in self.bonds}
         
          for i in range(self.natom[st]):
              for j in range(self.natom[st]):
                  bd = self.atom_name[st][i]+'-'+self.atom_name[st][j]
                  if bd not in self.bonds:
                     bd = self.atom_name[st][j]+'-'+self.atom_name[st][i]
                  self.vb_i[st][bd].append(i)
                  self.vb_j[st][bd].append(j)
   
          self.pmask[st] = {}
          for sp in self.spec:
              pmask = np.zeros([1,self.natom[st]])
              pmask[:,self.s[st][sp]] = 1.0
              self.pmask[st][sp] = np.array(pmask)

          for bd in self.bonds:
             if len(self.vb_i[st][bd])==0:
                continue
             pmask = np.zeros([1,self.natom[st],self.natom[st]])
             pmask[:,self.vb_i[st][bd],self.vb_j[st][bd]] = 1.0
             self.pmask[st][bd] = np.array(pmask)

  def get_data(self): 
      self.nframe      = 0
      strucs           = {}
      self.max_e       = {}
      # self.cell        = {}
      self.strcs       = []
      self.batch       = {}
      self.eself,self.evdw_,self.ecoul_ = {},{},{}

      if self.dataset:
         dataset = self.dataset
      else:
         dataset = self.data

      for st in dataset: 
          if st not in self.data:
             # if st in self.batch_size:
             #    batch_ = self.batch_size[st]
             # else:
             #    batch_ = self.batch_size['others']
             data_ = reax_force_data(structure=st,
                                 traj=self.dataset[st],
                               vdwcut=self.vdwcut,
                                 rcut=self.rcut,
                                rcuta=self.rcuta,
                              hbshort=self.hbshort,
                               hblong=self.hblong,
                                # batch=batch_,
                       variable_batch=True,
                               sample=self.sample,
                                    m=self.m_,
                             mf_layer=self.mf_layer_,
                       p=self.p_,spec=self.spec,bonds=self.bonds,
                  angs=self.angs,tors=self.tors,
                                  hbs=self.hbs,
                               screen=self.screen)
             self.data[st] = data_
          else:
             data_ = self.data[st]

          if data_.status:
             self.strcs.append(st)
             strucs[st]        = data_
             self.batch[st]    = strucs[st].batch
             self.nframe      += self.batch[st]
             # print('-  max energy of %s: %f.' %(st,strucs[st].max_e))
             self.max_e[st]    = strucs[st].max_e
             # self.evdw_[st]  = strucs[st].evdw
             # self.ecoul_[st] = strucs[st].ecoul  
             # self.cell[st]   = strucs[st].cell
             st_ = st.split('-')[0]
             if 'all' in self._device:
                 self.device[st] = (self._device['all'])
             elif st in self._device:
                 self.device[st] = (self._device[st])
             elif st_ in self._device:
                 self.device[st] = (self._device[st_])
             else:
                 self.device[st] = (self._device['others'])
             if st not in self.weight_force:
                if st_ in self.weight_force:
                   self.weight_force[st] = self.weight_force[st_]
                else:
                   self.weight_force[st] = 1
          else:
             print('-  data status of {:s}: '.format(st),data_.status)
      self.nstrc  = len(strucs)
      self.generate_data(strucs)
      # self.memory(molecules=strucs)
      # print('-  generating dataset ...')
      return strucs

  def generate_data(self,strucs):
      ''' get data '''
      self.dft_energy                  = {}
      self.dft_forces                  = {}
      self.q                           = {}
      self.bdid                        = {}
      self.dilink,self.djlink          = {},{}
      self.nbd,self.b,self.a,self.t    = {},{},{},{}
      self.ang_i,self.ang_j,self.ang_k = {},{},{}
      self.abij,self.abjk              = {},{}
      self.tij,self.tjk,self.tkl       = {},{},{}
      self.tor_j,self.tor_k            = {},{}
      self.tor_i,self.tor_l            = {},{}
      self.atom_name                   = {}
      self.natom                       = {}
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
      self.hb_i                        = {}
      self.hb_j                        = {}
      self.hb_k                        = {}
      self._data                       = {}
      self.estruc                      = {}
      for s in strucs:
          s_ = s.split('-')[0]
          self.natom[s]    = strucs[s].natom
          self.nang[s]     = strucs[s].nang
          self.ang_j[s]    = np.expand_dims(strucs[s].ang_j,axis=1)
          self.ang_i[s]    = np.expand_dims(strucs[s].ang_i,axis=1)
          self.ang_k[s]    = np.expand_dims(strucs[s].ang_k,axis=1)

          self.ntor[s]     = strucs[s].ntor
          self.tor_i[s]    = np.expand_dims(strucs[s].tor_i,axis=1)
          self.tor_j[s]    = np.expand_dims(strucs[s].tor_j,axis=1)
          self.tor_k[s]    = np.expand_dims(strucs[s].tor_k,axis=1)
          self.tor_l[s]    = np.expand_dims(strucs[s].tor_l,axis=1)

          self.hb_i[s]     = strucs[s].hb_i
          self.hb_j[s]     = strucs[s].hb_j
          self.hb_k[s]     = strucs[s].hb_k

          self.nbd[s]      = strucs[s].nbd
          self.na[s]       = strucs[s].na
          self.nt[s]       = strucs[s].nt
          # self.nv[s]     = strucs[s].nv
          self.nhb[s]      = strucs[s].nhb
          self.b[s]        = strucs[s].B
          self.a[s]        = strucs[s].A
          self.t[s]        = strucs[s].T

          self.bdid[s]     = strucs[s].bond  # bond index like pair (i,j).
          self.atom_name[s]= strucs[s].atom_name
          
          self.s[s]        = {sp:[] for sp in self.spec}
          for i,sp in enumerate(self.atom_name[s]):
              self.s[s][sp].append(i)
          self.ns[s]       = {sp:len(self.s[s][sp]) for sp in self.spec}

          self._data[s]    = Dataset(dft_energy=strucs[s].energy_dft,
                                     x=strucs[s].x,
                                     cell=strucs[s].cell,
                                     rcell=strucs[s].rcell,
                                     forces=strucs[s].forces,
                                     q=strucs[s].qij)

          self.dft_energy[s] = np.array(self._data[s].dft_energy)
          self.q[s]          = np.array(self._data[s].q)
          self.eself[s]      = np.array(strucs[s].eself)  
          if self._data[s].forces is None or self.weight_force[s]==0:
             self.dft_forces[s] = None
          else:
             self.dft_forces[s] = np.array(self._data[s].forces)

          if s_ in self.estruc:
             self.estruc[s] = self.estruc[s_]
          else:
             if s_ in self.MolEnergy_:
                self.pp[s_] = np.array(self.MolEnergy_[s_])
                self.estruc[s_] = self.pp[s_] 
                if s not in self.estruc: self.estruc[s]  = self.estruc[s_] 
             else:
                est = 0.0 # self.dft_energy[s] - self.E[s]
                self.pp[s_] = np.array(est) 
                self.estruc[s_] = self.pp[s_] 
                if s not in self.estruc: self.estruc[s]  = self.estruc[s_] 
  
  def set_m(self):
      self.m = set_matrix(self.m_,self.spec,self.bonds,
                          self.mfopt,self.beopt,self.bdopt,1,
                          (6,0),(6,0),0,0,
                          self.mf_layer,self.mf_layer_,self.MessageFunction_,self.MessageFunction,
                          self.be_layer,self.be_layer_,1,1,
                          (9,0),(9,0),1,1,
                          None,self.be_universal,self.mf_universal,None)

    #   for key in self.m:
    #       k = key.split('_')[0]
    #       if k[0]=='f' and (k[-1]=='w' or k[-1]=='b'):
    #          for i,m in enumerate(self.m[key]):
    #             for dev in self.devices:
    #                 self.m[key][i]   
    #       else:
    #          for dev in self.devices:
    #              self.m[key]  

  def get_penalty(self,st):
      ''' adding some penalty term to pretain the phyical meaning '''
      penalty = np.array(0.0)
      wb_p    = []
      # if self.regularize_be:
      wb_p.append('fe')
      # if self.vdwnn and self.regularize_vdw:
      #    wb_p.append('fv')
      w_n     = ['wi','wo',]
      b_n     = ['bi','bo']
      layer   = {'fe':self.be_layer[1]}

      wb_message = []
      for t in range(1,self.messages+1):
          wb_message.append('fm')          
          layer['fm'] = self.mf_layer[1]  

      self.penalty_bop     = {}
      self.penalty_bo      = {}
      self.penalty_bo_rcut = {}
      self.penalty_be_cut  = {}
      self.penalty_rcut    = {}
      self.penalty_ang     = {}
      self.penalty_w       = np.array(0.0)
      self.penalty_b       = np.array(0.0)
      # self.rc_bo         = {}
      for bd in self.bonds: 
          atomi,atomj = bd.split('-') 
          self.penalty_bop[bd]     = 0.0
          self.penalty_be_cut[bd]  = 0.0
          self.penalty_bo_rcut[bd] = 0.0
          self.penalty_bo[bd]      = 0.0
          rr   = self.log_/self.p['bo1_'+bd] 
          self.rc_bo[bd]=self.p['rosi_'+bd]*np.power(rr,1.0/self.p['bo2_'+bd])
          
          if self.fixrcbo:
             rcut_si = np.square(self.rc_bo[bd]-self.rcut[bd])
          else:
             rcut_si = np.maximum(0, self.rc_bo[bd]-self.rcut[bd])

          rc_bopi = self.p['ropi_'+bd]*np.power(self.log_/self.p['bo3_'+bd],1.0/self.p['bo4_'+bd])
          rcut_pi = np.maximum(0, rc_bopi-self.rcut[bd])

          rc_bopp = self.p['ropp_'+bd]*np.power(self.log_/self.p['bo5_'+bd],1.0/self.p['bo6_'+bd])
          rcut_pp = np.maximum(0, rc_bopp-self.rcut[bd])

          self.penalty_rcut[bd] = rcut_si + rcut_pi + rcut_pp
          penalty =  penalty + self.penalty_rcut[bd]*self.lambda_bd
 
          # for st in self.strcs:
          if self.nbd[st][bd]>0:       
             b_    = self.b[st][bd]
             bdid  = self.bdid[st][b_[0]:b_[1]]

             bo0_  = self.bo0[st][:,bdid[:,0],bdid[:,1]]
             bop_  = self.bop[st][:,bdid[:,0],bdid[:,1]]

             rc_bo = self.rc_bo[bd]

             fbo  = np.where(np.less(self.rbd[st][bd],rc_bo),0.0,1.0)    # bop should be zero if r>rcut_bo
             # print(bd,'bop_',bop_.shape,'rbd',self.rbd[st][bd].shape)
             self.penalty_bop[bd]  =  self.penalty_bop[bd]  + np.sum(bop_*fbo)                              #####  

             if bd in self.bo_clip:
                for pbo in self.bo_clip[bd]:
                    r,dil,diu,djl,dju,bo_l,bo_u = pbo
                    dbi = self.Dbi[st][bd] 
                    dbj = self.Dbj[st][bd] 
                    fe  = np.where(np.logical_and(np.less_equal(self.rbd[st][bd],r),
                                        np.logical_and(np.logical_and(np.greater_equal(dbi,dil),
                                                        np.greater_equal(dbj,djl)), 
                                                        np.logical_and(np.less_equal(dbi,diu),
                                                        np.less_equal(dbj,dju))  ) ),
                                       1.0,0.0)   ##### r< r_e that bo > bore_
                    self.penalty_bo[bd] += np.sum(np.maximum(0, (bo_l-bo0_)*fe)) 
                                                                                    #     self.bo0[bd]
                    fe   = np.where(np.logical_and(np.greater_equal(self.rbd[st][bd],r),
                                        np.logical_and(np.logical_and(np.greater_equal(dbi,dil),
                                                   np.greater_equal(dbj,djl)), 
                                                       np.logical_and(np.less_equal(dbi,diu),
                                                            np.less_equal(dbj,dju))  ) ),
                                                1.0,0.0)  ##### r> r_e that bo < bore_
                    self.penalty_bo[bd] += np.sum(np.maximum(0, (bo0_-bo_u)*fe))
             fao  = np.where(np.greater(self.rbd[st][bd],self.rcuta[bd]),1.0,0.0) ##### r> rcuta that bo = 0.0
             self.penalty_bo_rcut[bd] = self.penalty_bo_rcut[bd] + np.sum(bo0_*fao)

             fesi = np.where(np.less_equal(bo0_,self.botol),1.0,0.0)              ##### bo <= 0.0 that e = 0.0
             self.penalty_be_cut[bd]  = self.penalty_be_cut[bd]  + np.sum(np.maximum(0, self.esi[st][bd]*fesi))
             
          if self.lambda_ang>0.000001:
             self.penalty_ang[st] = np.sum(self.thet2[st]*self.fijk[st])
          
          penalty  = penalty + self.penalty_be_cut[bd]*self.lambda_bd
          penalty  = penalty + self.penalty_bop[bd]*self.lambda_bd      
          penalty  = penalty + self.penalty_bo_rcut[bd]*self.lambda_bd
          penalty  = penalty + self.penalty_bo[bd]*self.lambda_bd

          # penalize term for regularization of the neural networs
          if self.lambda_reg>0.000001:             # regularize to avoid overfit
             for k in wb_p:
                 for k_ in w_n:
                     key     = k + k_ + '_' + bd
                     self.penalty_w = self.penalty_w + np.sum(np.square(self.m[key]))
                  
                 for k_ in b_n:
                     key     = k + k_ + '_' + bd
                     self.penalty_b  = self.penalty_b + np.sum(np.square(self.m[key]))
                 for l in range(layer[k]):                                               
                     self.penalty_w = self.penalty_w + np.sum(np.square(self.m[k+'w_'+bd][l]))
                     self.penalty_b = self.penalty_b + np.sum(np.square(self.m[k+'b_'+bd][l]))

      if self.lambda_reg>0.000001:                # regularize neural network
         for sp in self.spec:
             for k in wb_message:
                 for k_ in w_n:
                     key     = k + k_ + '_' + sp
                     self.penalty_w = self.penalty_w  + np.sum(np.square(self.m[key]))
                 for k_ in b_n:
                     key     = k + k_ + '_' + sp
                     self.penalty_b = self.penalty_b + np.sum(np.square(self.m[key]))
                 for l in range(layer[k]):                                               
                     self.penalty_w = self.penalty_w + np.sum(np.square(self.m[k+'w_'+sp][l]))
                     self.penalty_b = self.penalty_b + np.sum(np.square(self.m[k+'b_'+sp][l]))
         penalty = penalty + self.lambda_reg*self.penalty_w
         penalty = penalty + self.lambda_reg*self.penalty_b
      return penalty

  def read_ffield(self,libfile):
      if libfile.endswith('.json'):
         lf                   = open(libfile,'r')
         j                    = js.load(lf)
         self.p_              = j['p']
         self.m_              = j['m']
         self.MolEnergy_      = j['MolEnergy']
         self.messages        = j['messages']
         self.BOFunction      = j['BOFunction']
         self.EnergyFunction_ =  j['EnergyFunction']
         self.MessageFunction_= j['MessageFunction']
         self.VdwFunction     = j['VdwFunction']
         self.mf_layer_       = j['mf_layer']
         self.be_layer_       = j['be_layer']
         rcut                 = j['rcut']
         rcuta                = j['rcutBond']
         re                   = j['rEquilibrium']
         lf.close()
         self.init_bonds()
         self.emol = 0.0
      else:
         (self.p_,zpe_,self.spec,self.bonds,self.offd,self.angs,
          self.torp,self.hbs) = read_ffield(libfile=libfile,zpe=False)
         self.m_        = None
         self.mf_layer_ = None
         self.be_layer_ = None
         self.emol      = 0.0
         rcut           = None
         rcuta          = None
         re             = None
         # self.vdwnn     = False
         self.EnergyFunction_ = 0
         self.MessageFunction_= 0
         self.VdwFunction     = 0
         self.p_['acut']   = 0.0001
         self.p_['hbtol']  = 0.0001
      if self.mf_layer is None:
         self.mf_layer = self.mf_layer_
      if self.be_layer is None:
         self.be_layer = self.be_layer_
      if self.m_ is None:
         self.nn=False
      # for sp in self.atom_name:
      #     if sp not in self.spec:
      #        self.spec.append(sp)
      return self.m_,rcut,rcuta,re
  
  def set_memory(self):
      self.r,self.vr,self.rbd      = {},{},{}
      self.frc,self.rc_bo          = {},{}
      self.E                       = {}
      self.force                   = {}
      self.esi,self.ebd,self.ebond = {},{},{}
      self.bop,self.bop_si,self.bop_pi,self.bop_pp = {},{},{},{}
      self.bo,self.bo0,self.bosi,self.bopi,self.bopp = {},{},{},{},{}

      self.Pbo,self.Nlp = {},{}

      self.Deltap,self.Delta = {},{}
      self.D_si,self.D_pi,self.D_pp = {},{},{}
      self.D,self.H,self.Hsi,self.Hpi,self.Hpp = {},{},{},{},{}

      self.Dbi   = {}
      self.Dbj   = {}
      for st in self.strcs:
          self.Dbi[st]   = {}
          self.Dbj[st]   = {}

      self.Delta_pi,self.delta_pi = {},{}
      self.SO,self.Delta_ang   = {},{}
      self.fbot,self.fhb = {},{}

      self.Elone,self.Eover,self.Eunder = {},{},{}
      self.elone,self.eover,self.eunder = {},{},{}
      self.eatomic,self.zpe             = {},{}

      self.Eang,self.Epen,self.Etcon    = {},{},{}
      self.eang,self.epen,self.etcon    = {},{},{}

      self.etor,self.Etor               = {},{}
      self.efcon,self.Efcon             = {},{}

      self.evdw,self.Evdw               = {},{}
      self.ecoul,self.Ecoul             = {},{}
      self.ehb,self.Ehb                 = {},{}

  def run(self,step=1000):
      optimizer = optax.adam(list(self.pp.values()), lr=0.0001 )
      for i in range(step):
          self.forward()
          loss = self.get_loss()
          
        
          if i%10==0:
             print( "{:8d} loss: {:10.5f}   energy: {:10.5f}   force: {:10.5f}".format(i,
                    loss,self.loss_e,self.loss_f))
          if i%1000==0:
             self.save_ffield('ffield_{:d}.json'.format(i))
          
          
          
          
      self.save_ffield('ffield.json')

  def save_ffield(self,ffield='ffield.json'):
      # print('save parameter file ...')
      for key in self.estruc:
          k = key.split('-')[0]
          self.MolEnergy_[k] = self.estruc[k]

      for k in self.p:
          key   = k.split('_')[0]
          unit  = self.unit if key in self.punit else 1.0
          value = float(self.p[k]/unit)
          # print(k,' = ',value)
          if key in ['V1','V2','V3','tor1','cot1']:
             k_ = k.split('_')[1]
             if k_ not in self.torp:
                self.p_.pop(k,None)
                continue
              
          if key in self.p_offd:
             b = k.split('_')[1]
             s = b.split('-')
             if s[0]==s[1]:
                self.p_[key+'_'+s[0]] = value
             self.p_[k] = value
          else:
             self.p_[k] = value
        # if k in self.ea_var:
        #    self.p_[k] = self.ea_var[k]
        # else:

      loss  = self.loss_e + self.loss_f
      score = loss if loss is None else -loss
         
      if ffield.endswith('.json'):
         for key in self.m:
             k_ = key.split('_')
             k  = k_[0]
             if len(k_)>1:
                if k[0]=='f' and (k[-1]=='w' or k[-1]=='b'):
                    for i,m in enumerate(self.m[key]):
                        # if isinstance(M, np.ndarray):
                        if self.device['diff'].type == 'cpu':
                           self.m_[key][i] = m.tolist()
                        else:
                           self.m_[key][i] = m.tolist()
                else:
                    if self.device['diff'].type == 'cpu':
                       self.m_[key] = self.m[key].tolist()  # covert ndarray to list
                    else:
                       self.m_[key] = self.m[key].tolist()
         # print(' * save parameters to file ...')
         fj = open(ffield,'w')
         j = {'p':self.p_,'m':self.m_,
              'score':score,
              'BOFunction':self.BOFunction,
              'EnergyFunction':self.EnergyFunction,
              'MessageFunction':self.MessageFunction, 
              'VdwFunction':self.VdwFunction,
              'messages':self.messages,
              'bo_layer':None,
              'mf_layer':self.mf_layer,
              'be_layer':self.be_layer,
              'vdw_layer':None,
              'rcut':self.rcut,
              'rcutBond':self.rcuta,
              'rEquilibrium':self.re,
              'MolEnergy':self.MolEnergy_}
         js.dump(j,fj,sort_keys=True,indent=2)
         fj.close()
      else:
         raise RuntimeError('Error: other format is not supported yet!')
  def close(self):
      self.set_memory()
      self.data = None
