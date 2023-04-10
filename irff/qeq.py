from __future__ import print_function
import numpy as np
# np.set_printoptions(threshold=np.inf) 

rcutoff=10.0
rd     = 1.0/rcutoff**7.0
Q8     = 20.0*rd
Q7     = -70.0*rd*rcutoff
Q6     = 84.0*rd*rcutoff*rcutoff
Q5     = -35.0*rd*rcutoff*rcutoff*rcutoff
Q1     = 1.0



def qtap(r,rmax):
    fr    = np.where(r<rmax,1.0,0.0)  
    r_    = np.where(r<rmax,r,0.0)  
    tp    = (((Q8*r_+Q7)*r_+Q6)*r_+Q5)*r_*r_*r_*r_ + fr
    return tp


class qeq(object):
  ''' 
     charges calculate for ReaxFF potential
  '''
  def __init__(self,p=None,atoms=None,rcutoff=10.0,
               itmax=100,tol=1.0e-8,converge_stat=True):
      # self.chi       = {}
      self.itmax     = itmax
      self.atoms     = atoms
      self.rcutoff   = rcutoff
      self.cell      = atoms.get_cell()
      self.box       = np.sqrt(np.sum(self.cell*self.cell,axis=1))

      cos_alpha      = np.sum(self.cell[1]*self.cell[2])/(self.box[1]*self.box[2])
      self.alpha     = np.arccos(cos_alpha)
      cos_beta       = np.sum(self.cell[0]*self.cell[2])/(self.box[0]*self.box[2])
      self.beta      = np.arccos(cos_beta)
      cos_gamma      = np.sum(self.cell[0]*self.cell[1])/(self.box[0]*self.box[1])
      self.gam       = np.arccos(cos_gamma)

      self.atom_name = atoms.get_chemical_symbols()
      self.natom     = len(self.atom_name)
      self.eye       = np.eye(self.natom,dtype=int)
      self.diag      = 1.0 - self.eye
      # print(self.diag)
      self.angstoev  = 14.39975840 
      self.tol       = tol
      self.p         = p
      self.converge_stat = converge_stat
      

  def initialize(self):
      self.mu        = np.zeros([self.natom])
      self.z         = np.zeros([self.natom+1])
      self.q_        = np.zeros([self.natom+1])    # resolution                   
      self.gamma     = np.zeros([self.natom,1])
      self.D_        = np.zeros([self.natom,self.natom])

      for i,atom in enumerate(self.atom_name):
          self.z[i]      =-self.p['chi_'+atom]
          self.gamma[i]  = self.p['gamma_'+atom]
          self.mu[i]     = self.p['mu_'+atom]
      self.z[self.natom] = 0.0
      self.q_[self.natom]= 0.0
      self.not_converged = True


  def calc(self,cell,positions):
      self.initialize()
      self.q = self.q_.copy()
      # self.z = self.z_.copy()
      self.D = self.D_.copy()
      gamma_ = np.sqrt(self.gamma*np.transpose(self.gamma,[1,0]))
      gamma2 = 1.0/gamma_**3

      x         = positions
      self.cell = cell
      vr        = self.compute_bond(x,self.natom)
      
      ast,bst,cst = 0,0,0
      aed,bed,ced = 1,1,1

      na  = int(np.ceil(self.rcutoff/self.cell[0][0]))
      nb  = int(np.ceil(self.rcutoff/self.cell[1][1]))
      nc  = int(np.ceil(self.rcutoff/self.cell[2][2]))

      for i in range(ast-na,aed+na):
          for j in range(bst-nb,bed+nb):
              for k in range(cst-nc,ced+nc):
                  cell = self.cell[0]*i + self.cell[1]*j+self.cell[2]*k
                  vr_  = vr + cell
                  R_   = np.sqrt(np.sum(np.square(vr_),axis=2))

                  tp   = qtap(R_,self.rcutoff)
                  gam_ = (tp/(R_**3+gamma2)**(1.0/3.0))

                  if i==0 and j==0 and k==0:
                     gam_ = gam_*self.diag
                     
                  self.D += gam_*self.angstoev # *self.ut

      D_          = self.D*self.eye
      self.D      = self.D - D_

      Deye        = D_ + 2.0*self.mu*self.eye
      self.D     += Deye
      self.Deye   = np.sum(Deye,axis=1)

      #  sparseAxV i.e. D x q 
      qd_       = self.dot(self.D,self.q)
      r         = self.z - qd_
      self.bnrm = np.sum(self.z*self.z)

      # sparseAdiagprecon D r
      r_     = r.copy()
      z      = self.z.copy()

      z[:-1] = r[:-1]/self.Deye
      z[-1]  = r[-1]
      self.iteration(z,r,r_)
 

  def iteration(self,z,r,r_):
      z_ = np.zeros([self.natom+1])
      it = 0

      while it<self.itmax and self.not_converged:
            z_[:-1] = r_[:-1]/self.Deye
            z_[-1]  = r_[-1]
            bk_ = np.sum(z*r_)

            if it==0:
               p =z.copy()
               p_=z_.copy()
            else:
               bk = bk_/bkd
               p  = bk*p  + z
               p_ = bk*p_ + z_
            bkd = bk_
          
            # operation between D |nxn| and p(=z|n+1|)
            z   = self.dot(self.D,p)
            akd = np.sum(z*p_)
            ak  = bk_/akd
            z_  = self.dot(self.D,p_)

            self.q = self.q + ak*p
            r      = r  - ak*z
            r_     = r_ - ak*z_
            z[:-1] = r[:-1]/self.Deye
            z[-1]  = r[-1]

            err = np.sqrt(np.sum(r*r))/self.bnrm
            # print('-  iteration = %2d  Error = %16.12f' %(it,err))

            if err<self.tol:
               self.not_converged = False
            it += 1  

      q = self.q[:-1]
      # print('-  QEq charges \n:',q)
      if self.converge_stat and self.not_converged:
         print('-  QEq charges not converged:',np.sum(q))


  def dot(self,D,q):
      q_   = q[:-1]
      qd_  = np.dot(self.D,q_)
      qd_ += q[-1]
      qd_  = np.append(qd_,[np.sum(q_)],axis=0)
      return qd_


  def compute_bond(self,x,natom):
      hfcell = 0.5
      u      = np.linalg.inv(self.cell)
      x      = np.array(x)
      xf     = np.dot(x,u) 

      xj = np.expand_dims(xf,axis=0)
      xi = np.expand_dims(xf,axis=1)
      vr = xj - xi

      lm = np.where(vr-hfcell>0)
      lp = np.where(vr+hfcell<0)

      while(lm[0].size!=0 or lm[1].size!=0 or lm[2].size!=0 or 
            lp[0].size!=0 or lp[1].size!=0 or lp[2].size!=0):
          vr = np.where(vr-hfcell>0,vr-1.0,vr)
          vr = np.where(vr+hfcell<0,vr+1.0,vr)     # apply pbc
          lm = np.where(vr-hfcell>0)
          lp = np.where(vr+hfcell<0)

      vr_ = np.dot(vr,self.cell)
      return vr_


  def close(self):
      print('-  Qeq job compeleted.')
      self.D       = None
      self.q       = None
      self.alpha   = None
      self.beta    = None
      self.gamma   = None
      self.Deye    = None
      self.eye     = None

