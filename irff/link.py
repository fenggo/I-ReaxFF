from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np


def get_bond_id(mol,i,j,bdlall):
    bdl = [mol,i,j]
    bdlr= [mol,j,i]
    if bdl in bdlall:
       ind = bdlall.index(bdl)   
    elif bdlr in bdlall:
       ind = bdlall.index(bdlr)  
    else:
       print('-  an error case for %s.........' %bdl)
       print(bdlall)
       ind = None
    return ind


class links(object):
  ''' links the bonds angles and torsions into a big graph '''
  def __init__(self,species=None,bonds=None,angs=None,
               tors=None,hbs=None,g=False,
               vdwcut=None,
               molecules=None):
      self.dft_energy = {}
      self.species= species
      self.bonds  = bonds
      self.angs   = angs
      self.tors   = tors
      self.hbs    = hbs
      self.vdwcut = vdwcut
      self.g      = g
      self.update_links(molecules)

      for mol in molecules:
          self.dft_energy[mol] = molecules[mol].energy_nw
      # self.histogram()


  def update_links(self,molecules):
      self.get_bond_link(molecules)
      self.get_atom_link(molecules)
      self.get_angle_link(molecules)
      self.get_torsion_link(molecules)
      self.get_vdw_link(molecules)
      self.get_hb_link(molecules)
      if self.g:
         self.get_glink()


  def get_atom_link(self,molecules):
      self.atomlist = {}
      self.atlab    = {}
      self.nsp      = {}
      for sp in self.species:
          self.atomlist[sp] = []
          self.atlab[sp]    = []
          for i,lab in enumerate(self.atom_lab):
              mol   = lab[0]
              iat   = lab[1]
              if molecules[mol].atom_name[iat]==sp:
                 self.atomlist[sp].append([i])
                 self.atlab[sp].append(lab)
          self.nsp[sp] = len(self.atlab[sp])

      self.atlall = []      # bond label concated all together
      for sp in self.species:
          self.atlall.extend(self.atlab[sp])
      
      self.dalink = []
      # print(self.atlall)
      for a in self.atom_lab:
          self.dalink.append([self.atlall.index(a)])

      self.atomlink = {}
      for mol in molecules:
          self.atomlink[mol] = []
          for i,atl in enumerate(self.atlall):
              if atl[0]==mol:
                 self.atomlink[mol].append([i])


  def get_glink(self):
      self.natom = len(self.atlall)
      glist = [[] for a in self.atlall]
      maxg  = 0

      for ia,ang in enumerate(self.angall): # [key,ai,aj,ak]
          atl = [ang[0],ang[2]]
          i_  = self.atlall.index(atl)
          glist[i_].append(ia+1)

      for i,a in enumerate(self.atlall):
          ng = len(glist[i])
          if ng>maxg: maxg=ng

      self.maxg    = maxg
      self.glist   = np.zeros([self.natom,self.maxg,1],dtype=np.int64)
      self.rijlist = np.zeros([self.natom,self.maxg,1],dtype=np.int64)
      self.riklist = np.zeros([self.natom,self.maxg,1],dtype=np.int64)

      for atom,gl in enumerate(glist):
          for ng,g in enumerate(gl):
              self.glist[atom][ng][0] = g  # gather theta and R is enough
              # print('-  info:',maxg,g,self.atlall[atom],self.angall[g-1])
              key,atomj,atomi,atomk = self.angall[g-1]
               
              ij = get_bond_id(key,atomi,atomj,self.bdlall)
              ik = get_bond_id(key,atomi,atomk,self.bdlall)
 
              self.rijlist[atom][ng][0] = ij
              self.riklist[atom][ng][0] = ik


  def get_bond_link(self,molecules):
      ''' label bonds and atoms '''
      self.max_nei  = 0
      self.nbond    = 0
      self.nbd      = {}
      self.bdlab    = {}
      self.atom_lab = []
      self.rbd      = {}
      self.natom    = 0

      # label atoms 
      atom = 0

      for key in molecules:   
          self.natom += molecules[key].natom
          self.nbond += molecules[key].nbond
          
          if molecules[key].max_nei>self.max_nei:    # get max neighbor
             self.max_nei = molecules[key].max_nei

          for iatom in range(molecules[key].natom):
              self.atom_lab.append([key,iatom])
          atom += 1

      # label bonds
      for bd in self.bonds:
          self.rbd[bd]   = []
          self.bdlab[bd] = []
          for key in molecules:  
              for i,b in enumerate(molecules[key].bond):
                  bn = molecules[key].atom_name[b[0]]+'-'+molecules[key].atom_name[b[1]]
                  bnr= molecules[key].atom_name[b[1]]+'-'+molecules[key].atom_name[b[0]]
                  if bn == bd:
                     self.rbd[bd].append(molecules[key].rbd[:,i])
                     self.bdlab[bd].append([key,b[0],b[1]])
                  if bnr==bd and (not bd==bn):
                     print('-  an error case of %s .........' %bd)

      self.bdlall = []      # bond label concated all together
      for bd in self.bonds:
          self.bdlall.extend(self.bdlab[bd])

      self.bdlink = {}
      for mol in molecules:
          self.bdlink[mol] = []
          for i,bl in enumerate(self.bdlall):
              if bl[0]==mol:
                 self.bdlink[mol].append([i])


      self.dlist = np.zeros([self.natom,self.max_nei,1],dtype=np.int64)
      self.dalist = np.zeros([self.natom,self.max_nei,1],dtype=np.int64)
      
      for i,lab in enumerate(self.atom_lab):
          mol   = lab[0]
          iatom = lab[1]
          for j,jatom in enumerate(molecules[mol].table[iatom]):
              # print('- bond label:',[mol,iatom,jatom])
              ind = get_bond_id(mol,iatom,jatom,self.bdlall)
              self.dlist[i][j][0]  = ind+1
              self.dalist[i][j][0] = self.atom_lab.index([mol,jatom]) # jatom

      self.dilink,self.djlink = {},{}
      for bd in self.bonds:
          self.nbd[bd] = len(self.bdlab[bd])
          self.dilink[bd] = np.zeros([self.nbd[bd],1],dtype=np.int64)
          self.djlink[bd] = np.zeros([self.nbd[bd],1],dtype=np.int64)
          for i,b in enumerate(self.bdlab[bd]):
              self.dilink[bd][i] = self.atom_lab.index([b[0],b[1]])
              self.djlink[bd][i] = self.atom_lab.index([b[0],b[2]])


  def get_angle_link(self,molecules):
      self.nang      = {}
      self.anglab    = {}
      self.dglist    = {}
      self.dgilist   = {}
      self.dgklist   = {}
      self.boaij     = {}
      self.boajk     = {}

      self.theta     = {}
      self.angall    = []

      for ang in self.angs:
          self.theta[ang]     = []
          self.anglab[ang]    = []
          self.dglist[ang]    = []
          self.dgilist[ang]   = []
          self.dgklist[ang]   = []
          self.boaij[ang]     = []
          self.boajk[ang]     = []
          for key in molecules:   
              iang = []
              for i,angi in enumerate(molecules[key].ang_i):
                  ai,aj,ak = molecules[key].ang_i[i],molecules[key].ang_j[i],molecules[key].ang_k[i]
                  an = molecules[key].atom_name[ai]+'-'+molecules[key].atom_name[aj]+'-'+molecules[key].atom_name[ak]
                  anr= molecules[key].atom_name[ak]+'-'+molecules[key].atom_name[aj]+'-'+molecules[key].atom_name[ai]
                  if an==ang or anr==ang:
                     if anr==ang:
                        ai,aj,ak = molecules[key].ang_k[i],molecules[key].ang_j[i],molecules[key].ang_i[i]
                     iang.append(i)
                     self.anglab[ang].append([key,ai,aj,ak])

                     # print('- angle label:',an,[key,ai,aj,ak])
                     ij = get_bond_id(key,ai,aj,self.bdlall)
                     jk = get_bond_id(key,aj,ak,self.bdlall)

                     self.boaij[ang].append([ij+1])
                     self.boajk[ang].append([jk+1])
                     self.dglist[ang].append([self.atom_lab.index([key,aj])])
                     self.dgilist[ang].append([self.atom_lab.index([key,ai])])
                     self.dgklist[ang].append([self.atom_lab.index([key,ak])])
              if iang:
                 self.theta[ang].extend(molecules[key].theta[iang])
          self.angall.extend(self.anglab[ang])
          self.nang[ang] = len(self.anglab[ang])
      
      self.anglink = {}
      for mol in molecules:
          self.anglink[mol] = {}
          for ang in self.angs:
              if self.nang[ang]>0:
                 self.anglink[mol][ang]=[]
                 for i,al in enumerate(self.anglab[ang]):
                     if al[0]==mol:
                        self.anglink[mol][ang].append([i])
  

  def get_torsion_link(self,molecules):
      self.ntor      = {}
      self.torlab    = {}
      self.tij       = {}
      self.tjk       = {}
      self.tkl       = {}
      self.dtj       = {}
      self.dtk       = {}
      self.s_ijk,self.s_jkl,self.w={},{},{}
      
      for tor in self.tors:
          self.torlab[tor]   = []
          self.tij[tor]      = []
          self.tjk[tor]      = []
          self.tkl[tor]      = []
          self.dtj[tor]      = []
          self.dtk[tor]      = []
          self.s_ijk[tor],self.s_jkl[tor] = [],[]
          self.w[tor] = []

          for key in molecules:   
              itor = []
              for i,tori in enumerate(molecules[key].tor_i):
                  atn   = molecules[key].atom_name
                  ti,tj = molecules[key].tor_i[i],molecules[key].tor_j[i]
                  tk,tl = molecules[key].tor_k[i],molecules[key].tor_l[i]
                  tn = atn[ti]+'-'+atn[tj]+'-'+atn[tk]+'-'+atn[tl]
                  tnr= atn[tl]+'-'+atn[tk]+'-'+atn[tj]+'-'+atn[ti]
                  if tn==tor or tnr==tor:
                     if tnr==tor and tn!=tor:
                        ti,tj = molecules[key].tor_l[i],molecules[key].tor_k[i]
                        tk,tl = molecules[key].tor_j[i],molecules[key].tor_i[i]
                     self.torlab[tor].append([key,ti,tj,tk,tl])
                     
                     itor.append(i)

                     ij = get_bond_id(key,ti,tj,self.bdlall)
                     jk = get_bond_id(key,tj,tk,self.bdlall)
                     kl = get_bond_id(key,tk,tl,self.bdlall)

                     self.tij[tor].append([ij+1])
                     self.tjk[tor].append([jk+1])
                     self.tkl[tor].append([kl+1])

                     self.dtj[tor].append([self.atom_lab.index([key,tj])])
                     self.dtk[tor].append([self.atom_lab.index([key,tk])])
              if itor:
                 self.s_ijk[tor].extend(molecules[key].s_ijk[itor])
                 self.s_jkl[tor].extend(molecules[key].s_jkl[itor])
                 self.w[tor].extend(molecules[key].w[itor])
          # print('- %s' %tor,self.ntor[tor],'\n')
          # print(self.torlab[tor])
          self.ntor[tor] = len(self.torlab[tor])
      
      self.torlink = {}
      for mol in molecules:
          self.torlink[mol] = {}
          for tor in self.tors:
              if self.ntor[tor]>0:
                 self.torlink[mol][tor]=[]
                 for i,tl in enumerate(self.torlab[tor]):
                     if tl[0]==mol:
                        self.torlink[mol][tor].append([i])


  def get_vdw_link(self,molecules):
      self.vlab = {}
      self.nvb  = {}
      self.rv   = {}
      self.vi   = {}
      self.vj   = {}
      self.qij  = {}
      
      for vb in self.bonds:
          self.vlab[vb]   = []
          self.rv[vb]     = []
          self.qij[vb]    = []
          self.vi[vb]     = []
          self.vj[vb]     = []

          for key in molecules:   
              for i,vi in enumerate(molecules[key].vi):
                  # vi = molecules[key].vi[i]
                  vj = molecules[key].vj[i]
                  vn = molecules[key].atom_name[vi]+'-'+molecules[key].atom_name[vj]
                  vnr= molecules[key].atom_name[vj]+'-'+molecules[key].atom_name[vi]
                  if vn==vb or vnr==vb:
                     self.rv[vb].append(molecules[key].rv[i,:])  # changed here by [:,i]
                     self.vlab[vb].append([key,vi,vj])                                 # angstoev
                     self.qij[vb].append(molecules[key].q[:,vi]*molecules[key].q[:,vj]*14.39975840)
                     self.vi[vb].append([self.atom_lab.index([key,vi])])
                     self.vj[vb].append([self.atom_lab.index([key,vj])])
          self.nvb[vb] = len(self.vlab[vb])
          # print(len(self.vi[vb]),len(self.rv[vb]))

      self.vlink = {}
      for mol in molecules:
          self.vlink[mol] = {}
          for vb in self.bonds:
              if self.nvb[vb]>0:
                 self.vlink[mol][vb]=[]
                 for i,vl in enumerate(self.vlab[vb]):
                     if vl[0]==mol:
                        self.vlink[mol][vb].append([i])


  def get_hb_link(self,molecules):
      self.nhb     = {}
      self.rhb     = {}
      # self.rik     = {}
      # self.rij     = {}
      self.hblab   = {}
      self.hij     = {}
      self.hbthe,self.frhb ={},{}

      for hb in self.hbs:
          self.hblab[hb] = []
          self.hbthe[hb] = []
          self.frhb[hb]  = []
          self.rhb[hb]   = []
          # self.rij[hb]   = []
          # self.rik[hb]   = []
          self.hij[hb]   = []
          for key in molecules:   
              for i,hbi in enumerate(molecules[key].hb_i):
                  hi = molecules[key].hb_i[i]
                  hj = molecules[key].hb_j[i]
                  hk = molecules[key].hb_k[i]
                  hn = molecules[key].atom_name[hi]+'-'+molecules[key].atom_name[hj]+'-'+molecules[key].atom_name[hk]
                  if hn==hb:
                     ij = get_bond_id(key,hi,hj,self.bdlall)
                     self.hij[hb].append([ij+1])
                     self.hblab[hb].append([key,hi,hj,hk])
                     self.rhb[hb].append(molecules[key].rhb[i,:])  # changed here by [:,i]
                     # self.rij[hb].append(molecules[key].rij[i,:])
                     # self.rik[hb].append(molecules[key].rik[i,:])
                     self.hbthe[hb].append(molecules[key].hbthe[i,:])
                     self.frhb[hb].append(molecules[key].frhb[i,:])

          self.nhb[hb]   = len(self.hblab[hb])
          self.rhb[hb]   = np.array(self.rhb[hb])
          # self.rik[hb]   = np.array(self.rik[hb])
          self.frhb[hb]  = np.array(self.frhb[hb])
          self.hbthe[hb] = np.array(self.hbthe[hb])

      self.hblink = {}
      for mol in molecules:
          self.hblink[mol] = {}
          for hb in self.hbs:
              if self.nhb[hb]>0:
                 self.hblink[mol][hb]=[]
                 for i,hl in enumerate(self.hblab[hb]):
                     if hl[0]==mol:
                        self.hblink[mol][hb].append([i])


  def histogram(self):
      print('-  plotting bond length histogram ...')
      for bd in self.bonds:
          if self.nbd[bd]>0:
             plt.figure()   
             ax = plt.gca() 
             ax.xaxis.set_major_locator(MultipleLocator(0.5)) 
             ax.xaxis.set_minor_locator(MultipleLocator(0.1))
             plt.ylabel('Bond length distribution')
             plt.xlabel('Radius (Angstrom)')
             plt.hist(np.reshape(self.rbd[bd],[-1]),1000,alpha=0.01,label='%s' %bd)
             plt.legend(loc='best')
             plt.savefig('%s_bh.eps' %bd) 
             plt.close() 
             
             
