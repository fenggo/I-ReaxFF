from os import system
from collections import deque
import random
# from .findmole import check_decomposed
from ase.io import read,write
from ase import Atoms
from ase.data import atomic_numbers,chemical_symbols
#from ase.calculators.lammpsrun import write_lammps_data
from ase.calculators.lammps import Prism,convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import TrajectoryWriter
from ase.data import atomic_numbers, atomic_masses
from ..molecule import molecules,enlarge,press_mol
from ..getNeighbors import get_neighbors
import matplotlib.pyplot as plt
import numpy as np


def run_lammps(inp_name='inp-lam',label='eos',np=4):
    print('mpirun -n %d lammps<%s> %s.out \n' %(np,inp_name, label))
    system('mpirun -n %d lammps<%s> %s.out' %(np,inp_name, label))

def get_lammps_thermal(logname='lmp.log',supercell=[1,1,1]):
    e0,p0,t0,v0,aa,ba,ca = 0.0,0.0,0.0,0.0,0,0,0
    e,t = [],[]
    a,b,c = [],[],[]
    alpha_a,beta_a,gamma_a,alpha,beta,gamma=0.0,0.0,0.0,[],[],[] 
    n, N, step,steps = 0,0,0,[]
    
    flog = open(logname,'r')
    flg = open('thermo.log','w')
    lread = False
    for line in flog.readlines():
        l = line.split()
        if len(l)>0:
           if l[0] == 'Step':
              lread = True
           elif l[0] == 'Loop':
              lread = False
              N = int(l[-2])
           if lread:
              if l[0]== 'Step':
                 lent = len(l)
                 #print lent
                 clm_t = l.index('Temp')
                 clm_e = l.index('TotEng')
                 clm_p = l.index('Press')
                 clm_v = l.index('Volume')
                 clm_a = l.index('Cella')
                 clm_b = l.index('Cellb')
                 clm_c = l.index('Cellc')
                 clm_al= l.index('CellAlpha')
                 clm_be= l.index('CellBeta')
                 clm_ga= l.index('CellGamma')
           if lread:
              if l[0] != 'Step' and len(l) ==lent:
                 # print('I do nothing!')
                 t0 += float(l[clm_t]) # colume number of T
                 e0 += float(l[clm_e])
                 e.append(float(l[clm_e]))
                 t.append(float(l[clm_t]))
                 p0 += float(l[clm_p])
                 v0 += float(l[clm_v])
                 step = int(l[0])
                 steps.append(step)
                 n += 1
                 a.append(float(l[clm_a])/supercell[0])
                 b.append(float(l[clm_b])/supercell[1])
                 c.append(float(l[clm_c])/supercell[2])
                 alpha.append(float(l[clm_al]))
                 beta.append(float(l[clm_be]))
                 gamma.append(float(l[clm_ga]))
                 aa += float(l[clm_a])/supercell[0]
                 ba += float(l[clm_b])/supercell[1]
                 ca += float(l[clm_c])/supercell[2]
                 alpha_a += float(l[clm_al])
                 beta_a  += float(l[clm_be])
                 gamma_a += float(l[clm_ga])
                 print(l[0],l[clm_t],l[clm_e],float(l[clm_p])*0.0001,l[clm_v],file=flg) # pressure GPa
    flg.close()
    flog.close()
    if n == 0:
       print('Error: n=0!')
    else:
       t0=t0/n          # for average
       p0=p0/n
       e0=e0/n
       v0=v0/n
       aa=aa/n
       ba=ba/n
       ca=ca/n
       alpha_a=alpha_a/n
       beta_a=beta_a/n
       gamma_a=gamma_a/n
    if N == 0:
       print('Error: N=0!')
    plt.figure()
    plt.ylabel(r'$Energy$ ($eV$)')
    plt.xlabel(r'$Step$')
    # plt.scatter(ph[i],vh[i],marker = 'o', color = cmap.to_rgba(t), s=50, alpha=0.4)
    plt.plot(steps,e,label=r'$energy .vs. step$', color='black', linewidth=1.5, linestyle='-.')
    plt.legend()
    plt.savefig('energy.pdf') 
    plt.close()

    plt.figure()
    plt.ylabel(r'$Temperature$ ($T$)')
    plt.xlabel(r'$Step$')
    plt.plot(steps,t,label=r'$Temperature .vs. step$', color='black', linewidth=1.5, linestyle='-.')
    plt.legend()
    plt.savefig('temperature.pdf') 
    plt.close()

    plt.figure()
    plt.ylabel(r'$Lattice constant$ ($Angstrom$)')
    plt.xlabel(r'$Step$')
    plt.plot(steps,a,label=r'$a$', color='red', linewidth=1.5, linestyle='--')
    plt.plot(steps,b,label=r'$b$', color='blue', linewidth=1.5, linestyle='-.')
    plt.plot(steps,c,label=r'$c$', color='black', linewidth=1.5, linestyle=':')
    plt.legend()
    plt.savefig('lattice.pdf') 
    plt.close()

    plt.figure()
    plt.ylabel(r'$Angle constant$ ($degree$)')
    plt.xlabel(r'$Step$')
    plt.plot(steps,alpha,label=r'$alpha$', color='red', linewidth=1.5, linestyle='--')
    plt.plot(steps,beta,label=r'$beta$', color='blue', linewidth=1.5, linestyle='-.')
    plt.plot(steps,gamma,label=r'$gamma$', color='black', linewidth=1.5, linestyle=':')
    plt.legend()
    plt.savefig('angle.pdf') 
    plt.close()
    return step,N,t0,p0,e0,v0,aa,ba,ca,alpha_a,beta_a,gamma_a  # N ,atoms number, t0 temperature, p0 pressure, e0 energy v0,volume

def get_reaxff_energies(logname='lmp.log'):
    e0,p0,t0,v0,aa,ba,ca = 0.0,0.0,0.0,0.0,0,0,0
    t = []
    # eb,elp,ev,ep = [],[],[],[]
    # epen,ecoa,ehb,et,eco,ew=[],[],[],[],[],[] 
    n, N, step,steps = 0,0,0,[]
    
    flog = open(logname,'r')
    lread = False
    for line in flog.readlines():
        l = line.split()
        if len(l)>0:
           if l[0] == 'Step':
              lread = True
           elif l[0] == 'Loop':
              lread = False
              N = int(l[-2])
           if lread:
              if l[0]== 'Step':
                 lent = len(l)
                 #print lent
                 clm_t = l.index('Temp')
                 clm_p = l.index('Press')
                 clm_e = l.index('E_pair')
                 clm_eb = l.index('v_eb')
                 clm_ea = l.index('v_ea')
                 clm_elp = l.index('v_elp')
                 clm_ev = l.index('v_ev')
                 clm_epen = l.index('v_epen')
                 clm_ecoa= l.index('v_ecoa')
                 clm_ehb= l.index('v_ehb')
                 clm_et= l.index('v_et')
                 clm_eco= l.index('v_eco')
                 clm_ew= l.index('v_ew')
                 clm_ep= l.index('v_ep')
           if lread:
              if l[0] != 'Step' and len(l) ==lent:
                 # print('I do nothing!')
                 t0 += float(l[clm_t]) # colume number of T
                 e0 += float(l[clm_e])
                 e = float(l[clm_e])*4.3364432032e-2
                 t.append(float(l[clm_t]))
                 p0 += float(l[clm_p])
                 # v0 += float(l[clm_v])
                 step = int(l[0])
                 steps.append(step)
                 n += 1
                 eb   = float(l[clm_eb])*4.3364432032e-2
                 elp  = float(l[clm_elp])*4.3364432032e-2
                 ev   = float(l[clm_ev])*4.3364432032e-2
                 epen = float(l[clm_epen])*4.3364432032e-2
                 ecoa = float(l[clm_ecoa])*4.3364432032e-2
                 ehb  = float(l[clm_ehb])*4.3364432032e-2
                 et   = float(l[clm_et])*4.3364432032e-2
                 eco  = float(l[clm_eco])*4.3364432032e-2
                 ew   = float(l[clm_ew])*4.3364432032e-2
                 ep   = float(l[clm_ep])*4.3364432032e-2
                 # print('h-bond:',ehb,'etor',et) # pressure GPa
    flog.close()
    if n == 0:
       print('Error: n=0!')
    else:
       t0=t0/n          # for average
       p0=p0/n
       e0=e0/n
       # v0=v0/n
    if N == 0:
       print('Error: N=0!')
    return e,eb,elp,0.0,0.0,ev,epen,ecoa,et,eco,ew,ehb,ep,0.0

def writeLammpsData(atoms,data='data.lammps',specorder=None, 
                    masses=None,
                    force_skew=False,
                    velocities=False,units="real",atom_style='charge'):
    """Write atomic structure data to a LAMMPS data_ file."""
    f = open(data, "w", encoding="ascii")
    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!"
            )
        atoms = atoms[0]

    f.write("{0} \n\n".format(f.name))

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    f.write("{0} \t atoms \n".format(n_atoms))

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        species = sorted(set(symbols))
    else:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
    n_atom_types = len(species)
    f.write("{0}  atom types\n".format(n_atom_types))

    if not masses:
       masses  = {s:atomic_masses[atomic_numbers[s]] for s in species }

    p = Prism(atoms.get_cell())
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
            "ASE", units)

    f.write("0.0 {0:23.17g}  xlo xhi\n".format(xhi))
    f.write("0.0 {0:23.17g}  ylo yhi\n".format(yhi))
    f.write("0.0 {0:23.17g}  zlo zhi\n".format(zhi))

    if force_skew or p.is_skewed():
        f.write(
            "{0:23.17g} {1:23.17g} {2:23.17g}  xy xz yz\n".format(
                xy, xz, yz
            )
        )
    f.write("\n\n")
    f.write("Masses \n\n")
    for i,sp in enumerate(species):
        f.write("%d  %6.4f\n" %(i+1,masses[sp]))

    f.write("\n\n")
    f.write("Atoms \n\n")
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=True)

    if atom_style == 'atomic':
        for i, r in enumerate(pos):
            # Convert position from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            s = species.index(symbols[i]) + 1
            f.write(
                "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
                    *(i + 1, s) + tuple(r)
                )
            )
    elif atom_style == 'charge':
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            f.write(
                "{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n".format(
                    *(i + 1, s, q) + tuple(r)
                )
            )
    elif atom_style == 'full':
        charges = atoms.get_initial_charges()
        molecule = 1 # Assign all atoms to a single molecule
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            f.write(
                "{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} {6:23.17g}\n".format(
                    *(i + 1, molecule, s, q) + tuple(r)
                )
            )
    else:
        raise NotImplementedError

    if velocities and atoms.get_velocities() is not None:
        f.write("\n\nVelocities \n\n")
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            # Convert velocity from ASE units to LAMMPS units
            v = convert(v, "velocity", "ASE", units)
            f.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    f.flush()
    f.close()

def writeLammpsIn(log='lmp.log',timestep=0.1,total=200, data=None,restart=None,
              species=['C','H','O','N'],
              bond_cutoff={'H-H':1.2,'H-C':1.6,'H-O':1.6,'H-N':1.6,
                           'C-C':2.0,'other':2.0},
              pair_coeff ='* * ffield C H O N',
              pair_style = 'reaxff control nn yes checkqeq yes',  # without lg set lgvdw no
              fix = 'fix   1 all npt temp 800 800 100.0 iso 10000 10000 100',
              fix_modify = ' ',
              more_commond = ' ',
              dump_interval=10,
              freeatoms=None,natoms=None,
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              restartfile='restart',
              **kwargs):
    '''
        pair_style     reaxff control.reax checkqeq yes
        pair_coeff     * * ffield.reax.rdx C H O N
        --- control ---
        tabulate_long_range	0 ! denotes the granularity of long range tabulation, 0 means no tabulation
        nbrhood_cutoff		3.5  ! near neighbors cutoff for bond calculations
        hbond_cutoff		7.5  ! cutoff distance for hydrogen bond interactions
        bond_graph_cutoff	0.3  ! bond strength cutoff for bond graphs
        thb_cutoff		    0.001 ! cutoff value for three body interactions
        nnflag              1    ! 0: do not use neural network potential
        mflayer_m           9
        mflayer_n           1
        belayer_m           9
        belayer_n           1
    '''
    random.seed()
    species_name = {'H':'hydrogen','O':'oxygen','N': 'nitrogen','C':'carbon'}
    fin = open('in.lammps','w')
    for i,sp in enumerate(species):
        species_ = sp if sp not in species_name else species_name[sp]
        print('#/atom {:d} {:s}'.format(i+1,species_), file=fin)
    for i in range(len(species)):
        for j in range(i,len(species)):
            bd = species[i]+'-'+species[j]
            bdr= species[j]+'-'+species[i]
            if bd in bond_cutoff:
               bc = bond_cutoff[bd]
            elif bdr in bond_cutoff:
               bc = bond_cutoff[bdr]
            else:
               bc = bond_cutoff['other']
            print('#/bond {:d} {:d} {:f}'.format(i+1,j+1,bc), file=fin)

    if 'units' in kwargs:
       units = kwargs['units']
       print('units     {:s}'.format(kwargs['units']), file=fin)
    else:
       units = 'real'
       print('units     real', file=fin)
    if 'atom_style' in kwargs:
       print('atom_style     {:s}'.format(kwargs['atom_style']), file=fin)
    else:
       print('atom_style     charge', file=fin)

    if data != None and data != 'None':
       print('read_data    {:s}'.format(data), file=fin)
       if 'T' in kwargs:
          print('velocity     all create {:d} {:d}'.format(kwargs['T'],random.randint(0,10000)), file=fin)
       else:
          print('velocity     all create 300 {:d}'.format(random.randint(0,10000)), file=fin)
    if restart != None and restart != 'None':
       print('read_restart {:s}'.format(restart), file=fin)
    print(' ', file=fin)
    print('pair_style     {:s}'.format(pair_style), file=fin) 
    if isinstance(pair_coeff, list):
       for pc in pair_coeff:
           print('pair_coeff     {:s}'.format(pc), file=fin)
    else:
       print('pair_coeff     {:s}'.format(pair_coeff), file=fin)
    if pair_style.find('reaxff')>=0:
       print('compute       reax all pair reaxff', file=fin)
       print('variable eb   equal c_reax[1]', file=fin)
       print('variable ea   equal c_reax[2]', file=fin)
       print('variable elp  equal c_reax[3]', file=fin)
       print('variable emol equal c_reax[4]', file=fin)
       print('variable ev   equal c_reax[5]', file=fin)
       print('variable epen equal c_reax[6]', file=fin)
       print('variable ecoa equal c_reax[7]', file=fin)
       print('variable ehb  equal c_reax[8]', file=fin)
       print('variable et   equal c_reax[9]', file=fin)
       print('variable eco  equal c_reax[10]', file=fin)
       print('variable ew   equal c_reax[11]', file=fin)
       print('variable ep   equal c_reax[12]', file=fin)
       print('variable efi  equal c_reax[13]', file=fin)
       print('variable eqeq equal c_reax[14]', file=fin)
    print(' ', file=fin)
    print('neighbor 2.5  bin', file=fin)
    print('neigh_modify  every 1 delay 1 check no page 200000', file=fin)
    print(' ', file=fin)
    if freeatoms:
       # fixatom = [i for i in range(natoms) if i not in free]
       # print(freeatoms)
       print('group  free id ',end=' ', file=fin)
       for j in freeatoms:
           print(j,end=' ', file=fin) 
       print(' ', file=fin)
       print('group  fixed subtract all free', file=fin)
       print('fix    freeze fixed setforce 0.0 0.0 0.0', file=fin)
       print(' ', file=fin)
       fix = fix.replace('all','free')
    print(fix, file=fin)
    print(fix_modify, file=fin)
    if pair_style.find('reaxff')>=0:
       if freeatoms:
          print('fix    rex free qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff', file=fin)
       else:
          print('fix    rex all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff', file=fin)
       print('fix    sp  all reaxff/species 1 20 20  species.out', file=fin) # every 1 compute bond-order, per 20 av bo, and per 20 calc species
    print(' ', file=fin)
    print(more_commond, file=fin)
    
    if 'minimize' in kwargs:
        print('min_style   cg', file=fin)
        print('min_modify  line quadratic', file=fin)
        print('minimize    {:s}'.format(kwargs['minimize']), file=fin)

    print(' ', file=fin)
    print('thermo        {:d}'.format(dump_interval), file=fin)
    print(thermo_style, file=fin)
    print(' ', file=fin)
    # timestep = convert(timestep, "time", "ASE", units)
    if units == 'metal':
       timestep = timestep*0.001
    print('timestep      {:f}'.format(timestep), file=fin)
    print(' ', file=fin)
    if pair_style.find('reaxff')>=0:
       print('dump          1 all custom {:d} lammps.trj id type x y z q fx fy fz'.format(dump_interval), file=fin)
    else:
       print('dump          1 all custom {:d} lammps.trj id type x y z fx fy fz'.format(dump_interval), file=fin) 
    print(' ', file=fin)
    print('log           %s'  %log, file=fin)
    print(' ', file=fin)
    print('restart       10000 restart', file=fin)
    print('run           %d'  %total, file=fin)
    print(' ', file=fin)
    print('write_restart {:s}'.format(restartfile), file=fin)
    print(' ', file=fin)
    fin.close()


class EOS(object):
  """evaluate eos by lammps"""
  def __init__(self,restart=None,
              pair_coeff = '* * ffield.reax.lg C O N',
              pair_style = 'reax/c control.reax lgvdw yes checkqeq yes',
              struc = 'cl20',
              supercell=[1,1,1],
              fac = 1.20,
              np  = 4):
     self.pair_style = pair_style
     self.pair_coeff = pair_coeff
     self.fix_modify = ' '
     self.restart = restart
     self.np = np
     self.supercell = supercell
     self.structure = struc
     if self.restart is None:
        get_structure(struc=struc,output='dftb',recover=True,center=True,supercell=[1,1,1])
        A = read('card.gen')
        natm,atoms,X,table = get_neighbors(Atoms=A,exception=['O-O','H-H'])
        m = molecules(natm,atoms,X,table=table)
        cell = A.get_cell()
        m,A = enlarge(m,cell=cell,fac = fac,supercell=[1,1,1])
        cell = A.get_cell()

        emdk(cardtype='gen',cardfile='structure.gen',cell=cell,supercell=supercell,output='xyz',log='log')
        cell = [c*supercell[l] for l,c in enumerate(cell)]
        emdk(cardtype='xyz',cardfile='card.xyz',cell=cell,supercell=[1,1,1],output='lammpsdata',log='log')
        system('mv card.df data')
        self.data='data'
     else:
        self.data= None

  def run(self,totalstep=2000,restart=None,P = 1.0,pfreq=1000,T = 300.0,tfreq=1000):
     self.fix  = 'fix NPT all npt temp %f %f %f iso %f %f %f \n' %(T,T,tfreq,P,P,pfreq) #Atmospheric pressure
     self.fix += 'fix rxc all qeq/reax 1 0.0 10.0 1.0e-6 reax/c'
     data = self.data if restart is None else None 
     write_lammps_in(log='eos.log',total=totalstep, data = data,restart=restart,
                    pair_coeff = self.pair_coeff,
                    pair_style = self.pair_style,
                    fix = self.fix,
                    fix_modify = self.fix_modify,
                    more_commond = ' ',
                    thermo_style ='thermo_style     custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
                    restartfile = 'restart.x')

     run_lammps(inp_name='inp-lam',label='eos',np=self.np)
     step,N,t0,p0,e0,v0,a,b,c,alpha,beta,gamma= get_lammps_thermal(logname='eos.log',supercell=self.supercell) 
     feos = open('eos.txt','w')
     print("****************************************************************",file=feos)
     print("* structure %s:  " %self.structure,file=feos)
     print("* Pressure: %f, Energy     : %f  " %(p0,e0),file=feos)
     print("* Volume  : %f, Temperature: %f  " %(v0,t0),file=feos)
     print("* Free particals number: %d      " %N,file=feos)
     print("* Lattice Parameters:  " ,file=feos)
     print("* a, b, c           : %f  %f  %f    " %(a,b,c),file=feos)
     print("* alpha, beta, gamma: %f  %f  %f    " %(alpha,beta,gamma),file=feos)
     print("****************************************************************",file=feos)
     feos.close()

#   def check_species(self,species=1):
#       check_decomposed(traj='lammps.trj',nmol=species)

def lattice(a,b,c):
    # a =  [ 14.90415061,    -0.12901043,   0.43404866 ]
    # b =  [  -6.08713737  ,  13.39520243  ,   0.32422886 ]
    # c =  [ -0.40595224  ,  .58474125  ,  16.43906506  ]

    ra = np.sqrt(np.dot(a,a))
    rb = np.sqrt(np.dot(b,b))
    rc = np.sqrt(np.dot(c,c))

    alpha = np.arccos(np.dot(b,c)/(rb*rc))
    beta  = np.arccos(np.dot(c,a)/(rc*ra))
    gamma = np.arccos(np.dot(a,b)/(ra*rb))

    # print(alpha*180.0/3.14159,beta*180.0/3.14159,gamma*180.0/3.14159)
    return ra,rb,rc,alpha,beta,gamma

def get_max_index(index):
    if np.isscalar(index):
        return index
    elif isinstance(index, slice):
        return index.stop if (index.stop is not None) else float("inf")

def construct_cell(diagdisp, offdiag):
    """Help function to create an ASE-cell with displacement vector from
    the lammps coordination system parameters.

    :param diagdisp: cell dimension convoluted with the displacement vector
    :param offdiag: off-diagonal cell elements
    :returns: cell and cell displacement vector
    :rtype: tuple
    """
    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag

    # create ase-cell from lammps-box
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = zhi - zlo
    celldispx = xlo - min(0, xy) - min(0, xz)
    celldispy = ylo - min(0, yz)
    celldispz = zlo
    cell = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    celldisp = np.array([celldispx, celldispy, celldispz])
    return cell, celldisp

def lammpstraj_to_ase(filename='lammps.trj',index=-1,traj='md.traj', 
                      mode='w',units='real',
                      inp='in.lammps',atomid=None,recover=False):
    """Process cleartext lammps dumpfiles
    :param filename: trajectory file name
    :returns: list of Atoms objects
    :rtype: list
    """
    e = []
    with open('lmp.log','r') as fl:
         lines = fl.readlines()
         readenergy = False
         for line in lines:
             if line.find('Step          Temp          E_pair         TotEng')>=0:
                readenergy = True
                l = line.split()
                nt = len(l)
             elif line.find('Loop time')>=0 or line.find('ERROR')>=0:
                readenergy = False
             l = line.split()
             if readenergy and l[0]!='Step' and len(l)==nt:
                e.append(float(l[2])*4.3364432032e-2) # unit conver to eV
    with open(inp,'r') as fi:
         lines = fi.readlines()
         model = 'reaxff'
         for line in lines:
             if line.find('pair_style')>=0:
                if line.find('quip')>=0:
                   model = 'quip'
             if line.find('pair_coeff')>=0:
                l = line.split()
                # print(model)
                if model=='quip':
                   atomType = l[5:]
                   atomType = [chemical_symbols[int(a)] for a in atomType if a.find('\"')<0 and len(a)<=2]
                   # print(atomType)
                else:
                   atomType = l[4:]
    # Load all dumped timesteps into memory simultaneously
    with open(filename,'r') as ft:
         lines = deque(ft.readlines())
    index_end = get_max_index(index)

    n_atoms = 0
    images = []

    his = TrajectoryWriter(traj, mode=mode)

    # avoid references before assignment in case of incorrect file structure
    cell, celldisp, pbc = None, None, False
    i_ = 0

    while len(lines) > n_atoms:
        line = lines.popleft()

        if "ITEM: TIMESTEP" in line:
            n_atoms = 0
            line = lines.popleft()
            # !TODO: pyflakes complains about this line -> do something
            # ntimestep = int(line.split()[0])  # NOQA

        if "ITEM: NUMBER OF ATOMS" in line:
            line = lines.popleft()
            n_atoms = int(line.split()[0])

        if "ITEM: BOX BOUNDS" in line:
            # save labels behind "ITEM: BOX BOUNDS" in triclinic case
            # (>=lammps-7Jul09)
            tilt_items = line.split()[3:]
            celldatarows = [lines.popleft() for _ in range(3)]
            celldata = np.loadtxt(celldatarows)
            diagdisp = celldata[:, :2].reshape(6, 1).flatten()

            # determine cell tilt (triclinic case!)
            if len(celldata[0]) > 2:
                # for >=lammps-7Jul09 use labels behind "ITEM: BOX BOUNDS"
                # to assign tilt (vector) elements ...
                offdiag = celldata[:, 2]
                # ... otherwise assume default order in 3rd column
                # (if the latter was present)
                if len(tilt_items) >= 3:
                    sort_index = [tilt_items.index(i)
                                  for i in ["xy", "xz", "yz"]]
                    offdiag = offdiag[sort_index]
            else:
                offdiag = (0.0,) * 3

            cell, celldisp = construct_cell(diagdisp, offdiag)

            # Handle pbc conditions
            if len(tilt_items) == 3:
                pbc_items = tilt_items
            elif len(tilt_items) > 3:
                pbc_items = tilt_items[3:6]
            else:
                pbc_items = ["f", "f", "f"]
            pbc = ["p" in d.lower() for d in pbc_items]

        if "ITEM: ATOMS" in line:
            colnames = line.split()[2:]
            datarows = [lines.popleft() for _ in range(n_atoms)]
            data = np.loadtxt(datarows, dtype=str)
            if i_<len(e):
               out_atoms = lammps_data_to_ase_atoms(
                    data=data,
                    colnames=colnames,
                    cell=cell,
                    celldisp=celldisp,
                    atomsobj=Atoms,
                    pbc=pbc,
                    energy=e[i_],
                    atomType=atomType,
                    atomid=atomid,
                    units=units,
                    recover=False)
               images.append(out_atoms)
               his.write(atoms=out_atoms)
            i_ += 1
    his.close()
    return images[index]

def lammps_data_to_ase_atoms(
    data,
    colnames,
    cell,
    celldisp,
    pbc=False,
    atomsobj=Atoms,
    order=True,
    specorder=None,
    prismobj=None,
    units="real",
    energy=0.0,
    atomType=None,
    atomid=None,
    recover=False):
    """Extract positions and other per-atom parameters and create Atoms

    :param data: per atom data
    :param colnames: index for data
    :param cell: cell dimensions
    :param celldisp: origin shift
    :param pbc: periodic boundaries
    :param atomsobj: function to create ase-Atoms object
    :param order: sort atoms by id. Might be faster to turn off.
    Disregarded in case `id` column is not given in file.
    :param specorder: list of species to map lammps types to ase-species
    (usually .dump files to not contain type to species mapping)
    :param prismobj: Coordinate transformation between lammps and ase
    :type prismobj: Prism
    :param units: lammps units for unit transformation between lammps and ase
    :returns: Atoms object
    :rtype: Atoms

    """

    # read IDs if given and order if needed
    if "id" in colnames:
        ids = data[:, colnames.index("id")].astype(int)
        if order:
            sort_order = np.argsort(ids)
            data = data[sort_order, :]

    # determine the elements
    if "element" in colnames:
        # priority to elements written in file
        elements = data[:, colnames.index("element")]
    elif "type" in colnames:
        # fall back to `types` otherwise
        elements = data[:, colnames.index("type")].astype(int) -1
      
        # reconstruct types from given specorder
        if specorder:
            elements = [specorder[t - 1] for t in elements]
        elements = np.array(atomType)[elements]
 
    else:
        # todo: what if specorder give but no types?
        # in principle the masses could work for atoms, but that needs
        # lots of cases and new code I guess
        raise ValueError("Cannot determine atom types form LAMMPS dump file")

    def get_quantity(labels, quantity=None):
        try:
            cols = [colnames.index(label) for label in labels]
            if quantity:
                return convert(data[:, cols].astype(float), quantity,
                               units, "ASE")

            return data[:, cols].astype(float)
        except ValueError:
            return None

    # Positions
    positions = None
    scaled_positions = None
    if "x" in colnames:
        # doc: x, y, z = unscaled atom coordinates
        positions = get_quantity(["x", "y", "z"], "distance")
    elif "xs" in colnames:
        # doc: xs,ys,zs = scaled atom coordinates
        scaled_positions = get_quantity(["xs", "ys", "zs"])
    elif "xu" in colnames:
        # doc: xu,yu,zu = unwrapped atom coordinates
        positions = get_quantity(["xu", "yu", "zu"], "distance")
    elif "xsu" in colnames:
        # xsu,ysu,zsu = scaled unwrapped atom coordinates
        scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
    else:
        raise ValueError("No atomic positions found in LAMMPS output")

    velocities = get_quantity(["vx", "vy", "vz"], "velocity")
    charges = get_quantity(["q"], "charge")
    forces = get_quantity(["fx", "fy", "fz"], "force")
    # !TODO: how need quaternions be converted?
    quaternions = get_quantity(["c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])

    # convert cell
    cell = convert(cell, "distance", units, "ASE")
    celldisp = convert(celldisp, "distance", units, "ASE")
    if prismobj:
        celldisp = prismobj.vector_to_ase(celldisp)
        cell = prismobj.update_cell(cell)

    if quaternions:
        if not atomid is None:
           elements = elements[atomid[0]:atomid[1]]
           positions = positions[atomid[0]:atomid[1]]
           quaternions = quaternions[atomid[0]:atomid[1]]
        out_atoms = Quaternions(
            symbols=elements,
            positions=positions,
            cell=cell,
            celldisp=celldisp,
            pbc=pbc,
            quaternions=quaternions,
        )
    elif positions is not None:
        # reverse coordinations transform to lammps system
        # (for all vectors = pos, vel, force)
        if prismobj:
            positions = prismobj.vector_to_ase(positions, wrap=True)
        if not atomid is None:
           elements = elements[atomid[0]:atomid[1]]
           positions = positions[atomid[0]:atomid[1]]

        out_atoms = atomsobj(
            symbols=elements,
            positions=positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell
        )
    elif scaled_positions is not None:
        if not atomid is None:
           elements = elements[atomid[0]:atomid[1]]
           scaled_positions = scaled_positions[atomid[0]:atomid[1]]
           # quaternions = quaternions[atomid[0]:atomid[1]]
        out_atoms = atomsobj(
            symbols=elements,
            scaled_positions=scaled_positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )

    if velocities is not None:
        if prismobj:
            velocities = prismobj.vector_to_ase(velocities)
        if not atomid is None:
           velocities = velocities[atomid[0]:atomid[1]]
        out_atoms.set_velocities(velocities)
    if charges is not None:
        if not atomid is None:
           charges = np.squeeze(charges)[atomid[0]:atomid[1]]
        out_atoms.set_initial_charges(np.squeeze(charges))
    if forces is not None:
        if prismobj:
            forces = prismobj.vector_to_ase(forces)
        if not atomid is None:
           forces = forces[atomid[0]:atomid[1]]
        # !TODO: use another calculator if available (or move forces
        #        to atoms.property) (other problem: synchronizing
        #        parallel runs
        if recover:
           out_atoms = press_mol(out_atoms)
        calculator = SinglePointCalculator(out_atoms, energy=energy,
                                           forces=forces)
        out_atoms.calc = calculator

    # process the extra columns of fixes, variables and computes
    #    that can be dumped, add as additional arrays to atoms object
    for colname in colnames:
        # determine if it is a compute or fix (but not the quaternian)
        if (colname.startswith('f_') or colname.startswith('v_') or
                (colname.startswith('c_') and not colname.startswith('c_q['))):
            out_atoms.new_array(colname, get_quantity([colname]),
                                dtype='float')

    return out_atoms

def get_lammps_forces(atoms,lmp='lammps',
                      ffield='ffield',
                      pair_style='reaxff control nn yes checkqeq yes',
                      atom_style='charge',units='real',
                      ncpu=1,**kwargs):
    '''
    Using lammps to compute atoms object forces
    atoms: ASE atoms object
    lmp:   lammps run command,your lammps run command, for e.g., lammps, lmp, ./lmp_ubuntu ... 
    example, the use of GAP potential:
        pair_style    quip
        pair_coeff    * * Carbon_GAP_20_potential/Carbon_GAP_20.xml "" 6
    '''
    if pair_style.find('reaxff')>=0:
       units      = 'real'
       atom_style = 'charge'
    else:
       units      = 'metal'
       atom_style = 'atomic'

    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp      = ' '.join(species)
    writeLammpsData(atoms,data='data.lammps',specorder=None,
                    force_skew=False,
                    velocities=False,units=units,atom_style=atom_style)
    if 'pair_coeff' in kwargs:
       pair_coeff = kwargs['pair_coeff']
    else:
       pair_coeff =  '* * {:s} {:s}'.format(ffield,sp)
    writeLammpsIn(log='lmp.log',timestep=0.1,total=0,restart=None,
              species=species,
              pair_style =pair_style,  # without lg set lgvdw no
              pair_coeff =pair_coeff,
              fix = '  ',
              fix_modify = ' ',
              more_commond = ' ',
              thermo_style ='thermo_style  custom step temp epair etotal press vol pxx pyy pzz pxy pxz pyz',
              data='data.lammps',
              units=units,atom_style=atom_style,
              restartfile='restart')
    print('\n-  running lammps nvt ...')
    if ncpu==1:
       system('{:s} <in.lammps>out'.format(lmp))
    else:
       system('mpirun -n {:d} {:s} -i in.lammps>out'.format(ncpu,lmp))
    atoms = lammpstraj_to_ase('lammps.trj',units=units,inp='in.lammps')
    return atoms

if __name__ == '__main__':
  # relax system first
  eos = EOS(pair_coeff = '* * ffield.reax.lg C    H    O    N',
            pair_style = 'reaxff control lgvdw yes checkqeq yes',
            struc = 'hmx',
            supercell=[4,4,4],
            fac = 1.25,
            np=4)

  for i,T in enumerate([280]):
      restart = None if i==0 else 'restart.x'
      # restart = 'restart.x'
      eos.run(totalstep=2000,restart=restart,P=100.0,T=280.0,pfreq=10,tfreq=10)
      # eos.check_species(species=1)

  eos.run(totalstep=100000,restart='restart.x',P=100.0,T=280.0,pfreq=1000,tfreq=1000)




