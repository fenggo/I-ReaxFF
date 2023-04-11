from __future__ import print_function
from os import system
from .emdk import get_structure,emdk
# from .findmole import check_decomposed
from ase.io import read,write
from ase import Atoms
from ase.calculators.lammpsrun import write_lammps_data
from ase.calculators.lammps import Prism,convert
from ase.io.trajectory import TrajectoryWriter
from .molecule import molecules,enlarge
from .getNeighbors import get_neighbors
import matplotlib.pyplot as plt
import numpy as np



def run_lammps(inp_name='inp-lam',label='eos',np=4):
    print('mpirun -n %d lammps<%s> %s.out \n' %(np,inp_name, label))
    system('mpirun -n %d lammps<%s> %s.out' %(np,inp_name, label))


def check_decomposed(traj='lammps.trj',nmol=1):
    mol_ = findmole(filename=traj,trjtype=1,
               frame=1000000000,timeinterval=0.005,runtype=2) # check molecule if decomposed
    mol  = mol_[-1]
    nmol_= len(mol)
    if nmol_>nmol:
       print('-  structure already decomposed, exit now!')
       send_msg('-  structure already decomposed, exit now!')
       exit()


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
    plt.savefig('energy.eps') 
    plt.close()

    plt.figure()
    plt.ylabel(r'$Temperature$ ($T$)')
    plt.xlabel(r'$Step$')
    plt.plot(steps,t,label=r'$Temperature .vs. step$', color='black', linewidth=1.5, linestyle='-.')
    plt.legend()
    plt.savefig('temperature.eps') 
    plt.close()

    plt.figure()
    plt.ylabel(r'$Lattice constant$ ($Angstrom$)')
    plt.xlabel(r'$Step$')
    plt.plot(steps,a,label=r'$a$', color='red', linewidth=1.5, linestyle='--')
    plt.plot(steps,b,label=r'$b$', color='blue', linewidth=1.5, linestyle='-.')
    plt.plot(steps,c,label=r'$c$', color='black', linewidth=1.5, linestyle=':')
    plt.legend()
    plt.savefig('lattice.eps') 
    plt.close()

    plt.figure()
    plt.ylabel(r'$Angle constant$ ($degree$)')
    plt.xlabel(r'$Step$')
    plt.plot(steps,alpha,label=r'$alpha$', color='red', linewidth=1.5, linestyle='--')
    plt.plot(steps,beta,label=r'$beta$', color='blue', linewidth=1.5, linestyle='-.')
    plt.plot(steps,gamma,label=r'$gamma$', color='black', linewidth=1.5, linestyle=':')
    plt.legend()
    plt.savefig('angle.eps') 
    plt.close()
    return step,N,t0,p0,e0,v0,aa,ba,ca,alpha_a,beta_a,gamma_a  # N ,atoms number, t0 temperature, p0 pressure, e0 energy v0,volume


def writeLammpsData(atoms,data='data',specorder=None, 
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
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

    f.write("{0} (written by ASE) \n\n".format(f.name))

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



def writeLammpsIn(log='lmp.log',timestep=0.1,total=200, data= None,restart=None,
              pair_coeff ='* * ffield.reax.lg C H O N',
              pair_style = 'reax/c control.reax lgvdw yes checkqeq yes',  # without lg set lgvdw no
              fix = 'fix   1 all npt temp 800 800 100.0 iso 10000 10000 100',
              fix_modify = ' ',
              more_commond = ' ',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              restartfile='restart.eq'):
    '''
        pair_style     reax/c control.reax checkqeq yes
        pair_coeff     * * ffield.reax.rdx C H O N
        ---control.reax---
        simulation_name         enery_material  ! output files will carry this name + their specific extension

        tabulate_long_range     0 ! denotes the granularity of long range tabulation, 0 means no tabulation
        energy_update_freq      0
        remove_CoM_vel          500 ! remove the trans. & rot. vel around the CoM every 'this many' steps

        nbrhood_cutoff          4.0  ! near neighbors cutoff for bond calculations in A
        hbond_cutoff            10.0  ! cutoff distance for hydrogen bond interactions
        bond_graph_cutoff       0.3  ! bond strength cutoff for bond graphs
        thb_cutoff              0.01 ! cutoff value for three body interactions
        q_err                   1e-6  ! average per atom error norm allowed in GMRES convergence

        geo_format              0    ! 0: xyz, 1: pdb, 2: bgf
        write_freq              50   ! write trajectory after so many steps
        traj_compress           0    ! 0: no compression  1: uses zlib to compress trajectory output
        traj_title              dump ! (no white spaces)
        atom_info               0    ! 0: no atom info, 1: print basic atom info in the trajectory file
        atom_forces             0    ! 0: basic atom format, 1: print force on each atom in the trajectory file
        atom_velocities         0    ! 0: basic atom format, 1: print the velocity of each atom in the trajectory file
        bond_info               0    ! 0: do not print bonds, 1: print bonds in the trajectory file
        angle_info              0    ! 0: do not print angles, 1: print angles in the trajectory file 
    '''
    fin = open('inp-lammps','w')
    print('units       real', file=fin)
    print('atom_style  charge', file=fin)
    if data != None and data != 'None':
       print('read_data    %s' %data, file=fin)
    if restart != None and restart != 'None':
       print('read_restart %s' %restart, file=fin)
    print(' ', file=fin)
    print('pair_style     %s'  %pair_style, file=fin) 
    print('pair_coeff     %s'  %pair_coeff, file=fin)
    print('compute       reax all pair reax/c', file=fin)
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
    print(fix, file=fin)
    print(fix_modify, file=fin)
    print('fix    rxc all qeq/reax 1 0.0 10.0 1.0e-6 reax/c', file=fin)
    print(' ', file=fin)
    print(more_commond, file=fin)
    print(' ', file=fin)
    print('thermo        50', file=fin)
    print(thermo_style, file=fin)
    print(' ', file=fin)
    print('timestep      %f' %timestep, file=fin)
    print(' ', file=fin)
    print('dump          1 all custom 100 lammps.trj id type x y z q', file=fin)
    print(' ', file=fin)
    print('log           %s'  %log, file=fin)
    print(' ', file=fin)
    print('restart       10000 restart', file=fin)
    print('run           %d'  %total, file=fin)
    print(' ', file=fin)
    print('write_restart %s'  %restartfile, file=fin)
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


  def check_species(self,species=1):
      check_decomposed(traj='lammps.trj',nmol=species)


def lattice(a,b,c):
    # a =  [ 14.90415061,    -0.12901043,   0.43404866 ]
    # b =  [  -6.08713737  ,  13.39520243  ,   0.32422886 ]
    # c =  [ -0.40595224  ,  -1.58474125  ,  16.43906506  ]

    ra = np.sqrt(np.dot(a,a))
    rb = np.sqrt(np.dot(b,b))
    rc = np.sqrt(np.dot(c,c))

    alpha = np.arccos(np.dot(b,c)/(rb*rc))
    beta  = np.arccos(np.dot(c,a)/(rc*ra))
    gamma = np.arccos(np.dot(a,b)/(ra*rb))

    # print(alpha*180.0/3.14159,beta*180.0/3.14159,gamma*180.0/3.14159)
    return ra,rb,rc,alpha,beta,gamma


def LammpsHistory(traj='ase.lammpstrj',frame=0,atomType=['C','H','O','N']):
    fl = open(traj,'r')
    lines = fl.readlines()
    nl    = len(lines) 
    fl.close()
    natom     = int(lines[3])

    his       = TrajectoryWriter(traj.split('.')[0]+'.traj',mode='w')
    n         = 0
    block     = natom+9

    atomName  = [' ' for i in range(natom)]
    cell      = np.zeros([3,3])
    line      = lines[block*frame + 5].split()
    cell[0][0]= float(line[1])-float(line[0])
    line      = lines[block*frame + 6].split()
    cell[1][1]= float(line[1])-float(line[0])
    line      = lines[block*frame + 7].split()
    cell[2][2]= float(line[1])-float(line[0])

    positions = np.zeros([natom,3])

    while n<=nl:    
        for i in range(natom):
            n = block*frame + i + 9
            line = lines[n].split()
            id_  = int(line[0])-1
            atomName[id_]=atomType[int(line[1])-1]
            positions[id_][0] = float(line[2])
            positions[id_][1] = float(line[3])
            positions[id_][2] = float(line[4])
            
        atoms  = Atoms(atomName,positions,cell=cell,pbc=[True,True,True])
        his.write(atoms=atoms)
        frame += 1
        n = block*frame + 9

    his.close()
    lines= None
    return atoms


if __name__ == '__main__':
  # relax system first
  eos = EOS(pair_coeff = '* * ffield.reax.lg C    H    O    N',
            pair_style = 'reax/c control.reax lgvdw yes checkqeq yes',
            struc = 'hmx',
            supercell=[4,4,4],
            fac = 1.25,
            np=4)

  for i,T in enumerate([280]):
      restart = None if i==0 else 'restart.x'
      # restart = 'restart.x'
      eos.run(totalstep=2000,restart=restart,P=100.0,T=280.0,pfreq=10,tfreq=10)
      eos.check_species(species=1)

  eos.run(totalstep=100000,restart='restart.x',P=100.0,T=280.0,pfreq=1000,tfreq=1000)




