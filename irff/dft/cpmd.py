from __future__ import print_function
from os import system,getcwd,chdir 
from os.path import isfile,exists
from ..molecule import enlarge,molecules,get_neighbors,press_mol
from ..structures import structure
# from .dingtalk import send_msg
from ase.io import read,write
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt

# example:
#N = 50
#x = np.random.rand(N)
#y = np.random.rand(N)
#colors = np.random.rand(N)
#area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
#plt.show()

#EKINC fictitious kinetic energy of the electrons in a.u. this quantity should oscillate but not
#increase during a simulation.
#TEMPP Temperature of the ions, calculated from the kinetic energy of the ions (EKIONS).

#EKS Kohn-Sham energy (the equivalent of the potential energy in classical MD).
#ECLASSIC = EKS + EKIONS
#EHAM = ECLASSIC + EKINC. Hamiltonian energy, this is the conserved quantity, depending
#on the time step and the electron mass, this might oscillate but should not drift.


def write_cpmd_inp(A,runtype='md',cellopt=True,
                   ensemble='nve',
                   T=200,Ttor=50,TE=0.002,step=20000,
                   functional='REVPBE',pw_cutoff=100.0,
                   restart=None,
                   psp=None,
                   electron=None,
                   wall=False,lw=2.0):
    if psp is None:
       psp = {'C':'C_MT_revPBE.psp','H':'H_MT_revPBE.psp','N':'N_MT_revPBE.psp',
           'O':'O_MT_revPBE.psp','Fe':'Fe_MT_PBE_NLCC.psp'}       #pseudopotential
    if electron is None:
       electron = {'C':'LMAX=P LOC=P','H':'LMAX=S LOC=S','N':'LMAX=P LOC=P',
           'O':'LMAX=P LOC=P','Fe':'LMAX=F'}
    nspec = {}
    cell = A.get_cell()
    # cell[cell<0.0000001] = 0.0 # or
    cell = np.where(cell>0.0000001,cell,0.0)

    r = [0.0 for i in range(3)]
    r[0] = cell[0][0]*cell[0][0]+ cell[0][1]*cell[0][1]+ cell[0][2]*cell[0][2]
    r[0] = np.sqrt(r[0])
    r[1] = cell[1][0]*cell[1][0]+ cell[1][1]*cell[1][1]+ cell[1][2]*cell[1][2]
    r[1] = np.sqrt(r[1])
    r[2] = cell[2][0]*cell[2][0]+ cell[2][1]*cell[2][1]+ cell[2][2]*cell[2][2]
    r[2] = np.sqrt(r[2])

    atoms_label = A.get_chemical_symbols()
    species     = []
    for lab in atoms_label:
        if not lab in species:
           species.append(lab)
           nspec[lab] = 0
    for s in species:
        for lab in atoms_label:
            if lab==s:
               nspec[s] += 1

    nspecie = len(species)
    natm    = len(atoms_label)

    if runtype=='wvopt' or runtype=='geometry':
       filename = 'inp-'+runtype  
    else:
       filename ='inp-'+ensemble
       
    fin = open(filename,'w')

    print('\n',file=fin)
    print('&CPMD',file=fin)
    if runtype=='md':
       print('  MOLECULAR DYNAMICS CP',file=fin)
       # WAVEFUNCTION COORDINATES VELOCITIES CELL ACCUMULATORS NOSEE NOSEP LATEST
       if not restart is None:
          print('  RESTART %s' %restart, file=fin)
       print('  EMASS',file=fin)
       print('    270',file=fin)
       if ensemble=='nve':
          print('  TEMPERATURE',file=fin)
          print('    %d' %T,file=fin)
       elif ensemble=='nvt':
          print('  NOSE IONS',file=fin)
          print('    %7.2f  %d' %(T,Ttor),file=fin)
          print('  NOSE ELECTRONS',file=fin)
          print('    %7.2f 10000' %TE,file=fin)
       print('  TIMESTEP',file=fin)
       print('    2.5   ',file=fin)    #### =0.06 fs
       print('  MAXSTEP',file=fin)
       print('    %d' %step,file=fin)
       print('  VDW CORRECTION ON',file=fin)
       print('  STRESS TENSOR',file=fin)
       print('    100',file=fin)
       print('  STORE',file=fin)
       print('    100',file=fin)  
       print('  TRAJECTORY FORCES',file=fin) 
    elif runtype=='wvopt':
       print('  OPTIMIZE WAVEFUNCTION',file=fin)
       # WAVEFUNCTION COORDINATES CELL LATEST
       # WAVEFUNCTION COORDINATES LATEST
       if not restart is None:
          print('  RESTART %s' %restart, file=fin)

       print('  PCG MINIMIZE',file=fin) # or ODIIS
       print('  MAXSTEP',file=fin)
       print('    500',file=fin)

       print('  CONVERGENCE ORBITALS',file=fin)
       print('    4.0d-9',file=fin) 
    elif runtype=='geometry':
       print('  OPTIMIZE GEOMETRY',file=fin)
       print('  RESTART WAVEFUNCTION COORDINATES LATEST',file=fin)
       if cellopt:
          print('  STEEPEST DESCENT CELL',file=fin)
       print('  ODIIS',file=fin) # or ODIIS
       print('    15',file=fin) # or ODIIS
       print('  MAXSTEP',file=fin)
       print('    500',file=fin)
    print('&END\n',file=fin)

    print('&DFT',file=fin)
    print('  FUNCTIONAL %s' %functional,file=fin)   # PBE0
    print('  GC-CUTOFF',file=fin) 
    print('    1.0D-8',file=fin)
    print('&END\n',file=fin)
    print('\n',file=fin)

    print('&VDW',file=fin)
    print('  VDW CORRECTION',file=fin) 
    print('  ALL DFT-D2',file=fin)
    print('&END\n',file=fin)
    print('\n',file=fin)

    print('&SYSTEM',file=fin)
    print('  ANGSTROM',file=fin) 

    if wall:
       print('  SYMMETRY',file=fin)
       print('   0',file=fin)
       print('  CELL',file=fin)
       print('  ',cell[0][0],cell[1][1]/cell[0][0],
                  cell[2][2]/cell[0][0],
                  0.0,0.0,0.0,file=fin)
       print('  BOX WALLS',file=fin)
       print('  ',lw,file=fin)
    else:
       print('  CELL VECTORS',file=fin)
       for c in cell:
           print('  ',c[0],c[1],c[2],file=fin)

    print('  CUTOFF',file=fin) 
    print('    %f' %pw_cutoff,file=fin)
    print('&END\n',file=fin)
    print('\n',file=fin)

    print('&ATOMS',file=fin)
    for s in species:
        print('*%s KLEINMAN-BYLANDE' %psp[s],file=fin)
        print(electron[s],file=fin)
        print(nspec[s],file=fin)
        for a in A:
            if a.symbol==s:
               print('  ',a.x,a.y,a.z,file=fin)
    print('&END\n',file=fin)
    print('\n',file=fin)
    fin.close()


def collect_data(strucs=['cl20mol','hmxmol'],T=200,step=20000,np=12):
    ''' gather datas '''
    root = getcwd() 
    restart_l = 'WAVEFUNCTION COORDINATES VELOCITIES CELL ACCUMULATORS NOSEE NOSEP LATEST'
    for s in strucs:
        for i in range(5):
            dirc = s+'_'+str(i)
            system('mkdir '+dirc)
            chdir(dirc)
            if i==0:
               system('cp ../*.psp ./')
               A = structure(s)
               write_cpmd_inp(A,runtype='wvopt',
                              restart=None)
               print('*  optimizing wave functions of %s ...' %s)
               system('mpirun -n %d cpmd inp-wvopt>wv.out' %np)
               write_cpmd_inp(A,runtype='md',ensemble='nve',T=T,step=step,
                  restart=restart_l)
            else:
               system('cp ../'+s+'_'+str(i-1)+'/*.psp ./')
               system('cp ../'+s+'_'+str(i-1)+'/RESTART.1 ./')
               system('cp ../'+s+'_'+str(i-1)+'/LATEST ./')
               write_cpmd_inp(A,runtype='wvopt',
                              restart=restart_l)
               print('*  optimizing wave functions of %s ...' %s)
               system('mpirun -n %d cpmd inp-wvopt>wv.out' %np)

               write_cpmd_inp(A,runtype='md',ensemble='nvt',T=T,step=step,
                  restart=restart_l)
            print('*  Car-Parinnello MD simulations ...')
            system('mpirun -n %d cpmd inp-nvt>nvt.out' %np)
            chdir(root)


def opt_and_md(struc='cl20',file='card.gen',supercell=[1,1,1],
               T=3000,step=20000,
               wall=False,
               restart=None,
               psp=None,functional='REVPBE',
               restart_nve='WAVEFUNCTION COORDINATES LATEST',
               ncpu=12):
    if file=='card.gen':
       A = structure(struc)
    else:
       A = read(file)
    run_cpmd(A,res_opt=restart,res_md=restart_nve,wall=wall,
             psp=psp,functional=functional,ncpu=ncpu)


def compress(struc='cl20',file=None,supercell=[1,1,1],
             T=200,TE=0.002,step=20000,
             comp=[1.0,1.0,1.0],
             res_opt=None,
             res_md='WAVEFUNCTION COORDINATES CELL LATEST',
             ncpu=12):
    if not file is None:
       A = read(file)
       cell = A.get_cell()
    else:
       A = read('GEOMETRY.xyz')
       cell = get_lattice(inp='inp-nve')
       A.set_pbc([True,True,True])
       A.set_cell(cell)

    comp = np.reshape(np.array(comp),[3,1])
    cell = comp*cell

    A.set_cell(cell)
    run_cpmd(A,res_opt=res_opt,res_md=res_md,ncpu=ncpu)


def run_cpmd(A=None,res_opt=None,res_md='ALL LATEST',
            T=3000,step=20000,wall=False,
            psp=None,functional='REVPBE',
            ncpu=8):
    write_cpmd_inp(A,runtype='wvopt',
                  restart=res_opt,psp=psp,functional=functional,
                  wall=wall)
    system('rm ENERGIES TRAJECTORY STRESS')
    print('*  optimizing wave functions ...')
    # send_msg('*  optimizing wave functions ...')
    system('mpirun -n %d cpmd inp-wvopt>wv.out' %ncpu)

    write_cpmd_inp(A,runtype='md',ensemble='nve',T=T,step=step,
                   wall=wall,psp=psp,functional=functional,
                   restart=res_md)
    print('*  Car-Parinnello MD simulations ...')
    # send_msg('*  Car-Parinnello MD simulations ...')
    system('mpirun -n %d cpmd inp-nve>nve.out' %ncpu)
    N,temp,etot,mass,volu = cpmdplot(cpmdout='nve.out')
    P = get_press()
    # send_msg('*  CPMD job have completed!')


def scale(A=None,supercell=[1,1,1],comp=1.0,
          restart='WAVEFUNCTION COORDINATES LATEST'):
    if A is None:
       A = read('GEOMETRY.xyz')
       cell = get_lattice(inp='inp-nve')
       cell = [[a*comp for a in r] for r in cell]
       A.set_pbc([True,True,True])
       A.set_cell(cell)

    pos  = A.get_positions()
    box  = np.array([[cell[0][0],cell[1][1],cell[2][2]]])
    pos  = np.mod(pos,box)  

    A.set_positions(pos)
    A.write('scale.gen')
    # run_cpmd(A,res_opt=None,res_md='WAVEFUNCTION COORDINATES LATEST')
    return A


def cpmdnvt(np=12):
    print('*  Car-Parinnello MD simulations ...')
    system('rm ENERGIES  GEOMETRY  GEOMETRY.xyz TRAJECTORY STRESS')
    system('mpirun -n %d cpmd inp-nvt>nvt.out' %np)
    N,temp,etot,mass,volu = cpmdplot(cpmdout='nvt.out')
    P = get_press()


def cpmdplot(cpmdout='cpmd.log'):
    fname = cpmdout #str(raw_input('Enter the CPMD out file name:'))
    fout = open(fname,'r')

    lines = fout.readlines()

    readatom = False; mass = 0.0; N=0; volu=0.0
    masses = {'C':12.00,'H':1.008,'N':14.0,'O':15.999,'Fe':55.845}

    for iline, line in enumerate(lines):
        if line.find('NFI    EKINC   TEMPP           EKS      ECLASSIC          EHAM         DIS    TCPU') >= 0:
           sline = iline+1
        if line.find('RESTART INFORMATION WRITTEN ON FILE')>= 0:
           eline = iline
        if line.find(' ***************************** ATOMS ****************************')>= 0:
           readatom = True
        elif line.find('****************************************************************')>=0:
           readatom = False
        if readatom:
           if line.split()[0] != 'NR' and line.split()[1] != 'ATOMS' :
              mass += masses[line.split()[1]]
              N += 1  # atoms number
        if line.find('INITIAL VOLUME')>=0:
           volu = float(line.split()[-1])*(0.52917721067**3)


    NFI = []; EKINC = []; TEMPP = []; EKS = []; ECLASSIC = []; EHAM = []; DIS = []; TCPU = []
    etot = 0.0; temp = 0.0; c = 0

    for i in range(sline,eline):
        if len(lines[i].split())==8 and lines[i].split()[0] \
            != 'FILE' and lines[i].split()[0] != '=' and lines[i].split()[0] != 'RESTART':
           #print lines[i]
           #print sline,eline
           NFI.append(int(lines[i].split()[0]))
           EKINC.append(float(lines[i].split()[1]))
           TEMPP.append(float(lines[i].split()[2]))
           temp += float(lines[i].split()[2])

           EKS.append(float(lines[i].split()[3]))
           ECLASSIC.append(float(lines[i].split()[4]))
           etot += float(lines[i].split()[4])

           EHAM.append(float(lines[i].split()[5]))
           DIS.append(float(lines[i].split()[6]))
           TCPU.append(float(lines[i].split()[7]))
           c += 1

    temp = temp/float(c)
    etot = etot/float(c)

    plt.figure()
    plt.ylabel(r'$Energy$')
    plt.xlabel(r'$Time Step$')

    pl1, =plt.plot(NFI,EKS,label=r'$EKS$', linewidth=2.0)
    pl2, =plt.plot(NFI,ECLASSIC,label=r'$ECLASSIC$',color='red', linewidth=2.0,
                   linestyle='-.')
    pl3, =plt.plot(NFI,EHAM,label=r'$EHAM$',color='blue', linewidth=2.0,
                   linestyle='--')

    # linestyles: '-' or 'solid','--' or 'dashed', '-.' or 'dashdot'
    #             ':' or 'dotted', 'None', ' ' draw nothing

    plt.legend(handles=[pl1, pl2,pl3], labels=[r'$EKS$', r'$ECLASSIC$',r'$EHAM$'],  loc='best')
    plt.savefig('cpmd_energy.eps') 

    plt.figure()             # temperature
    plt.ylabel(r'$Energy$')
    plt.xlabel(r'$Time Step$')
    pl1, =plt.plot(NFI,TEMPP,label=r'$Temperature$', linewidth=2.0)
    plt.legend(handles=[pl1], labels=[r'$Temperature$'], loc='best')
    plt.savefig('tempp.eps') 
    plt.close()
    return N,temp,etot,mass,volu


def get_press():
    pressure = 0.0
    if isfile('STRESS'):
       fs = open('STRESS','r')
       c = 0
       ts=[]; pxx=[];pyy=[];pzz=[]; PP = [];
       lines = fs.readlines()
       for iline, line in enumerate(lines):
           if line.find('TOTAL STRESS TENSOR (kB):') >= 0:
              if int(line.split()[-1])==1:
                 c = 0
                 pressure = 0.0; ts=[]; pxx=[];pyy=[];pzz=[]; PP = []
              #print iline
              #print lines[iline]
              px =float(lines[iline+1].split()[0])
              py =float(lines[iline+2].split()[1])
              pz =float(lines[iline+3].split()[2])
              pxx.append(px)
              pyy.append(py)
              pzz.append(pz)
              P = px + py + pz
              P = P*1.0e8/3.0
              pressure += P
              PP.append(-P)
              ts.append(c)
              c += 1
       fs.close()

       pressure = -pressure/float(c)  # pressure & stress tensor

       plt.figure()              # press
       plt.ylabel(r'$press$')
       plt.xlabel(r'$Time Step$')
       pl1, =plt.plot(ts,pxx,label=r'$pxx$',color='red', linewidth=2.0)
       pl2, =plt.plot(ts,pyy,label=r'$pyy$',color='blue', linewidth=2.0)
       pl3, =plt.plot(ts,pzz,label=r'$pzz$',color='green', linewidth=2.0)
       plt.legend(handles=[pl1,pl2,pl3], labels=[r'$pxx$',r'$pyy$',r'$pzz$'], loc='best')
       plt.savefig('press.eps')

       plt.figure()              # total pressure
       plt.ylabel(r'$Pressure$')
       plt.xlabel(r'$Time Step$')
       plt.plot(ts,PP,label=r'$pressure$', linewidth=2.0)
       plt.legend(loc='best')
       plt.savefig('pressure.eps')
    return pressure
  

def hugstate(p0=0.0,v0=0.0,e0=0.0,t0=0.0,cpmdout='nve-cpmd.log'):
    N,temp,etot,mass,volu = cpmdplot(cpmdout=cpmdout)
    dens = (mass/volu)*10.0000/6.02253
    pressure = get_press()

    dt = (0.5*(pressure+p0)*(v0-volu)*1.0e-12 + (e0 - etot)*4.359710)/((3*N-3)*1.381*1.0e-5)
    print('-------- Average values of current stage: ---------')
    print('* Average energy(ECLASSIC): %f' %etot)
    print('* Average Temperature: %f' %temp)
    print('* Average Pressure: %f' %pressure)
    print('* The volum of system: %f' %volu)
    print('* Temperature devation is %f K' %(dt))
    print('* Target temperature is %f K' %(dt+temp))
    print('* The density is %f g/cm^3' %dens)
    print('---------------------------------------------------')


def get_lattice(inp='inp-nve'):
    finp = open(inp,'r')
    il = 0
    cell = []
    readlatv = False
    readlat  = False
    for line in finp.readlines():
        l = line.split()
        if line.find('CELL')>=0 and  line.find('VECTORS')>=0:
           readlatv = True
        elif line.find('CELL')>=0 and  line.find('VECTORS')<0:
           if line.find('RESTART')<0:
              readlat  = True
        if readlatv and il < 4:
           if not il==0:
              cell.append( [float(l[0]),float(l[1]),float(l[2])])
           il += 1
        if readlat and il < 2:
           if not il==0:
              cell = [[float(l[0]),0.0,0.0],
                      [0.0,float(l[1])*float(l[0]),0.0],
                      [0.0,0.0,float(l[2])*float(l[0])]]
           il += 1
    finp.close()
    return cell


if __name__ == '__main__':
   from ase.io import read,write
   from ase import Atoms
   from cpmd import write_cpmd_inp
   supercell = [1,1,1]
   struc = 'cl20mol'

   A = structure(struc)
   # emdk(cardtype='xyz',cardfile='siesta.xyz',
   #  cell=[[10.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]],
   #  output='dftb',
   #  center='.True.',log='log') 

   write_cpmd_inp(A,runtype='md',ensemble='nve')
   # write_cpmd_inp(A,runtype='md',ensemble='nvt',restart=True)


