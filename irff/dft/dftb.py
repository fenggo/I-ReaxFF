#!/usr/bin/env python
from __future__ import print_function
from os import system, getcwd, chdir,listdir
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read,write
from ase.io.dftb import write_dftb_velocities
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
# from dingtalk import send_msg
# libatlas-base-dev  arlapack dftb install must install atlas first


def mass(element):
    mas = {'C':12.000,'H':1.008,'N':14.000,'O':15.999,'Fe':55.845}
    if element in mas:
       return mas[element]
    else:
       return 0.0


def write_polynomial(skf='C-C',rcut=3.9,c=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],direc='./'):
    if direc!='./':
       cdir = getcwd()
       chdir(direc)
    system('mv '+skf+ '.skf' + ' '+ skf+ '.skf.b')
    fb  = open(skf+'.skf.b','r')
    fskf= open(skf+'.skf','w')

    if skf.split('-')[0]==skf.split('-')[1]:
       p = 2
    else:
       p = 1
    for i,line in enumerate(fb.readlines()):
        if i==p:
           print(mass(skf.split('-')[0]), \
              c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],rcut,'10*0.0', file=fskf)
        else:
           print(line[:-1], file=fskf)

    fb.close()
    fskf.close()
    if direc!='./':
       chdir(cdir)


def zero_spline(direc=None):
    cdir = getcwd()
    if not direc is None:
       chdir(direc)
       outs = listdir(direc)
    else:
       outs = listdir(cdir)

    for skf in outs:
        if skf.find('.b')<0 and skf.find('.skf')>=0:
           write_zero_spline(skf=skf,direc='./')


def write_zero_spline(skf='C-C.skf',direc='./'):
    if direc!='./':
       cdir = getcwd()
       chdir(direc)
    system('mv '+skf + ' '+ skf+ '.b')
    fb  = open(skf+'.b','r')
    fskf= open(skf,'w')

    sline = False; p = 0
    for i,line in enumerate(fb.readlines()):
        if len(line.split())>=1:
           if line.split()[0] == 'Spline':
              #sline_s = i
              sline = True
           elif line.split()[0] == '<Documentation>':
              sline = False
              #sline_e = i
        if not sline:
           print(line[:-1], file=fskf)
        else:
           if p==0:
              p = 1
              print('Spline', file=fskf)
              print('5 4.0', file=fskf)
              print('0.0 0.0 0.0', file=fskf)
              print('1.0 1.5 0.0 0.0 0.0 0.0', file=fskf)
              print('1.5 2.0 0.0 0.0 0.0 0.0', file=fskf)
              print('2.0 2.5 0.0 0.0 0.0 0.0', file=fskf)
              print('2.5 3.0 0.0 0.0 0.0 0.0', file=fskf)
              print('3.0 3.5 0.0 0.0 0.0 0.0', file=fskf)
              print('3.5 4.0 0.0 0.0 0.0 0.0 0.0 0.0', file=fskf)

    fb.close()
    fskf.close()
    if direc!='./':
       chdir(cdir)


def read_splines(skf='C-C'):
    fs = open(skf+'.skf','r')
    reads = False
    spline = []
    for line in fs.readlines():
        if line.find('Spline')>=0:
           reads = True
        if line.find('Spline')<0 and len(line.split())<2:
           reads = False
        if reads and line.find('Spline')<0:
           if len(line.split())==2:
              rcut  = line.split()[1]
           if len(line.split())==6:
              s = []
              for ss in line.split():
                  s.append(float(ss)) 
              spline.append(s)
    fs.close()
    return spline


def get_initial_poly(skf='C-C'):
    fs = open(skf+'.skf','r')
    reads = False
    spline = []
    for line in fs.readlines():
        if line.find('Spline')>=0:
           reads = True
        if line.find('Spline')<0 and len(line.split())<2:
           reads = False
        if reads and line.find('Spline')<0:
           if len(line.split())==2:
              rcut  = line.split()[1]
           if len(line.split())==6:
              s = []
              for ss in line.split():
                  s.append(float(ss)) 
              spline.append(s)
    fs.close()

    x = np.linspace(2.3,3.6,50)
    y = []
    for r in x:
        y.append(get_spline_3(r,spline))
    p0 = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]

    def sque(p,rcut=3.9,x=x):
        squ = 0.0
        for r in x:
            y1 = get_spline_3(r,spline)
            y2 = p[0]*(rcut-r)**2 + p[1]*(rcut-r)**3 + p[2]*(rcut-r)**4 + \
              p[3]*(rcut-r)**5 + p[4]*(rcut-r)**6 + p[5]*(rcut-r)**7 + \
              p[6]*(rcut-r)**8 + p[7]*(rcut-r)**9
            squ += (y1 - y2)**2
        return squ

    result = minimize(sque, p0, method ='BFGS',tol = 0.0000001,
                  options={'disp':True,'maxiter': 1000000000})

    yp = polynomial_8(result.x,x,rcut=3.9)
    # plt.plot(x,yp,label=r'$fitted$ $by$ $polynomial$',
    #          color='blue', linewidth=1.5, linestyle=':')
    # plt.legend(loc='best')
    # plt.savefig('spline.eps')
    return result.x


def write_dftb_in(coordinate='dftb.gen',
                  runtype = 'energy',   # canbe energy, opt, md ...
                  ensemble='nvt',
                  step=2000,dt=0.1,T=300,timescale=500,
                  AdaptFillingTemp='No',P=0.0,
                  latopt = 'yes',
                  label = 'dftb',
                  restart=10,
                  velocities=None,
                  maxf  = 1.0e-5,
                  maxam = {'C':'p','H':'s','O':'p','N':'p'},
                  Hubbard={'C':-0.1492,'H': -0.1857,'O':-0.1575,'N': -0.1535},
                  maxscc=1000,
                  readinitialcharges = 'No',
                  polynomial = {},
                  dispersion = None, # dftd3
                  thirdorder = 'True',  # must
                  analysis = True,
                  skf_dir = './' ,
                  w_poly = True):
    ''' prepare dftb input '''

    fin = open('dftb_in.hsd','w')
    print('Geometry = GenFormat { ', file=fin)
    print('   <<< "%s"' %coordinate, file=fin)
    print('}', file=fin)
    print(' ', file=fin)
    print(' ', file=fin)
    if runtype=='energy':
         print('Driver = {} ', file=fin)
    elif runtype=='opt':
         print('Driver = LBFGS{ ', file=fin)  # ConjugateGradient     
         print('    LatticeOpt = %s' %latopt, file=fin)
         print('    OutputPrefix = %s' %label, file=fin)
         print('    MaxForceComponent = %f' %maxf, file=fin) 
         print('    MaxSteps = 2000 }', file=fin)
    elif runtype=='md':
         print('Driver = VelocityVerlet{ ', file=fin)  # ConjugateGradient     
         print('    Steps = %d' %step, file=fin)
         print('    TimeStep [Femtosecond]  = %f' %dt, file=fin)
         print('    Thermostat = Berendsen {' , file=fin) 
         print('        Temperature [Kelvin] = %f' %T, file=fin)
         print('        Timescale = %d' %timescale, file=fin)
         print('        AdaptFillingTemp = %s' %AdaptFillingTemp, file=fin)
         print('       }', file=fin)
         if not velocities is None:
            print('    Velocities =  { ' , file=fin)
            print('         <<+ " %s" '  %velocities, file=fin)
            print('                       } ' , file=fin)
         if ensemble=='npt':
            print('    Barostat = { ', file=fin) 
            print('        Pressure [pa] = %f' %P, file=fin)
            print('        Timescale = %d}' %timescale, file=fin)
         print('    OutputPrefix = %s' %label, file=fin)  # ConjugateGradient     
         print('    MDRestartFrequency = %d' %restart, file=fin)
         print('    MovedAtoms = "1:-1"', file=fin)
         print('    KeepStationary = Yes', file=fin) 
         print('    ConvergentForcesOnly = Yes', file=fin)
         print('}', file=fin)
    print(' ', file=fin)
    print(' ', file=fin)
    print('Hamiltonian = DFTB{ ', file=fin)
    print('    KPointsAndWeights = { ', file=fin)
    print('         SupercellFolding = 2 0 0, 0 2 0, 0 0 2, 0.5 0.5 0.5', file=fin)
    print('            }', file=fin)
    print('    MaxAngularMomentum = { ', file=fin)
    for key in maxam:
        print('    %s =' %key,maxam[key], file=fin)
    print('            }', file=fin)
    print('    MaxSCCIterations = %d' %maxscc, file=fin)
    print('    SCC = Yes', file=fin)
    print('    SlaterKosterFiles = Type2FileNames{ ', file=fin)
    print('        Prefix = %s'   %skf_dir, file=fin)
    print('        Separator = "-" ', file=fin)
    print('        Suffix = ".skf" ', file=fin)
    print('             }', file=fin)
    print('    Mixer = Broyden {', file=fin)
    print('            MixingParameter = 0.2', file=fin)
    print('            InverseJacobiWeight = 0.01', file=fin)
    print('            MinimalWeight = 1', file=fin)
    print('            MaximalWeight = 100000', file=fin)
    print('            WeightFactor = 0.01', file=fin)
    print('             }', file=fin)
    print('    ReadInitialCharges  = %s' %readinitialcharges, file=fin)
    print('    Charge  = 0 ', file=fin)
    print('    Filling = Fermi {', file=fin)
    print('    Temperature = 0.0', file=fin)
    print('    IndependentKFilling = No', file=fin)
    print('             }', file=fin)
    if thirdorder:
       print('    ThirdOrderFull = Yes', file=fin)
       #print>>fin,'    DampXH = Yes'
       print('    HubbardDerivs {', file=fin)
       for key in Hubbard:
           print('    %s = %s' %(key,Hubbard[key]), file=fin)
       print('        }', file=fin)
    if dispersion == 'dftd3':
       print('    Dispersion = DftD3 {}', file=fin)
    else:
       print('    Dispersion = LennardJones {', file=fin)
       print('    Parameters = UFFParameters {}', file=fin)
       print('             }', file=fin)
    print('    PolynomialRepulsive = {', file=fin)
    for pair in polynomial:
        p = polynomial[pair]
        if w_poly:
           write_polynomial(skf=pair,rcut=3.9,c=p,direc=skf_dir)
        print('     %s = Yes' %pair, file=fin)
    print('             }', file=fin)
    print('    SCCTolerance = 1.0e-7', file=fin)
    print('} ', file=fin)
    print(' ', file=fin)
    print(' ', file=fin)
    print('Options { ', file=fin)
    print('   WriteResultsTag = Yes ', file=fin) 
    print('} ', file=fin)

    print('ParserOptions {', file=fin)
    print('   ParserVersion = 7', file=fin)
    print('}', file=fin)
    if analysis:
       print('Analysis = { ', file=fin)
       print('CalculateForces = Yes', file=fin)
       print('}', file=fin)
    fin.close()


def get_dftb_forces():
    ff = open('detailed.out','r')
    readforce = False
    havread = False
    forces = []
    for line in ff.readlines():
        if line.find('Total Forces')>=0:
           readforce = True
           havread = True
        if len(line.split())<2:
           readforce = False
        if readforce and len(line.split())==3:
           fx = float(line.split()[0])/0.194469064593167E-01
           fy = float(line.split()[1])/0.194469064593167E-01
           fz = float(line.split()[2])/0.194469064593167E-01
           forces.append(fx)
           forces.append(fy)
           forces.append(fz)
    ff.close()
    if not havread or len(forces)==0:
       print('* error: dftb forces not find!')
       exit()
    # print(len(forces))
    return forces


def get_dftb_energy(out='dftb.out'):
    fo = open(out,'r')
    lines = fo.readlines()
    fo.close()

    energy = []
    p,t    = [],[]
    for i,line in enumerate(lines):
        if line.find('Geometry step:')>=0:
           toread = True
           step   = int(line.split()[3])
           if step!=len(p)+1:
              continue
           i_ = i 
           while toread:
                 i_ += 1
                 line_ = lines[i_]
                 if line_.find('Total MD Energy:')>=0:
                    energy.append(float(line_.split()[-2]))  # energy in unit eV
                 elif line_.find('MD Temperature:')>=0:
                    t.append(float(line_.split()[-2]))       # temperature in unit K
                 elif line_.find('Pressure:')>=0:
                    p.append(float(line_.split()[-2]))       # pressure in unit Pa
                    toread = False
                 elif line_.find('SCC is NOT converged')>=0:
                   energy.append(0.0)
                   t.append(0.0)
                   p.append(0.0)
                   toread = False
    return np.array(energy),np.array(p),np.array(t)
     

def run_dftb(cmd='dftb+>dftb.out'):
    system(cmd)


def reaxyz(fxyz):
    # cell = get_lattice()
    f = open(fxyz,'r')
    lines = f.readlines()
    f.close()
    
    unit = 1.0/98.2269478846
    natom  = int(lines[0])
    nframe = int(len(lines)/(natom+2))
    
    positions  = []
    velocities = []
    atom_name  = []
    energies   = []
    frames     = []

    ln = 0
    nf = 0
    while ln<(len(lines)-1):
        le    = lines[nf*(natom+2)+1].split()
        frame = int(le[2])
        if frame not in frames:
           frames.append(frame)
           pos_ = []
           vel_ = []
           for na in range(natom):
               ln = nf*(natom+2)+2+na
               l  = lines[ln].split()
              
               if nf==0:
                  atom_name.append(l[0])

               pos_.append([float(l[1]),float(l[2]),float(l[3])])
               vel_.append([float(l[4])*unit,float(l[5])*unit,float(l[6])*unit])

           positions.append(pos_)
           velocities.append(vel_)
           # print('- frame:',nf)
        ln = nf*(natom+2)+2+natom
        nf += 1
    return atom_name,np.array(positions),np.array(velocities),frames


def xyztotraj(fxyz,gen='poscar.gen',mode='w'):
    A = read(gen)
    atom_name,positions,velocities,frames = reaxyz(fxyz)
    e,p,t = get_dftb_energy(out='dftb.out')
    cell = A.get_cell()
    # box = [cell[0][0],cell[1][1],cell[2][2]]
    his  = TrajectoryWriter('dftb.traj',mode=mode)

    for i,p_ in enumerate(positions):
        # pos = np.mod(positions[i],box) # aplling simple pbc conditions
        A = Atoms(atom_name,positions[i],cell=cell,pbc=[True,True,True])
        # print(i,len(e),len(positions),e[frames[i]])
        calc = SinglePointCalculator(A,energy=e[frames[i]])
        A.set_calculator(calc)
        A.set_velocities(velocities[i])
        his.write(atoms=A)
        del A
    his.close()
    return e[frames],p[frames],t[frames]


class DFTB(object):
  def __init__(self,pressure=None,dT=5.0,ncpu=1,gen='poscar.gen',
               skf_dir='./',
               maxam={'C':'p','H':'s','O':'p','N':'p'},
               hubbard={'C':-0.1492,'H': -0.1857,'O':-0.1575,'N': -0.1535},
               maxscc=1000):
      '''
      Hugoniot equation of state.
      pressure = [0.001,2,4,6,8,10,12,14,16,18,20,22]
      '''
      # self.pressure  = np.linspace(0.0001, 20, 20)
      self.pressure = pressure
      self.np       = ncpu
      self.dT       = dT
      self.gen      = gen
      self.skf_dir  = skf_dir
      self.maxam    = maxam
      self.hubbard  = hubbard
      self.maxscc   = maxscc


  def nvt(self,T=2500,compress=1.0,mdRestart=10,
          velocities=None,
          initCharge='No'):
      A = read(self.gen,index=-1)
      cell = A.get_cell()
      cell = cell*compress
      A.set_cell(cell)
      A.write(self.gen)

      write_dftb_in(coordinate=self.gen,
                    runtype = 'md',   # canbe energy, opt, md ...
                    ensemble='nvt',
                    step=2000,dt=0.1,T=T,timescale=500,
                    AdaptFillingTemp='No',P=0.0,
                    latopt='no',
                    label='dftb',
                    velocities=velocities,
                    restart=mdRestart,
                    maxf=1.0e-5,
                    maxam=self.maxam,
                    Hubbard=self.hubbard,
                    maxscc=self.maxscc,
                    readinitialcharges=initCharge,
                    polynomial={},
                    dispersion=None, # dftd3
                    thirdorder='True',  # must
                    analysis=True,
                    skf_dir=self.skf_dir,
                    w_poly=False)
      self.run_dftb()
      e,p,t = xyztotraj('dftb.xyz')
      # send_msg('-  Dftb+ task completed.')
      # return e,p,t


  def get_traj(self):
        e,p,t = xyztotraj('dftb.xyz')
        return e,p,t


  def run_dftb(self,gen='poscar.gen'):
      if self.np==1:
         system('dftb+>dftb.out')
      else:
         system('mpirun -n %d dftb+>dftb.out' %self.np)


  def get_thermal(self,out='dftb.out'):
      e,p,t = get_dftb_energy(out=out)
      return e,p,t
      

  def close(self):
      print('-  Hugstate calculation compeleted.')
      


if __name__ == '__main__':
   if isfile('dftb.traj'):
      A = read('dftb.traj',index=-1)
      write_dftb_velocities(A,'velocities')
      A.write('poscar.gen')
      v='velocities'
   else:
      v = None

   ''' evergy (mdRestart) to write MD trajectories i.e. positions and velocities 
       when task completed, can use " ase gui dftb.traj to see the MD results"
   '''
   dftb = DFTB(pressure=None,dT=5.0,ncpu=40,
               skf_dir='/home/leno/scisoft/3ob-3-1/')
   dftb.nvt(T=2500,compress=0.996,mdRestart=10,velocities=v)

 
