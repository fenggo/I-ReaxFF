#!/usr/bin/env python
from __future__ import print_function
from os import system
from ase.io import read,write
from ase import Atoms
import numpy as np
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
# from irff.molecule import get_mol


       
def write_nwchem_inp(A,struc='MOL',task='dft qmd', 
                     T=300,step=2000,timestep=1.0,
                     com_step=10,
                     thermo='berendsen',thermo_num=100,
                     xc='b3lyp',mult=None,
                     basis='6-311+G*',
                     elec={'C':6,'H':1,'O':8,'N':7,'Al':13,'Fe':26}):
    ''' [b3lyp     ],[beckehandh],[HFexch    ],[becke88   ],[lyp       ],
        [slater    ],[vwn_1_rpa ],[hcth      ],[becke97   ],[cpbe96    ],
        [gill96    ],[xperdew91 ],[xbecke97  ],[cbecke97  ],[hcth120   ],
        [pbe0      ],[becke97gga],[hcth407   ],[optx      ],[cft97     ],
        [hcth407p  ],[xtpss03   ],[becke97-3 ],[pbeop     ],[rpbe      ],
        [revpbe    ],[bc95      ],[mpw1b95   ],[pwb6k     ],[m05       ],
        [m05-2x    ],[cpw6b95   ],[cvs98     ],[m06-L     ],[xm06-hf   ],
        [m06       ],[cm08-hx   ],[sogga11   ],[xcampbe96 ],[mp2       ],
        [b2plyp    ],[xcamlsd   ],[bhlyp     ],[s12g      ],[mvs       ],
        [b3p86     ],[pbe96     ],[dldf      ],[hse03     ],[becke86b  ],
        [hle16     ] '''
    atoms = A.get_chemical_symbols()
    if mult==None:
       mult  = 0
       for a in atoms:
           mult += elec[a]
       mult = mult%2
       mult = 1 if mult==0 else 2
          

    timestep = timestep/0.02419 # fs to a.u.

    fin = open('inp-nw','w')
    print('start QMD-%s' %struc,file=fin)
    print('echo\n',file=fin)
    # print('print low\n',file=fin)
    print('geometry noautosym',file=fin) #  autosym will cause an error # autoz
    for a in A:
        print('  ',a.symbol,a.x,a.y,a.z,file=fin)
    print('end\n',file=fin)
    print('basis',file=fin)
    print('  * library %s' %basis,file=fin)
    print('end\n',file=fin)
    print('dft',file=fin)
    print('  mult %s' %mult,file=fin)
    print('  xc %s' %xc,file=fin)
    print('  convergence energy 1e-07',file=fin)
    print('  maxiter 200',file=fin)
    print('end\n',file=fin)

    if task.find('qmd')>=0:
       print('qmd' ,file=fin)
       print('  nstep_nucl  %d' %step,file=fin)     # Simulation steps
       print('  dt_nucl     %d' %timestep,file=fin) # 1 a.u. = 0.02419 fs
       print('  targ_temp   %d' %T,file=fin)        # targ_temp
       print('  rand_seed   12346',file=fin)
       print('  com_step    %d' %com_step,file=fin) # How often center-of-mass 
       print('  thermostat  %s %8.3f' %(thermo,thermo_num),file=fin) 
       print('  print_xyz   1 ' ,file=fin)   
       print('end\n',file=fin)

    print('task %s\n' %task,file=fin)
    fin.close()


def get_nw_gradient(out='nw.out'):        
    fout = open(out,'r')
    lines = fout.readlines()
    fout.close()
    gfind = False
    eng = 0.0
    for iline, lin in enumerate(lines):
        if lin.find('Output coordinates in angstroms') >= 0:
           nline = iline
        if lin.find('Total DFT energy') >= 0:
           eng = float(lin.split()[4]) * 27.211396 ### a.u. to eV
        if lin.find('DFT ENERGY GRADIENTS') >= 0:
           gline = iline
           gfind = True
           #print >>ff, 'gradients eV/Angs'
        if lin.find('XYZ format geometry') >= 0:
           nalin = iline
        if lin.find('error')>=0 or lin.find('close but too far?')>=0:
           #raise AssertionError('error is found in output!')
           print('* error is found in output, negelected!') 
           return None,None

    lll = lines[nalin+2]
    natm = int(lll.split()[0])

    kkk = 1
    gradient = []
    if not gfind:
       print('* error: gradients not find in file: %s' %out)
    if eng == 0.0:
       print('* error: Energy not find in file: %s' %out)
    for k in range(0,natm):
        lie = lines[gline+4+k]
        fx = float(lie.split()[5])*27.211396/0.529   # eV/AA
        fy = float(lie.split()[6])*27.211396/0.529
        fz = float(lie.split()[7])*27.211396/0.529
        gradient.append(fx)
        gradient.append(fy)
        gradient.append(fz)
    return eng,gradient


def out_xyz(out_file=None):
    X = []
    atoms = []
    if out_file != None:

       fout = open(out_file,'r')
       lines = fout.readlines()
       fout.close()

       for iline, lin in enumerate(lines):
           if lin.find('Output coordinates in angstroms') >= 0:
              nline = iline
           elif lin.find('XYZ format geometry') >= 0:
              line1 = lines[iline+2]
              natm = int(line1.split()[0])
           elif lin.find('error') >= 0:
              print('* An error file !')
              exit()

       for k in range(0,natm):
           lie = lines[nline+4+k]
           X.append([float(lie.split()[3]),float(lie.split()[4]),float(lie.split()[5])])
           atoms.append(lie.split()[1])
    return natm,atoms,X


def read_xyz(file=None):
    X = []
    atoms = []
    if file != None:

       fout = open(file,'r')
       lines = fout.readlines()
       fout.close()

       natm = int(lines[0])

       for k in range(0,natm):
           lie = lines[2+k]
           X.append([float(lie.split()[1]),float(lie.split()[2]),float(lie.split()[3])])
           atoms.append(lie.split()[0])
    return natm,atoms,X


def out_to_xyz(out_file=None):
    X = []
    if out_file != None:

       fout = open(out_file,'r')
       lines = fout.readlines()
       fout.close()

       for iline, lin in enumerate(lines):
           if lin.find('Output coordinates in angstroms') >= 0:
              nline = iline
           elif lin.find('XYZ format geometry') >= 0:
              line1 = lines[iline+2]
              natm = int(line1.split()[0])
           elif lin.find('error') >= 0:
              print('* An error file !')
              exit()
              #break
       xyzname = out_file[:-4]+'.xyz'
       fxyz = open(xyzname,'w')
       print(natm, file=fxyz)
       print(xyzname[:-4], file=fxyz)
       for k in range(0,natm):
           lie = lines[nline+4+k]
           print(lie.split()[1],lie.split()[3],lie.split()[4],lie.split()[5], file=fxyz)
           X.append([float(lie.split()[3]),float(lie.split()[4]),float(lie.split()[5])])
       fxyz.close()
    return natm,X

    
def xyz_to_nw(xyzname,delxyz=True,center = True, basis='6-311G'):
    fxyz = open(xyzname,'r')
    line = fxyz.readlines()[0]
    if len(line.split())==1:
       natm = line.split()[0]

    fxyz.close()
    fem=open('in.geo','w')
    print('%'+'coord','  xyz', file=fem)
    print('%'+'file   %s' %xyzname, file=fem)
    print('%'+'cellpart',natm, file=fem)
    print('%'+'element','4 C H O N', file=fem)
    print('%'+'output nwchem %s' %basis, file=fem)
    print('%'+'cellpara 1', file=fem)
    print('10.0 10.0 10.0 90.0 90.0 90.0', file=fem)
    if center: print('%'+'center  .true.', file=fem)
    fem.close()
    system('emdk<in.geo>log')
    inpname = xyzname[:-4] + '.inp'
    system('mv inp-nw %s' %inpname)
    #fin = open(inpname,'a')
    #print>>fin,'task dft gradient'
    #fin.close()
    if delxyz:
       system('rm %s' %xyzname) 


def get_nw_data(task='gradient',np=4):
    line = getoutput('ls *.inp' )
    for ss in line.split('\n'):
        fin = open(ss,'a')
        #print>>fin,'geometry adjust'
        #print>>fin,'  zcoord'
        #print>>fin,'     bond  1 4   %f nn constant' %v_r
        #print>>fin,'  end'
        #print>>fin,'end'
        print('task dft %s' %task, file=fin)
        fin.close()
        out_name = ss[:-4]+'.out'
        print('running for %s ...' %ss)
        system('mpirun -n %d nwchem %s > %s' %(np,ss,out_name))
        sss = ['*.0','*.1','*.2','*.3','*.4','*.5','*.6','*.7','*.movecs',
               '*.gridpts.*','*.aoints.*',
               '*.b','*.db','*.b^-1','*.d','*.c','*.p','*.zmat','*.hess']
        for s in sss:
            lls=getoutput('ls %s' %s)
            if len(lls.split())==1:
               system('rm %s' %s) 


def decompostion(gen='rdx.gen',ncpu=12):
    ''' decompostion energy of N-NO2 '''
    traj = TrajectoryWriter('stretch.traj',mode='w')
    ncfg = 20
    A = read(gen)
    s = stretch(A,shrink=0.8,enlarge=2.6,
                fix_end=3,move_end=18,move_atoms=[18,19,20],nconfig=ncfg)
    for i in range(ncfg):
        atoms = s.move(i)
        write_nwchem_inp(atoms,struc=gen.split('.')[0],task='dft gradient', 
                         xc='b3lyp',basis='6-311G*')

        system('mpirun -n %d nwchem inp-nw > nw.out' %(ncpu))
        e,grad_ = get_nw_gradient(out='nw.out')
        system('cp nw.out nw_%s.out' %str(i))
        system('rm QMD-*') 
        print('-  energy(NWchem):',e)
        calc = SinglePointCalculator(atoms,energy=e)
        atoms.set_calculator(calc)
        traj.write(atoms=atoms)
    traj.close()


class stretch(object):
  def __init__(self,A,fix_end=None,move_end=None,move_atoms=[],
               enlarge=1.8,shrink=0.8,nconfig=10):
    vr  = A.positions[move_end] - A.positions[fix_end]
    vr2 = np.square(vr)
    self.r = np.sqrt(np.sum(vr2))
    self.v = vr/self.r
    cfg_ = np.arange(shrink,enlarge,(enlarge-shrink)/nconfig)
    
    self.moves = []
    for cfg in cfg_:
        self.moves.append((cfg-1.0)*self.r*self.v)

    self.move_atoms = np.array(move_atoms)
    self.A = A 

  def move(self,step):
      A_moved = self.A.copy()
      A_moved.positions[self.move_atoms,:] += self.moves[step]
      return A_moved


if __name__ == '__main__':
   decompostion()

     



