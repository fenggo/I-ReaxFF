from __future__ import print_function
from os import system, getcwd, chdir,listdir
from os.path import isfile
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms


def mass(element):
    mas = {'C':12.000,'H':1.008,'N':14.000,'O':15.999,'Fe':55.845}
    if element in mas:
       return mas[element]
    else:
       return 0.0


def get_structure(struc=' ',output='dftb',recover=False,center=False,basis='6-311g',
                  supercell=[1,1,1]):
    fin = open('in.s','w')
    print('%'+'structure %s' %struc, file=fin)
    if output == 'nwchem':
       print('%'+'output %s %s' %(output,basis), file=fin)
    else:
       print('%'+'output %s' %output, file=fin)
    #print>>fin,'%'+'cellpara 1'
    #print>>fin,'10.0 10.0 10.0 90.0 90.0 90.0'
    if center :  print('%'+'center  .true.', file=fin) 
    if recover:  print('%'+'recover  .true.', file=fin)
    print('%'+'supercell  ',supercell[0],supercell[1],supercell[2], file=fin)
    fin.close()
    system('emdk<in.s>log')
    struc_xyz = struc+'.xyz'
    if isfile('card.xyz'):
       system('mv card.xyz %s' %struc_xyz)
    if isfile('card.cif'):
       system('mv card.cif %s' %struc+'.cif')
    if isfile('card.cfg'):
       system('mv card.cfg %s' %struc+'.cfg')


def emdk(cardtype='xyz',cardfile='card.xyz',
         cell=[[10.0,0.0,0.0],[0.0,10.0,0.0],[0.0,0.0,10.0]],
         output='species',basis='6-311g',
         supercell=[1,1,1],
         center='.False.',
         element='4 C H O N',
         masses='12.000 1.008 15.999 14.0',
         log='log'):
    ''' '''
    if cardtype == 'xyz' or cardtype == 'gen':
       fc = open(cardfile,'r')
       line = fc.readlines()[0]
       natm = line.split()[0]
       fc.close()

    fem=open('in.geo','w')
    print('%'+'coord','  %s' %cardtype, file=fem)
    print('%'+'file   %s' %cardfile, file=fem)
    print('%'+'cellpart',natm, file=fem)
    print('%'+'element %s' %element, file=fem)
    print('%'+'masses  %s' %masses, file=fem)
    if output == 'nwchem':
       print('%'+'output %s %s' %(output,basis), file=fem)
    else:
       print('%'+'output %s' %output, file=fem)
    print('%'+'cellvector ', file=fem)
    print('%f %f %f' %(cell[0][0],cell[0][1],cell[0][2]), file=fem)
    print('%f %f %f' %(cell[1][0],cell[1][1],cell[1][2]), file=fem)
    print('%f %f %f' %(cell[2][0],cell[2][1],cell[2][2]), file=fem)
    print('%'+'center %s ' %center, file=fem)
    print('%'+'supercell ',supercell[0],supercell[1],supercell[2], file=fem)
    fem.close()
    system('emdk<in.geo>%s' %log)


def findmole_gulp():
    system('cat his.xyz | tail -%s>>struc.xyz' %str(200+2))

    fout = open('gulp.out','r')
    lines = fout.readlines()
    for i,line in enumerate(lines):
        if line.find('Final Cartesian lattice vectors')>=0:
           iline = i
           break 
    fout.close()


    v = []
    pbc = open('pbc.txt','w')
    for l in range(3):
        line = lines[iline+l+2]
        print(line[:-1], file=pbc)
    pbc.close()

    findmole(filename='struc.xyz',trjtype=2,
                 frame=10000000,timeinterval=5.0,runtype=2,
                 order = True)


def MathineLearning(X,Y,pkl):
    if isfile(pkl):
       ml = joblib.load(pkl)
    else:
       ml = GaussianProcessRegressor()
       ml.fit(X,Y)
       joblib.dump(ml, pkl)
       print(('* training completed, and save in file %s.' %pkl))
    return ml


def gulp_out_to_input(natm=200,scale= 1.05):
    system('cat his.xyz | tail -%s>>card.xyz' %str(natm+2))

    # get lattice vector
    fout = open('gulp.out','r')
    lines = fout.readlines()
    for i,line in enumerate(lines):
        if line.find('Final Cartesian lattice vectors')>=0:
           iline = i
           break 
    fout.close()
 
    v = []
    for l in range(3):
        line = lines[iline+l+2]
        #print line
        v.append([float(line.split()[0])*scale,float(line.split()[1])*scale,
                float(line.split()[2])*scale])


    system('cp card.xyz card.xyz.sample')

    fs = open('card.xyz.sample','r')
    fc = open('card.xyz','w')

    for line in fs.readlines():
        ls = line.split()
        if len(ls)==4:
           print(ls[0],float(ls[1])*scale,float(ls[2])*scale,float(ls[3])*scale, file=fc)
        else:
           print(line[:-1], file=fc)

    fs.close()
    fc.close()

    emdk(cardtype='xyz',cardfile='card.xyz',
         vec={'a':v[0],'b':v[1],'c':v[2]},
         output='gulp opt',center='.False.',log='log')


def xyz_to_gulp(xyzname,runtype='gradient',output='gulp opt',
                         delxyz=True,check = False,center=True,lib='bop'):
    emdk(cardtype='xyz',cardfile=xyzname,output=output,
         vec={'a':[10.0,0.0,0.0],'b':[0.0,10.0,0.0],'c':[0.0,0.0,10.0]},
         center=center,log='log')
    if runtype == 'gradient':
       fin  = open('gin','w')
       fgin = open('inp-gulp', 'r')
       for line in fgin.readlines():                # prepare input file
           if len(line.split())>=1:
              if line.split()[0] == 'opti':
                 print('gradient nosymmetry conv', file=fin)
              elif line.split()[0] == 'library':
                 print('library bop_SA', file=fin) 
              else:
                 print('%s' %line[:-1], file=fin)       # delete /n
       fgin.close()
       fin.close()
    elif runtype == 'md':
       fin  = open('gin','w')
       fgin = open('inp-gulp', 'r')
       for line in fgin.readlines():                # prepare input file
           if len(line.split())>=1:
              if line.split()[0] == 'library':
                 print('library %s' %lib, file=fin) 
              elif line.split()[0] == 'write':
                 print('write        1', file=fin) 
              elif line.split()[0] == 'sample':
                 print('sample        1', file=fin) 
              else:
                 print('%s' %line[:-1], file=fin) # delete /n

       fgin.close()
       fin.close()


def out_to_gulp(out=None):
    X = []
    if out != None:
       fout = open(out,'r')
       lines = fout.readlines()
       fout.close()

       outname = 'inp-gulp'
       fgulp = open(outname,'w')
       
       print('gradient nosymmetry conv', file=fgulp)
       print('title', file=fgulp)
       print('GULP calculation', file=fgulp)
       print('end', file=fgulp)
       print('library bop', file=fgulp)    
       print('output movie xyz his.xyz', file=fgulp)
       print('dump 50 restart.grs', file=fgulp)
       print('vectors', file=fgulp)
       print('10.0 0.0 0.0', file=fgulp)
       print('0.0 10.0 0.0', file=fgulp)
       print('0.0 0.0 10.0', file=fgulp)
       print(' ', file=fgulp)
       print('cartesian', file=fgulp)

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
           print(lie.split()[1],' core ',lie.split()[3],lie.split()[4],lie.split()[5], \
                    0.0,1.0,0.0,1,1,1, file=fgulp)
       fgulp.close()
    


def findmole(filename='lammps.trj',trjtype=1,
             frame=10000000,timeinterval=0.005,runtype=2,
             order = True):
    ffm = open('in.fm','w')
    print(filename, file=ffm)
    print(trjtype,' # 1 lammpstrj, 2 xyz', file=ffm)
    print(timeinterval,' # time interval in trajectory', file=ffm)
    print(runtype,' # 1 all frames 2 only one frames', file=ffm)
    print(frame,' # for last frame only', file=ffm)
    ffm.close()
    system('findmole<in.fm>fm.out')

    mol = None
    if order:
      fm = open('molecular_structure.txt')
      for line in fm.readlines():
        molname = []
        mol  = {}
        molecules = {}
        if line.find('Molecular structure -vs- number')>=0:
           molname = []
           mol  = {}
        elif line.find('Time(ps)')>=0:
           t = line.split()[1]
        elif len(line.split()) ==2 and line.find('Time') <0:
           mol[line.split()[0]] = int(line.split()[1])
           molecules[line.split()[0]] = int(line.split()[1])
      if len(molecules)>= 5:
         lran = 5
      else:
         lran = len(molecules)

      for i in range(0,lran):
          first = 0
          #print i
          for key in mol:
              if mol[key]>first:
                 first = mol[key]
                 firstname = key
          molname.append(firstname)
          del mol[firstname]
      fm.close()
      molnum = []
      for m in molname:
          molnum.append(molecules[m]) 
    return molname,molnum


def get_lattice(inp='inp-nve'):
    if isfile('inp-nve'):
       inp = 'inp-nve'
    elif isfile('inp-nvt'):
       inp = 'inp-nvt'
    else:
       print('-  error: input file of cpmd not found!')
    finp = open(inp,'r')
    il = 0
    cell = []
    readlatv = False
    readlat  = False
    for line in finp.readlines():
        l = line.split()
        if line.find('CELL')>=0 and line.find('VECTORS')>=0 and line.find('RESTART')<0:
           readlatv = True
        elif line.find('CELL')>=0 and line.find('VECTORS')<0 and line.find('RESTART')<0:
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


def gofr(ele1,ele2,traj='trajectory.xyz'):
    elem={'C':1,'H':2,'O':3,'N':4}
    cell = get_lattice()

    fpbc=open('pbc.txt','w')
    print(cell[0][0],cell[0][1],cell[0][2],file=fpbc)
    print(cell[1][0],cell[1][1],cell[1][2],file=fpbc)
    print(cell[2][0],cell[2][1],cell[2][2],file=fpbc)
    fpbc.close()

    fin = open('in.gofr','w')
    print(traj,file=fin)
    print(2,file=fin)
    print(elem[ele1],elem[ele2],file=fin)
    print(0,file=fin)
    fin.close()
    system('gofr<in.gofr')

    # plot gofr 
    fg = open('gofr_'+ele1+ele2+'.txt','r')
    x,y=[],[]
    for row in fg.readlines():
        if row.find('#')==0:
           continue
        r = row.split()
        x.append(float(r[0]))
        y.append(float(r[1]))
    fg.close()

    plt.figure()
    plt.ylabel(r'$Pair$ $Distribution$ $Function$')
    plt.xlabel(r'$Radius$ $Unit:$ $\AA$')
    plt.xticks(np.arange(0,10.0,0.5))
    ax = plt.gca()
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(-45)
    plt.plot(x,y,label=r'%s-%s' %(ele1,ele2), color='black', 
             linewidth=2, linestyle='-')
    plt.legend()
    plt.savefig('gofr_'+ele1+ele2+'.eps') 

       
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


def get_gulp_gradient(out='gulp.log'):
    fg = open(out,'r')
    energy = None
    gradient = []
    lr = False
    kk = 0
    for line in fg.readlines():
        if line.find('Final internal derivatives :') >= 0:
           lr = True
           kk = 0
           #print 'find it'
        elif line.find('Total lattice energy') >= 0 and line.find('eV') > 0:
           energy = float(line.split()[4])
        elif line.find('Maximum abs') >= 0:
           lr = False
           exit

        lrr = False
        if lr :
           if line.find('--------------------------------------------------------------------------------')>=0:
              kk += 1
        if kk == 2:
           if len(line.split())==7:
              gx = line.split()[3]
              gy = line.split()[4]
              gz = line.split()[5]
              if gx == 'NaN': gx = '999.0'
              if gy == 'NaN': gy = '999.0'
              if gz == 'NaN': gz = '999.0'
              if gx.find('*')>=0: gx = '999.0'
              if gy.find('*')>=0: gy = '999.0'
              if gz.find('*')>=0: gz = '999.0'
              gradient.append(float(gx))
              gradient.append(float(gy))
              gradient.append(float(gz))
           if len(line.split())<7 and line.find('---')<0:
              gz =999.0;gy=999.0;gx=999.0 
              gradient.append(float(gx))
              gradient.append(float(gy))
              gradient.append(float(gz))
    fg.close()
    #print gradient
    #gradient = np.array(gradient)
    return energy,gradient


def gulp_mdin(mol='cl20mol'):
    get_struc(struc=mol,output='gulp md-nvt')
    fin = open('inp-gulp','r')
    finp= open('gin','w')
    for line in fin.readlines():
        if line.find('ensemble')>=0:
           finp.write('ensemble nve \n')
        elif line.find('tau_thermostat')>=0:
           finp.write('tau_thermostat 0.02 ps \n')
        elif line.find('temperature')>=0:
           finp.write('temperature    30.0 K \n')
        elif line.find('timestep')>=0:
           finp.write('timestep       0.001 ps \n')
        elif line.find('production')>=0:
           finp.write('production     0.5 ps \n')
        elif line.find('equilibration')>=0:
           finp.write('equilibration  0.0    ps \n')
        elif line.find('write')>=0:
           finp.write('write          1 \n')
        elif line.find('sample')>=0:
           finp.write('sample         1 \n')
        elif line.find('library')>=0:
           finp.write('library        bop_SA \n')
        else:
           finp.write(line)
    fin.close(); finp.close()


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


def get_table(run_emdk=False,structure='cl20mol'):
    '''return molecular neighbor table'''
    if run_emdk:
       get_structure(struc=structure,output='dftb',recover=False,center=False,
                     supercell=[1,1,1])
       emdk(cardtype='gen',
            cardfile='card.gen',
            output  ='table',
            center  ='.False.')

    atoms_name,table = [],[]
    ft = open('atable.log','r')
    for line in ft.readlines():
        if line.split()[0]!='ATOM':
           atoms_name.append(line.split()[1])
           tab = []
           for i in range(0,len(line.split())):
               if i>1 and i%2==0:
                  tab.append(int(line.split()[i])-1)
           if tab:
              table.append(tab)
    ft.close()
    return atoms_name,table


def get_bond():
    f_b = open('bonds.txt','r')
    bond_name = []
    bond_length = []
    for b in f_b.readlines():
        bb = b.split()
        bond_name.append(bb[0][0]+bb[1][0])
        bond_length.append(float(bb[2]))
    f_b.close()
    return bond_name,bond_length


def write_xyz(natm,atomname,X):
    fxyz = open('card.xyz','w')
    print(natm, file=fxyz)
    print('write by emdk', file=fxyz)
    for i,x in enumerate(X):
        print(atomname[i],x[0],x[1],x[2], file=fxyz)
    fxyz.close()


def project_force(fo,vec):
    '''
     project the force vector to the bond direction vector,vec is unit vectior
    '''
    rdot = fo[0]*vec[0] + fo[1]*vec[1] + fo[2]*vec[2] 
    r1   = np.sqrt(fo[0]*fo[0] + fo[1]*fo[1] + fo[2]*fo[2] )
    r2   = 1 #np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2] )
    cos_fv =  rdot/r1*r2
    return r1*cos_fv


def plot_nw_energy(atom_pair=None):
    line = getoutput('ls *.out')
    #print line
    i = 0
    fe = open('e.txt','w')
    for out_file in line.split('\n'):
        print('interpretating %s ...' %out_file)
        el = getoutput('grep error %s' %out_file)
        if len(el) != 0:
           print('Skipping an error file ...')
           continue

        eline=getoutput('grep "Total DFT energy =" %s' %out_file)
        e_ab = float(eline.split()[-1])*27.211396  #au to ev
        if atom_pair==None:
           print(i,e_ab, file=fe)
        else:
           eline = getoutput('grep "Stretch                  %d     %d" %s' %(atom_pair[0],atom_pair[1],out_file))
           if len(eline)>0:
              r = eline.split()[-1]
              print(r,e_ab, file=fe)
        i += 1
    fe.close()
    gplot(eps_name='pes.eps',plotname='e.txt',axis=[(1,2)],
          xlab='Radius',ylab='energy (eV)',title=['NWchem/BLYP/6-311++'])


def nw_dimer(elm1='C',elm2='C',rmax=1.6,rmin=1.0,dr=0.02,task='gradient',np=8,basis='6-311'):
    r=rmin
    while r < rmax:
        fxyz = open('dimer.xyz','w')
        print(2, file=fxyz)
        print(' ', file=fxyz)
        print(elm1,0.0,0.0,0.0, file=fxyz)
        print(elm2,0.0,0.0,r, file=fxyz)
        fxyz.close()
        xyz_to_nw('dimer.xyz',delxyz=True,basis=basis)
        system('mv dimer.inp %s%s_%s.inp' %(elm1,elm2,str(r)))
        r += dr
    get_nw_data(task=task,np=np)
    system('rm *.inp')
    plot_nw_energy(atom_pair=[1,2])


def gplot(eps_name='pic.eps',plotname='txt',axis=[(1,2)],xlab=' ',ylab=' ',title=['notitle']):
    ############ gnuplot ############
    # for example
    #eps_name = 'e' + '.eps'  
    #plotname = 'e' + '.txt'
    fplot = plotname[:-4] + '.plt'
    fplt = open(fplot,'w')
    if eps_name.find('.eps') >= 0:
       print('set terminal post eps color enhanced solid linewidth 3', file=fplt)
    else:
       print('set terminal png truecolor enhanced size 1280,960 font arial 36 linewidth 10', file=fplt)
    print('set out "%s"' %eps_name, file=fplt)
    print('set  xlabel "%s"' %xlab, file=fplt)
    print('set  ylabel "%s"' %ylab, file=fplt)
    lena = len(axis)
    lent = len(title)
    ax = axis[0]
    if len(axis)>1:
       print('plot "%s"  using %d:%d with points ps 1.6 pt 6 title "%s" ' %(plotname,ax[0],ax[1],title[0]), file=fplt) #, \\' 
    else:
       print('plot "%s"  using %d:%d with points ps 1.6 pt 6 title "%s", \\' %(plotname,ax[0],ax[1],title[0]), file=fplt) #, \\' 
    for i in range(1,lena):
        if lent<i-1:
           tit = 'notitle'
        else:
           tit = title[i]
        if i==lena-1:
           print('      ""   using %d:%d with points ps 1.6 pt 6 title "%s" ' %(axis[i][0],axis[i][1],tit), file=fplt)
        else:
           print('      ""   using %d:%d with points ps 1.6 pt 6 title "%s" , \\' %(axis[i][0],axis[i][1],tit), file=fplt)
    fplt.close()
    system('gnuplot %s' %fplot)
  

def get_distances(ra,rb):
    Dvec = np.zeros(3)
    Dvec[0] = rb[0] - ra[0]
    Dvec[1] = rb[1] - ra[1]
    Dvec[2] = rb[2] - ra[2]

    r2 = Dvec[0]*Dvec[0] + Dvec[1]*Dvec[1] + Dvec[2]*Dvec[2]
    r = np.sqrt(r2)
    if r > 1.98:
       print(' \n')
       print('------------------------------------------------------------')
       print('         * connection is beyond ML can predict,')
       print('         * connections should be rebuild!')
       print('------------------------------------------------------------')
       print(' \n')
    return r 
    

def top2nwchem(ftop='joy.new.top',fg='amber_new.par'):
    ftop = open(ftop,'r')
    flib = open(fg,'w')

    lr = False
    latm = False
    lb = False
    la = False
    ld = False
    lnb= False

    atoms = []
    blib = {}; alib = {}; tlib = {}; nblib={};
    bln = {}; aln = {}; tln={}

    for line in ftop.readlines():
        if line.find('[ atoms ]')>=0:
           latm = True
        elif line.find('[ atomtypes ]')>=0:
           latm = False
           lnb = True
        elif line.find('[ bonds ]')>=0:
           latm = False
           lnb = False
           lb = True
        elif line.find('[ angles ]')>=0:
           latm = False
           lnb = False
           lb = False
           la = True
        elif line.find('[ dihedrals ]')>=0:
           latm = False
           lnb = False
           lb = False
           la = False
           ld = True
        if latm and len(line.split())==8:
           atoms.append(line.split()[1])
        if lb and len(line.split())==5: 
           a = atoms[int(line.split()[0])-1]
           b = atoms[int(line.split()[1])-1]
           if not a+'-'+b in blib:
              # k(kj/mol*nm**2 to eV/A**2) = 1.0364*0.0001
              blib[a+'-'+b] = [float(line.split()[4]),float(line.split()[3])]
              bln[a+'-'+b] = 1 
           else:
              blib[a+'-'+b][0] +=  float(line.split()[4])#*1.0364*0.0001
              blib[a+'-'+b][1] +=  float(line.split()[3])#*10.0
              bln[a+'-'+b] += 1 
        if la and len(line.split())==6: 
           a = atoms[int(line.split()[0])-1]
           b = atoms[int(line.split()[1])-1]
           c = atoms[int(line.split()[2])-1]
           if not a+'-'+b+'-'+c in alib:
              #k_unit convertion 1.0364*0.01   ####  /degree**2 to /rad**2
              #alib[a+'-'+b+'-'+c] = [float(line.split()[5])*1.0364*0.01*(3.1415926/180.0)**2, \
              #                   float(line.split()[4])*(3.1415926/180.0)]

              alib[a+'-'+b+'-'+c] = [float(line.split()[5]), \
                                 float(line.split()[4])]
              aln[a+'-'+b+'-'+c]  = 1
           else:
              alib[a+'-'+b+'-'+c][0] += float(line.split()[5])#*1.0364*0.01*(3.1415926/180.0)**2
              alib[a+'-'+b+'-'+c][1] += float(line.split()[4])#*(3.1415926/180.0)
              aln[a+'-'+b+'-'+c] += 1
        if ld and len(line.split())==7: 
           if line.split()[0] == ';':
              continue
           a = atoms[int(line.split()[0])-1]
           b = atoms[int(line.split()[1])-1]
           c = atoms[int(line.split()[2])-1]
           d = atoms[int(line.split()[3])-1]
           if not a+'-'+b+'-'+c+'-'+d in tlib:
              #tlib[a+'-'+b+'-'+c+'-'+d] = [ float(line.split()[6])*1.0364*0.01*(3.1415926/180.0)**2, \
              #                      float(line.split()[4])*(3.1415926/180.0),float(line.split()[5])]
              tlib[a+'-'+b+'-'+c+'-'+d] = [ float(line.split()[6]), \
                                    float(line.split()[4]),float(line.split()[5])]
              tln[a+'-'+b+'-'+c+'-'+d]  = 1
           else:
              #tlib[a+'-'+b+'-'+c+'-'+d][0] += float(line.split()[6])*1.0364*0.01*(3.1415926/180.0)**2
              tlib[a+'-'+b+'-'+c+'-'+d][0] += float(line.split()[6])
              tlib[a+'-'+b+'-'+c+'-'+d][1] += float(line.split()[4]) #*(3.1415926/180.0)
              tlib[a+'-'+b+'-'+c+'-'+d][2] += float(line.split()[5])
              tln[a+'-'+b+'-'+c+'-'+d] += 1
        if lnb and len(line.split())==6: 
           if line.split()[0] == ';':
              continue
           a = line.split()[0]
           if not a in nblib:
              #nblib[a]  = [float(line.split()[5])*1.0364*0.0001,float(line.split()[4])]
              nblib[a]  = [float(line.split()[5]),float(line.split()[4])]

    print('AMBER 99  parameter extensions: SPC/E water, Quantum OH groups, Solvents', file=flib)
    print('Electrostatic 1-4 scaling factor     0.833333', file=flib)
    print('Relative dielectric constant     1.000000', file=flib)
    print('Parameters epsilon R*', file=flib)
    print(' ', file=flib) 
    print('Atoms', file=flib)
    print('Cross', file=flib)
    print('Bonds', file=flib)

    for a in blib:
        k2=blib[a][0]/bln[a]
        k1=blib[a][1]/bln[a]
        print(a.split('-')[0],'-'+a.split('-')[1], k1,k2, file=flib)

    print('Angles', file=flib) 
    for a in alib:
        k1=alib[a][0]/aln[a]
        k2=alib[a][1]/aln[a]
        print(a.split('-')[0],'-'+a.split('-')[1],'-'+a.split('-')[2], \
               k1,k2, file=flib)
    print('Proper dihedrals', file=flib) 
    print('Improper dihedrals', file=flib) 
    print('End', file=flib) 
    flib.close()


def stretch(xyz='config.xyz',xyz_type='xyz',inptyp='nwchem',species=['ALL','ALL'],comp='nocomp',ns=1,dr=0.1):
    fin = open('in.s','w')
    fxyz= open(xyz,'r')
    line = fxyz.readlines()[0]
    natm = line.split()[0]
    print('%'+'coord  %s' %xyz_type, file=fin)
    print('%'+'file   %s' %xyz, file=fin)
    print('%'+'cellpart',natm, file=fin)
    print('%'+'output','gulp opt', file=fin)
    print('%'+'element  4 C H O  N ', file=fin)
    print('%'+'masses   12.000 1.008 15.999 14.0', file=fin)
    print('%'+'cellpara 1', file=fin)
    print('10.0 10.0 10.0 90.0 90.0 90.0', file=fin)
    print('%'+'center  .true.', file=fin)
    print('%'+'stretch %s %s free' %(xyz[:-4],comp), file=fin)
    print(inptyp, species[0],species[1],'6-311g', file=fin)
    print('dr %f elongation %d shorten %d' %(dr,ns,ns), file=fin)
    print('fix-end 1  move-end 3 to-be-moved 2 atoms 5 4 # this line not used', file=fin) 
    fin.close()
    fxyz.close()
    system('emdk<in.s>stretch.log')


