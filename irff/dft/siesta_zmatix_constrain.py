from __future__ import print_function
from os import system,getcwd,chdir 
from os.path import isfile,exists
from ..molecule import enlarge,molecules,get_neighbors
from .mdtodata import MDtoData
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Ry
from ase import Atoms
# from .dingtalk import send_msg
import numpy as np
'''  tools used by siesta '''


def siesta_traj(batch=10000):
    cwd = getcwd()
    d = MDtoData(structure='siesta',dft='siesta',direc=cwd,batch=batch)
    images = d.get_traj()
    d.close()
    return images


def xv2xyz():
    fxv = open('siesta.XV','r')
    fxyz= open('siesta.xyz','w')

    elements = {'6':'C','1':'H','7':'N','8':'O','26':'Fe'}

    for line in fxv.readlines():
        if len(line.split()) == 1:
            print>>fxyz,line.split()[0]
            print>>fxyz,'xyz file'
        elif len(line.split()) == 8:
            x = float(line.split()[2])*0.53
            y = float(line.split()[3])*0.53
            z = float(line.split()[4])*0.53
            print(elements[line.split()[1]],x,y,z,file=fxyz)


def SinglePointEnergies(traj,label='aimd',frame=50,cpu=4):
    his = TrajectoryWriter(label+'.traj',mode='w')
    images = Trajectory(traj)
    tframe = len(images)
    E = []
    if tframe>frame:
       ind_   = np.linspace(0,tframe,num=frame,dtype=np.int32)

    for i in range(tframe):
        if tframe>frame:
           if i in ind_:
              atoms = images[i]
           else:
              continue
        else:
           atoms = images[i]

        e_ = atoms.get_potential_energy()
        atoms_ = single_point(atoms,cpu=cpu)
        e  = atoms_.get_potential_energy()
        his.write(atoms=atoms_)
        print('- Energies from MLP: %9.5f DFT: %9.5f' %(e_,e))
        E.append(e)
    his.close()
    images = None
    return E


def single_point(atoms,cpu=4):
    if isfile('siesta.out'):
       system('rm siesta.*')

    write_siesta_in(atoms,coord='cart',
                    md=True,opt='SinglePoint',
                    xcf='VDW',xca='DRSLL',
                    us=False) # for MD
    system('mpirun -n %d siesta<in.fdf>siesta.out' %cpu)
    with open('siesta.out','r') as fs:
         for line in fs.readlines():
             if line.startswith('siesta: E_KS(eV)'):
                e = float(line.split()[-1])
                break

    calc = SinglePointCalculator(atoms,energy=e)
    atoms.set_calculator(calc)
    return atoms


def siesta_opt(atoms,ncpu=4,supercell=[1,1,1],
               VariableCell='true',
               dt=0.5,T=None,tstep=5000,
               us='F',P=0.0,
               coord='cart'):
    write_siesta_in(atoms,coord=coord,
                    VariableCell=VariableCell,
                    md=False,opt='CG',P=P,
                    xcf='VDW',xca='DRSLL',
                    us=us) # for MD
    run_siesta(ncpu=ncpu)
    images = siesta_traj()
    return images


def siesta_md(atoms=None,gen='poscar.gen',ncpu=4,supercell=[1,1,1],
              dt=0.5,P=0.0,T=None,tstep=5000,opt='Verlet',
              us='F',FreeAtoms=None):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    if atoms is None:
       atoms = read(gen)
    write_siesta_in(atoms,coord='cart',
                    md=True,opt=opt,
                    xcf='VDW',xca='DRSLL',
                    dt=dt,T=T,P=P,tstep=tstep,
                    us=us,FreeAtoms=FreeAtoms) # for MD
    run_siesta(ncpu=ncpu)
    images = siesta_traj(batch=tstep)
    return images


def run_siesta(ncpu=12):
    system('mpirun -n %d siesta<in.fdf>siesta.out' %ncpu)


def write_siesta_in(Atoms,coord='zmatrix',
                    VariableCell='false',
	                  fac=1.0,
	                  opt='CG',  # 'FIRE' 'SinglePoint'
                    md=False,
                    xcf='VDW',
                    xca='DRSLL',
                    spin='F',
                    dt=0.5,T=300,P=0.0,tstep=3000,
                    us='F',
                    maxiter=500,
                    singlek=False,
                    FreeAtoms=None):
    ''' siesta input generation
        us = use saved data
    ''' 
    elements = {'C':'6','H':'1','N':'7','O':'8','F':'9','Al':'13','Fe':'26'}
    cell = Atoms.get_cell()
    natm,atoms,X,table = get_neighbors(Atoms=Atoms,cell=cell,exception=['O-O','H-H'])
    m = molecules(natm,atoms,X,table=table,cell=cell)

    m,A  = enlarge(m,cell=cell,fac = fac,supercell=[1,1,1])
    cell = A.get_cell()
    n_m  = len(m)

    r = [0.0 for i in range(3)]
    r[0] = cell[0][0]*cell[0][0]+ cell[0][1]*cell[0][1]+ cell[0][2]*cell[0][2]
    r[0] = np.sqrt(r[0])
    r[1] = cell[1][0]*cell[1][0]+ cell[1][1]*cell[1][1]+ cell[1][2]*cell[1][2]
    r[1] = np.sqrt(r[1])
    r[2] = cell[2][0]*cell[2][0]+ cell[2][1]*cell[2][1]+ cell[2][2]*cell[2][2]
    r[2] = np.sqrt(r[2])
    
    kgrid = min(r)
    if singlek:
       kgrid=0.0

    atoms_label = A.get_chemical_symbols()
    species     = []
    for lab in atoms_label:
        if not lab in species:
           species.append(lab)

    nspecie = len(species)
    natm    = len(atoms_label)

    fdf = open('in.fdf','w')
    print('\n################## Species and atoms ##################',file=fdf)
    print('SystemName       siesta',file=fdf)
    print('SystemLabel      siesta',file=fdf)
    print('NumberOfSpecies  ',nspecie,file=fdf)
    print('NumberOfAtoms    ',natm,file=fdf)
    print('   ',file=fdf)
    print('   ',file=fdf)
    print('%block ChemicalSpeciesLabel',file=fdf)
    for i,spec in enumerate(species):
        print(i+1,elements[spec],spec,file=fdf)
    print('%endblock ChemicalSpeciesLabel',file=fdf)
    print('   ',file=fdf)
    print('   ',file=fdf)
    print('SolutionMethod   Diagon # ## OrderN or Diagon',file=fdf)
    print('PAO.BasisSize    DZP     # standard basis set, Like DZ plus polarization',file=fdf)
    print('MaxSCFIterations %d' %maxiter,file=fdf)
    # print('PAO.EnergyShift  100 meV',file=fdf)
    # print('PAO.BasisType    split',file=fdf)
    print('SpinPolarized    %s' %spin,file=fdf)
    print('   ',file=fdf)
    print('   ',file=fdf)
    print('DM.MixingWeight      0.4   ',file=fdf)
    print('DM.NumberPulay       9',file=fdf)
    print('DM.Tolerance         1.d-4',file=fdf)
    print('   ',file=fdf)
    print('   ',file=fdf)
    if opt!='SinglePoint':
       print('###################   RunInfo  ###################',file=fdf)
       print('MD.TypeOfRun     %s  # CG/FIRE optimize Verlet MD' %opt,file=fdf)
       if opt=='FIRE':
          print('MD.FIRE.TimeStep       0.3 fs',file=fdf)
          print('MD.FIRE.Mass           1.0',file=fdf)
       print('MD.VariableCell  %s' %VariableCell,file=fdf) 
       # print('MD.RelaxCellOnly true',file=fdf)
       print('MD.TargetStress  %f GPa' %P,file=fdf)
       if md: 
          print('MD.InitialTimeStep    1',file=fdf)
          print('MD.FinalTimeStep      %d' %tstep,file=fdf)
          print('MD.LengthTimeStep     %f fs' %dt,file=fdf)
          if not T is None:
             print('MD.InitialTemperature %f K ' %T,file=fdf)
          if opt=='Nose' or opt=='NoseParrinelloRahman':
             print('MD.TargetTemperature  %f K' %T,file=fdf)
             print('MD.NoseMass           100 Ry*fs**2',file=fdf)
          if opt=='NoseParrinelloRahman':
             print('MD.TargetPressure   %f GPa' %T,file=fdf)
             print('MD.ParrinelloRahmanMass 100 Ry*fs**2',file=fdf)
       else:
          print('MD.NumCGsteps    2000 ',file=fdf)
          print('MD.MaxCGDispl    0.1 Ang ',file=fdf)
       print('MD.MaxForceTol   0.03 eV/Ang ',file=fdf)
       print('MD.UseSaveXV     %s ' %us,file=fdf) # T for restart
       print('MD.UseSaveZM     false ',file=fdf)
    print('   ',file=fdf)
    print('   ',file=fdf)
    print('################### FUNCTIONAL ###################',file=fdf)
    print('XC.functional    %s    # Exchange-correlation functional type' %xcf,file=fdf)
    print('XC.Authors       %s    # Particular parametrization of xc func'%xca,file=fdf)
    print('   ',file=fdf)
    print('   ',file=fdf)
    print('MeshCutoff       200. Ry # Equivalent planewave cutoff for the grid ',file=fdf)
    print('KgridCutoff      %f Ang' %kgrid,file=fdf)
    print('   ',file=fdf)
    print('   ',file=fdf)
    print('WriteCoorInitial T',file=fdf)
    print('WriteCoorXmol    T',file=fdf)
    print('WriteMDhistory   T',file=fdf)
    print('WriteMullikenPop 1',file=fdf)
    print('WriteForces      T',file=fdf)
    print('   ',file=fdf)
    print('###################  GEOMETRY  ###################\n',file=fdf)
    print('LatticeConstant  1.00 Ang',file=fdf)
    print('%block LatticeVectors',file=fdf)
    print(cell[0][0],cell[0][1],cell[0][2],file=fdf)
    print(cell[1][0],cell[1][1],cell[1][2],file=fdf)
    print(cell[2][0],cell[2][1],cell[2][2],file=fdf)
    print('%endblock LatticeVectors',file=fdf)
    print('   ',file=fdf)
    
    if FreeAtoms is None:
       if coord=='cart':
          print('AtomicCoordinatesFormat Ang',file=fdf)
          print('   ',file=fdf)
          print('%block AtomicCoordinatesAndAtomicSpecies',file=fdf)
          for i,a in enumerate(A):
             print(a.x,a.y,a.z,species.index(atoms_label[i])+1,file=fdf)
          print('%endblock AtomicCoordinatesAndAtomicSpecies',file=fdf)
       elif coord=='zmatrix':
          print('   ',file=fdf)
          print('ZM.UnitsLength Ang',file=fdf)
          print('ZM.UnitsAngle rad',file=fdf)
          print('    ',file=fdf)
          print('%block Zmatrix',file=fdf)

          for mol in m:
              # print(len(mol.mol_index))
              if len(mol.mol_index)>1:
                 neighors,nneigh = [],[]
                 x,y,z = [],[],[]
                 for a in mol.mol_index:
                     x.append(A[a].x)
                     y.append(A[a].y)
                     z.append(A[a].z)

                 print('molecule_cartesian',file=fdf) # 0 for fixed, 1 for varying
                 min_x = min(x)
                 min_y = min(y)
                 min_z = min(z)
                 stx = mol.mol_index[x.index(min_x)]
                 sty = mol.mol_index[y.index(min_y)]
                 stz = mol.mol_index[z.index(min_z)]

                 max_x = max(x)
                 max_y = max(y)
                 max_z = max(z)
                 smax = mol.mol_index[x.index(max_x)]
                 smay = mol.mol_index[y.index(max_y)]
                 smaz = mol.mol_index[z.index(max_z)]

                 if len(table[stx])==1:
                    st = stx
                 elif len(table[sty])==1:
                    st = sty
                 elif len(table[stz])==1:
                    st = stz
                 elif len(table[smax])==1:
                    st = smax
                 elif len(table[smay])==1:
                    st = smay
                 elif len(table[smaz])==1:
                    st = smaz
                 else:
                      print('-  end atom not find!')
                      print(table)
                 # for st in mol.mol_index:
                 #     if len(table[st])==4:
                 #        pos = A.get_positions()
                 #        print_zmatrix(st,pos,neighors,nneigh,species,atoms_label,table,fdf)
                 #        break
                 pos = A.get_positions()
                 print_zmatrix(st,pos,neighors,nneigh,species,atoms_label,table,fdf)

              elif len(mol.mol_index)==1:
                 print('molecule_cartesian',file=fdf)
                 print(species.index(atoms_label[mol.mol_index[0]])+1,0,0,0,file=fdf)
          print('%endblock Zmatrix',file=fdf)
          # A.write('test.gen')
          print('    ',file=fdf)
    else:
       print('ZM.UnitsLength    Ang',file=fdf)
       print('   ',file=fdf)
       print('%block Zmatrix',file=fdf)
       print('cartesian',file=fdf)
       for i,a in enumerate(A):
          if i in FreeAtoms:
             print(species.index(atoms_label[i])+1,a.x,a.y,a.z,1,1,1,file=fdf)
          else:
             print(species.index(atoms_label[i])+1,a.x,a.y,a.z,0,0,0,file=fdf)
       print('%endblock Zmatrix',file=fdf)
    fdf.close()


def print_zmatrix(ind,pos,neighors,nneigh,species,atoms_label,table,fdf):
    # print(atoms_label[ind],ind,table[ind])
    if not ind in neighors:
       print(species.index(atoms_label[ind])+1,0,0,0,pos[ind][0],pos[ind][1],pos[ind][2],1,1,1,file=fdf)
       neighors.append(ind)
       nneigh.append(len(neighors))
    nind = neighors.index(ind)

    for a in table[ind]:
        if not a in neighors:
           i  = ind
           r   = np.subtract(pos[a],pos[i])
           rad = np.sqrt(np.sum(np.square(r)))
           r   = r/rad
           if len(neighors)==1:
              ia,ja,ka = nneigh[nind],0,0
              fix_x,fix_y,fix_z = 0,0,1
              # print(r[2]/rad)
              ang = np.arccos(np.divide(r[2],rad))
              if r[1]<0:
                 ang = - ang
              ri  = np.sqrt(r[0]*r[0]+r[1]*r[1])
              tor = np.arccos(np.divide(r[0],ri))

           elif len(neighors)==2:
              ia,ka = nneigh[nind],0
              if nneigh[-2] == nneigh[nind]:
                 ja = nneigh[-1]
                 j  = neighors[-1]
              else:
                 ja = nneigh[-2]
                 j  = neighors[-2]
              fix_x,fix_y,fix_z = 0,0,1
              
              rij = np.subtract(pos[j],pos[i])
              rij = rij/np.sqrt(np.sum(np.square(rij)))
              ang = np.arccos(np.dot(r,rij))
              # print('* ang',np.dot(rij,rpj),ang*180.0/3.1415926)

              rk = [pos[j][0],pos[j][1],pos[j][2]+1.0]
              rkj = np.subtract(rk,pos[j])
              rkj = rkj/np.sqrt(np.sum(np.square(rkj)))

              rpij = np.cross(r,rij)    # y axis: rijk, z axis: rij x: y X z
              rpij = rpij/np.sqrt(np.sum(np.square(rpij)))
              rijk = np.cross(rkj,rij)
              rijk = rijk/np.sqrt(np.sum(np.square(rijk)))
              tor  = np.arccos(np.dot(rpij,rijk))

              rx   = np.cross(rijk,rij)
              rx   = rx/np.sqrt(np.sum(np.square(rx)))
              pre  = np.dot(rpij,rx)
              if pre<0:
                 tor = - tor # 2.0*3.1415926
           else:
              fix_x,fix_y,fix_z = 0,0,0
              ia = nneigh[nind]
              ja,ka = None,None
              for aa in table[ind]:
                  if aa in neighors:
                     nam = nneigh[neighors.index(aa)]
                     if nam == ia:
                        continue
                     if ja is None:
                        ja = nam
                        j  = aa
                        # print(ja)
                     elif ka is None:
                        ka = nam
                        k  = aa
                        # print(ka)
                        break

              if ka is None:
                 for aa in table[ind]:
                     for aaa in table[aa]:
                         if aaa in neighors:
                            nam = nneigh[neighors.index(aaa)]
                            if nam != ia:
                               ka = nam
                               k  = aaa
                               # print(ka)
                               break

              rij = np.subtract(pos[j],pos[i])
              rij = rij/np.sqrt(np.sum(np.square(rij)))
              ang = np.arccos(np.dot(r,rij))

              rkj = np.subtract(pos[k],pos[j])
              rkj = rkj/np.sqrt(np.sum(np.square(rkj)))

              rpij = np.cross(r,rij)    # y axis: rijk, z axis: rij x: y X z
              rpij = rpij/np.sqrt(np.sum(np.square(rpij)))
              rijk = np.cross(rkj,rij)
              rijk = rijk/np.sqrt(np.sum(np.square(rijk)))
              tor  = np.arccos(np.dot(rpij,rijk))
              # print(rpij,rijk,tor)
              rx   = np.cross(rijk,rij)
              rx   = rx/np.sum(np.square(rx))
              pre  = np.dot(rpij,rx)
              if pre<0:
                 tor = -tor # 2.0*3.1415926
           print(species.index(atoms_label[a])+1,ia,ja,ka,rad,ang,tor,
                 fix_x,fix_y,fix_z,file=fdf)
 
           # print(atoms_label[a],ia,ja,ka)
           neighors.append(a)
           nneigh.append(len(neighors))

    for a in table[ind]:
        link = False
        for aa in table[a]:
            if not aa in neighors:
               link = True
               break
        if link:
           print_zmatrix(a,pos,neighors,nneigh,species,atoms_label,table,fdf)
    

def pseudo_gen(pa='vw',rc_dic=None):
    '''
      Pseudopotential generation
      pg: simple generation
    '''
    if rc_dic is None:
       rc_dic = {'C':1.15,'H':1.00,'N':1.15,'O':1.15}
    # pa = 'vw'# pseudo authour 
    for key in rc_dic:
        if exists(key+'.'+pa):
           system('rm -rf '+key+'.'+pa)
        if exists(key+'.'+pa+'.inp'):
           system('rm '+key+'.'+pa+'.inp') 
        fp = open(key+'.'+pa+'.inp','w')
        print('   pg %s TM2 Pseudopotencial GS ref' %key, file=fp)
        if key =='H':
           print('        tm2     2.00', file=fp)
        else:
           print('        tm2     2.00', file=fp)
        print(' n=%s  c=%s ' %(key,pa), file=fp)
        print('         0 ', file=fp)
        rc = rc_dic[key]
        if key=='H':
           print('    0    4', file=fp)
           print('    1    0      1.00      0.00', file=fp)
           print('    2    1      0.00      0.00', file=fp)
           print('    3    2      0.00      0.00', file=fp)
           print('    4    3      0.00      0.00', file=fp)
        elif key == 'O':
           print('    1    4', file=fp)
           print('    2    0      2.00      0.00', file=fp)
           print('    2    1      4.00      0.00', file=fp)
           print('    3    2      0.00      0.00', file=fp)
           print('    4    3      0.00      0.00', file=fp)
        elif key == 'N':
           print('    1    4', file=fp)
           print('    2    0      2.00      0.00', file=fp)
           print('    2    1      3.00      0.00', file=fp)
           print('    3    2      0.00      0.00', file=fp)
           print('    4    3      0.00      0.00', file=fp)
        elif key == 'C':
           print('    1    4', file=fp)
           print('    2    0      2.00      0.00', file=fp)
           print('    2    1      2.00      0.00', file=fp)
           print('    3    2      0.00      0.00', file=fp)
           print('    4    3      0.00      0.00', file=fp)

        print('      %4.2f      %4.2f      %4.2f      %4.2f      0.00      0.00' %(rc,rc,rc,rc), file=fp)
        print(' ', file=fp)
        print('#2345678901234567890123456789012345678901234567890 Ruler', file=fp)
        fp.close()
        print('* generating pseudo for %s, by Author %s    ' %(key,pa))
        system('./pg.sh '+key+'.'+pa+'.inp' )
        

def get_siesta_energy(label='siesta'):
    fe = open(label+'.MDE','r')
    lines = fe.readlines()
    fe.close()
    l = lines[-1].split()
    if len(l)==0:
       l = lines[-2].split()
    nframe = int(l[0])
    return nframe


def get_siesta_cart(label='siesta'):
    nframe = get_siesta_energy()
    fin = open('in.fdf','r') 
    lines= fin.readlines()
    fin.close()           # getting informations from input file
    atom_name = []

    calc = Siesta(label='Si',
                  xc='VDW',
                  mesh_cutoff=200 * Ry,
                  basis_set='DZP',
                  kpts=[2, 2, 2],
                  fdf_arguments={'DM.MixingWeight': 0.1,
                                 'MaxSCFIterations': 300})

    for i,line in enumerate(lines):
        l = line.split()
        if len(l)>0:
           if l[0] == 'NumberOfSpecies':
              ns = int(l[1])
           if l[0] == 'NumberOfAtoms':
              natom = int(l[1])
           if l[0]=='%block':
              if l[1]=='ChemicalSpeciesLabel':
                 spl = i+1
              if l[1]=='AtomicCoordinatesAndAtomicSpecies':
                 atml= i+1

    sp = []
    for isp in range(ns):
        l = lines[spl+isp].split() 
        sp.append(l[2])

    for na in range(natom):
        l = lines[atml+na].split() 
        atom_name.append(sp[int(l[3])-1])

    fe = open(label+'.MD_CAR','r')
    lines = fe.readlines()
    fe.close()
    nl = len(lines)
    if nl-(natom+7)*nframe!=0:
       fra = (nl-(natom+7)*nframe)/(natom+7)
       print('-  %d frames more than expected, error case            ' %fra)
       exit()
       
    lsp = lines[5].split()
    nsp = [int(l) for l in lsp]
    xs = []
    his = TrajectoryWriter('siesta.traj',mode='w')

    for nf in range(nframe):
        block = natom + 7
        nl = block*nf
        la = lines[nl+2].split()
        lb = lines[nl+3].split()
        lc = lines[nl+4].split()

        a = [float(la[0]),float(la[1]),float(la[2])]
        b = [float(lb[0]),float(lb[1]),float(lb[2])]
        c = [float(lc[0]),float(lc[1]),float(lc[2])]
        x = []
        il= 0
        for i,s in enumerate(nsp):
            for ns in range(s):
                l = lines[nl+7+il].split()
                xd = [float(l[0]),float(l[1]),float(l[2])]

                x1 = xd[0]*a[0]+xd[1]*b[0]+xd[2]*c[0]
                x2 = xd[0]*a[1]+xd[1]*b[1]+xd[2]*c[1]
                x3 = xd[0]*a[2]+xd[1]*b[2]+xd[2]*c[2]

                x.append([x1,x2,x3])
                il += 1
        xs.append(x)

        A = Atoms(atom_name,x,cell=[a,b,c],pbc=[True,True,True])
        A.set_calculator(calc)
        # A._calc.results['energy'] = 
        his.write(atoms=A)

    cell=np.array([a,b,c])
    his.close()



if __name__ == '__main__':
   from emdk import get_structure,emdk
   from ase.io import read,write
   from ase import Atoms
   from siesta import write_siesta_in

   supercell = [1,1,1]
   struc = 'cl20'
   get_structure(struc=struc,output='dftb',recover=True,center=True)

   A = read('card.gen')
   write_siesta_in(A,coord='zmatrix',fac=1.20)

