from os import system
from os.path import isfile
import numpy as np
import json as js
from ase import Atoms
from ase.io import read  # ,write
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
import matplotlib.pyplot as plt
from ..molecule import press_mol
from ..reaxfflib import write_lib

input_sample = '''
md qiterative conv;

temperature        350.000000 K;
timestep           0.100000 fs;
production         0.100000 ps;
equilibration      0.0 ps;
write              1;
sample             1;

title;
GULP calculation;
end;

library reax;
output movie xyz his.xyz;
dump 100 restart.grs;

'''

def write_gulp_in(A, runword='gradient qiterative nosymmetry conv verb debu',
                  time_step=0.1,
                  tot_step=10.0,
                  maxcyc=2000,
                  output=None,
                  restart=100,
                  lib='reax',movieFrequency=10,**kwargs):
    ''' runword = keyword in gulp input
        can be 'md conv' 'md conp' 'opti conp'
        qite = iterative solve charges, norx = no charges '''
    finp = open('inp-gulp', 'w')
    rw = runword.split()
    print(runword, file=finp)
    print('#', file=finp)
    print('#', file=finp)
    if rw[0] == 'opti':
        print('maxcyc %d' % maxcyc, file=finp)
        if 'gtol' in kwargs:
            print('gtol opt  {:f} '.format(kwargs['gtol']), file=finp)
        else:
            print('gtol opt  0.00001', file=finp)
    elif rw[0] == 'md':
        #  if rw[1]=='conv':
        #     print('ensemble nvt',file=finp)
        #     print('tau_thermostat     0.05 ps',file=finp)
        #  if rw[1]=='conp':
        #     print('integrator leapfrog verlet',file=finp)
        #     print('ensemble           npt',file=finp)
        #     print('pressure           0.00 GPa',file=finp)
        #     print('tau_barostat       0.1 ps',file=finp)
        #     print('tau_thermostat     0.1 ps',file=finp)
        if 'ensemble' in kwargs:
            print('ensemble   =  {:s} '.format(kwargs['ensemble']), file=finp)
        if 'tau_thermostat' in kwargs:
            print('tau_thermostat   =  {:f} ps'.format(kwargs['tau_thermostat']), file=finp)

        # print('temperature        %f K' % T, file=finp)
        print('timestep           %f fs' % time_step, file=finp)
        print('production         %f ps' %
              float(tot_step*time_step/1000.0), file=finp)
        print('equilibration      0.0 ps', file=finp)
        print('write              1', file=finp)
        print('sample             1', file=finp)
        
    if 'T' in kwargs: 
       print('temperature          {:f} K'.format(kwargs['T']),file=finp)
    if 'pressure' in kwargs:
       print('pressure          {:f} GPa'.format(kwargs['pressure']),file=finp)

    print('#', file=finp)
    print('title', file=finp)
    print('GULP calculation', file=finp)
    print('end', file=finp)
    print('#', file=finp)
    if lib == 'brenner':
        print('%s' % lib, file=finp)
    else:
        print('library %s' % lib, file=finp)

    if output is None:
        print('output movie {:d} xyz his.xyz'.format(
            movieFrequency), file=finp)
    else:
        print('output {:s}'.format(output), file=finp)
    print('output cif gulp produces gulp.cif', file=finp)
    print('dump {:d} restart.grs'.format(restart), file=finp)
    print('#', file=finp)

    print('vectors', file=finp)
    cell = A.get_cell()
    for a in cell:
        print(a[0], a[1], a[2], file=finp)
    print('\n', file=finp)

    print('cartesian', file=finp)
    pos = A.get_positions()
    symb = A.get_chemical_symbols()
    for sp, x in zip(symb, pos):
        print(sp, 'core', x[0], x[1], x[2], 0.0, 1.0, 0.0, 1, 1, 1, file=finp)
    finp.close()


def get_reaxff_q(natom, fo='out'):
    q = []
    fout = open(fo, 'r')
    lines = fout.readlines()
    fout.close()

    for i, line in enumerate(lines):
        if line.find('Final charges from ReaxFF :') >= 0:
            charge_line = i
        elif line.find('E(coulomb)') >= 0:
            ecoul = float(line.split()[2])
        elif line.find('E(self)') >= 0:
            eself = float(line.split()[2])
        elif line.find('E(vdw)') >= 0:
            evdw = float(line.split()[2])

    for i in range(natom):
        line = lines[i+5+charge_line]
        l = line.split()
        q.append(float(l[2]))
    return q, ecoul, eself, evdw


def get_reax_energy(fo='out'):
    fout = open(fo, 'r')
    for line in fout.readlines():
        if line.find('E(bond)') >= 0:
            ebond = float(line.split()[2])
        elif line.find('E(lonepair)') >= 0:
            elp = float(line.split()[2])
        elif line.find('E(over)') >= 0:
            eover = float(line.split()[2])
        elif line.find('E(under)') >= 0:
            eunder = float(line.split()[2])
        elif line.find('E(val)') >= 0:
            eang = float(line.split()[2])
        elif line.find('E(coa)') >= 0:
            tconj = float(line.split()[2])
        elif line.find('E(pen)') >= 0:
            epen = float(line.split()[2])
        elif line.find('E(tors)') >= 0:
            etor = float(line.split()[2])
        elif line.find('E(conj)') >= 0:
            fconj = float(line.split()[2])
        elif line.find('E(vdw)') >= 0:
            evdw = float(line.split()[2])
        elif line.find('E(hb)') >= 0:
            ehb = float(line.split()[2])
        elif line.find('E(coulomb)') >= 0:
            ecl = float(line.split()[2])
        elif line.find('E(self)') >= 0:
            esl = float(line.split()[2])
        elif line.find('ReaxFF force field') >= 0:
            e = float(line.split()[4])
    fout.close()
    return e, ebond, elp, eover, eunder, eang, epen, tconj, etor, fconj, evdw, ehb, ecl, esl

def get_md_results(out='gulp.out'):
    ''' get the result from the  gulp MD simulation '''
    flog = open(out, 'r')
    E, Epot, T, P = [], [], [], []

    for line in flog.readlines():
        l = line.split()
        if line.find('Potential energy  (eV) =') >= 0:
            Epot.append(float(l[4]))
        elif line.find('Temperature       (K)  =') >= 0:
            T.append(float(l[3]))
        elif line.find('Total energy      (eV) =') >= 0:
            E.append(float(l[4]))
        elif line.find('Pressure         (GPa) =') >= 0:
            P.append(float(l[3]))
    flog.close()
    return E, Epot, T, P


def plot_md(E, Epot, T, P, show=True):
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.ylabel(r'$Total Energy (Unit: eV)$')
    plt.xlabel(r'$Iterations$')
    plt.plot(E, label=r'$Total Energy$', color='blue', linewidth=2)
    plt.legend(loc='best', edgecolor='yellowgreen')

    plt.subplot(2, 2, 2)
    plt.ylabel(r'$Potential Energy (Unit: eV)$')
    plt.xlabel(r'$Iterations$')
    plt.plot(Epot, label=r'$Potential Energy$', color='blue', linewidth=2)
    plt.legend(loc='best', edgecolor='yellowgreen')

    plt.subplot(2, 2, 3)
    plt.ylabel(r'$Temperature (Unit: K)$')
    plt.xlabel(r'$Iterations$')
    plt.plot(T, label=r'$Temperature$', color='blue', linewidth=2)
    plt.legend(loc='best', edgecolor='yellowgreen')

    plt.subplot(2, 2, 4)
    plt.ylabel(r'$Pressure (Unit: GPa)$')
    plt.xlabel(r'$Iterations$')
    plt.plot(P, label=r'$Pressure$', color='blue', linewidth=2)
    plt.legend(loc='best', edgecolor='yellowgreen')

    if show:
        plt.show()
    else:
        plt.savefig('MD_results.pdf')
    plt.close()


def plot_optimize(log='gulp.out'):
    ''' plot the energies or other from gulp log file'''
    flog = open(log, 'r')
    cyc, e = [], []

    for line in flog.readlines():
        l = line.split()
        if line.find('Cycle:') >= 0 and line.find('Energy:') >= 0:
            cyc.append(float(l[1]))
            e.append(float(l[3]))

    flog.close()
    plt.figure()
    plt.ylabel(r'$Energy (Unit: eV)$')
    plt.xlabel(r'$Iterations$')
    # plt.xlim(6,500)
    # plt.ylim(-625,-600)
    plt.plot(cyc, e, label=r'$energy .VS. iterations$',
             color='blue', linewidth=2)
    plt.savefig('energy_vs_iteration.png')
    plt.close()


def get_lattice(inp='inp-gulp'):
    finp = open(inp, 'r')
    il = 0
    cell = []
    readlatv = False
    readlat = False
    for line in finp.readlines():
        l = line.split()
        if line.find('vectors') >= 0:
            readlatv = True

        if readlatv and il < 4:
            if il != 0:
                cell.append([float(l[0]), float(l[1]), float(l[2])])
            il += 1
    finp.close()
    return np.array(cell)


def reaxyz(fxyz):
    # cell = get_lattice()
    f = open(fxyz, 'r')
    lines = f.readlines()
    f.close()

    natom = int(lines[0])
    nframe = int(len(lines)/(natom+2))

    positions = np.zeros([nframe, natom, 3])
    atom_name = []
    energies = []
    for nf in range(nframe):
        le = lines[nf*(natom+2)+1].split()
        if le[2] != 'NaN':
            energies.append(float(le[2]))
            for na in range(natom):
                ln = nf*(natom+2)+2+na
                l = lines[ln].split()

                if nf == 0:
                    atom_name.append(l[0])

                positions[nf][na][0] = float(l[1])
                positions[nf][na][1] = float(l[2])
                positions[nf][na][2] = float(l[3])
    return atom_name, positions, energies

def xyztotraj(fxyz, inp='inp-gulp', checkMol=False, mode='w', traj='gulp.traj', scale=True):
    atom_name, positions, e = reaxyz(fxyz)
    cell = get_lattice(inp=inp)
    u = np.linalg.inv(cell)
    his = TrajectoryWriter(traj, mode=mode)

    for i, e_ in enumerate(e):
        if scale:
            pos_ = np.dot(positions[i], u)
            posf = np.mod(pos_, 1.0)          # aplling simple pbc conditions
            pos = np.dot(posf, cell)
        else:
            pos = positions[i]
        A = Atoms(atom_name, pos, cell=cell, pbc=[True, True, True])

        if checkMol:
            A = press_mol(A)

        calc = SinglePointCalculator(A, energy=e[i])
        A.set_calculator(calc)
        his.write(atoms=A)
        del A
    his.close()

def opt(atoms=None,T=350,gen='siesta.traj',step=200,i=-1,l=0,
        pressure=0.0,
        lib='reaxff_nn',ncpu=1):
    if atoms is None:
       atoms = read(gen,index=i)
    if l==0:
       runword='opti conv qiterative'
    elif l==1:
       runword= 'opti conp qiterative stre atomic_stress'

    write_gulp_in(atoms,runword=runword,
                  T=T,maxcyc=step,pressure=pressure,
                  lib=lib)
    # print('\n-  running gulp optimize ...')
    if ncpu > 1:
        system('mpirun -n %d gulp<inp-gulp>gulp.out' % ncpu)
    else:
        system('gulp<inp-gulp>gulp.out')

    atoms = read('gulp.cif')
    e = 0.0
    with open('gulp.out','r') as fo:
         for line in fo:
             if line.find('Final energy')>=0:
                e = float(line.split()[3])
    return e,atoms

def nvt(atoms=None, gen='poscar.gen', T=350, time_step=0.1, tot_step=100,
        keyword='md qiterative conv', movieFrequency=10,
        ncpu=1, index=-1, mode='w', lib='reaxff_nn'):
    if atoms is None:
        atoms = read(gen, index=index)
    write_gulp_in(atoms, runword=keyword,
                  T=T,
                  time_step=time_step,
                  tot_step=tot_step,
                  movieFrequency=movieFrequency,
                  lib=lib)
    print('\n-  running gulp nvt ...')
    if ncpu > 1:
        system('mpirun -n {:d} gulp<inp-gulp>gulp.out'.format(ncpu))
    else:
        system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode=mode,traj='md.traj',scale=False)


def npt(T=350, time_step=0.1, tot_step=10.0, ncpu=1):
    A = read('packed.gen')
    write_gulp_in(A, runword='md conp qiterative',
                  T=T,
                  time_step=time_step,
                  tot_step=tot_step,
                  lib='reax')
    if ncpu > 1:
        system('mpirun -n {:d} gulp<inp-gulp>gulp.out'.format(ncpu))
    else:
        system('gulp<inp-gulp>gulp.out')
    xyztotraj('his.xyz',mode='w',traj='md.traj',scale=False)


def gulp(atoms=None, gen='poscar.gen', inp=None, out='gulp.out', ncpu=1):
    if atoms is None:
        atoms = read(gen)

    if inp is None:
        inp = input_sample

    with open('gin', 'w') as finp:
        inp_ = inp.split(';')
        for line in inp_:
            l_ = line.strip()
            print(l_, file=finp)

        print(' ', file=finp)
        print('vectors', file=finp)
        cell = atoms.get_cell()
        for a in cell:
            print(a[0], a[1], a[2], file=finp)
        print('\n', file=finp)

        print('cartesian', file=finp)
        pos = atoms.get_positions()
        symb = atoms.get_chemical_symbols()
        for sp, x in zip(symb, pos):
            print(sp, 'core', x[0], x[1], x[2],
                  0.0, 1.0, 0.0, 1, 1, 1, file=finp)

    print('-  running gulp ...')
    if ncpu > 1:
        system('mpirun -n {:d} gulp< {:s} > {:s}'.format(ncpu, 'gin', out))
    else:
        system('gulp< {:s} > {:s}'.format('gin', out))
    if isfile('his.xyz'):
        xyztotraj('his.xyz')
    print('-  Job completed.')


def jstolib(nn=False, jsfile='ffield.json', libfile='reax.lib'):
    lf = open(jsfile, 'r')
    j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    bo_layer_ = j['bo_layer']
    mf_layer = j['mf_layer']
    lf.close()

    spec, bonds, offd, angs, torp, hbs = init_bonds(p_)
    write_lib(p_, spec, bonds, offd, angs, torp, hbs,
              m=m_, mf_layer=mf_layer, libfile=libfile)


def init_bonds(p_):
    spec, bonds, offd, angs, torp, hbs = [], [], [], [], [], []
    for key in p_:
        # key = key.encode('raw_unicode_escape')
        # print(key)
        k = key.split('_')
        if k[0] == 'bo1':
            bonds.append(k[1])
        elif k[0] == 'rosi':
            kk = k[1].split('-')
            # print(kk)
            if len(kk) == 2:
                if kk[0] != kk[1]:
                    offd.append(k[1])
            elif len(kk) == 1:
                spec.append(k[1])
        elif k[0] == 'theta0':
            angs.append(k[1])
        elif k[0] == 'tor1':
            torp.append(k[1])
        elif k[0] == 'rohb':
            hbs.append(k[1])
    return spec, bonds, offd, angs, torp, hbs


def get_gulp_energy(atoms, ffield='reaxff_nn'):
    write_gulp_in(atoms, runword='gradient nosymmetry conv qite verb',
                  lib=ffield)
    system('gulp<inp-gulp>out')

    (e_, eb_, el_, eo_, eu_, ea_, ep_,
     etc_, et_, ef_, ev_, ehb_, ecl_, esl_) = get_reax_energy(fo='out')
    ge = {}
    ge['ebond'] = eb_
    ge['elonepair'] = el_
    ge['eover'] = eo_
    ge['eunder'] = eu_
    ge['eangle'] = ea_
    ge['econjugation'] = etc_
    ge['epenalty'] = ep_
    ge['etorsion'] = et_
    ge['fconj'] = ef_
    ge['evdw'] = ev_
    ge['ecoul'] = ecl_
    ge['ehb'] = ehb_
    ge['eself'] = esl_
    ge['E'] = e_
    return ge


def get_gulp_force_from_drv(fdrv, wforce=False):
    forces = []
    with open(fdrv, 'r') as fdrv:
        gl = -1
        for i, line in enumerate(fdrv.readlines()):
            if line.find('gradients cartesian eV/Ang') >= 0:
                gl = i
                natom = int(line.split()[3])
            if gl > 0:
                if i > gl and i <= gl+natom:
                    forces.append([-float(line.split()[1]), -float(line.split()[2]),
                                   -float(line.split()[3])])
    # print(forces)
    if wforce:
        with open('Forces.FA', 'w') as fsiesta:
            print(natom, file=fsiesta)
            for i, f in enumerate(forces):
                print('{:4d} {:12.8f} {:12.8f} {:12.8f}'.format(
                    i+1, f[0], f[1], f[2]), file=fsiesta)  # Why?
                # print('{:4d} {:12.8f} {:12.8f} {:12.8f}'.format(i+1,f[0],f[1],f[2]),file=fsiesta)
    return np.array(forces)


def get_gulp_forces(images, traj='gulp_force.traj', ffield='reaxff_nn', wforce=False):
    ''' calculate the force by GULP '''
    his = TrajectoryWriter(traj, mode='w')
    for atoms in images:
        write_gulp_in(atoms, runword='gradient nosymmetry conv qite verb',
                      output='drv gulp', restart=1,
                      lib=ffield)
        system('gulp<inp-gulp>out')
        forces = get_gulp_force_from_drv('gulp.drv', wforce=wforce)
        #atoms.forces = forces
        (e, eb_, el_, eo_, eu_, ea_, ep_,
         etc_, et_, ef_, ev_, ehb_, ecl_, esl_) = get_reax_energy(fo='out')
        #e = atoms.get_potential_energy()
        calc = SinglePointCalculator(atoms, energy=e,
                                     forces=forces)

        atoms.set_calculator(calc)
        his.write(atoms=atoms)
    his.close()
    return atoms
