#!/usr/bin/env python
from os import listdir
import subprocess
import sys
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from irff.md.gulp import get_gulp_forces
from irff.md.lammps import get_lammps_forces
from irff.dft.siesta import parse_fdf, parse_fdf_species, single_point, write_siesta_in
from irff.irff import IRFF

'''
phonon compute work flow
   使用Phononpy和GULP/Siesta/LAMMPS计算声子色散曲线
   体系: CL-20/TNT 共晶
'''
def read_banddata(bdfile):
    data = [[]]
    with open(bdfile,'r') as f:
        lines = f.readlines()
        ib = 0
        for i,line in enumerate(lines):
            if i<2: continue
            l = line.split()
            if len(l) == 0:
                data.append([])
                ib += 1
            else:
                #print(ib,l)
                data[ib].append((float(l[0]),float(l[1])))
    return data

def plotband(label=''):
    data_nn   = read_banddata('band-{:s}.dat'.format(label))

    plt.figure(figsize=(8,6))
    plt.grid(visible=True,which='major',axis='x',color='r',lw=1)
    ax = plt.subplot()
    ax.set_ylabel(r"$Frequency$ ($THz$)", weight="medium",fontdict={"fontsize":18})
    ax.set_xticks([0.00000000, 0.12362750, 0.19554850, 0.33884600])
    ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"])
    plt.xticks(fontsize=25)

    xmax= 0.0
    ymax= 0.0
    X   = []
    Y   = []

    for db in data_nn:
        #print(db)
        if len(db)==0: continue
        db = np.array(db)
        n = len(db[:,0])
        x = db[:,0]
        y = db[:,1]
        index_ = random.sample(range(n),20)
        x     = x[index_]
        y     = y[index_]
        xmax = np.max(x)
        ymax = np.max(y)
        # ax.plot(x,y,color='r',label='ReaxFF-nn')
        X.extend(x)
        Y.extend(y)
    # ax.scatter(X,Y,marker='o',color='none',edgecolors='b',s=1,label=label)
    ax.plot(X,Y,color='b',label=label)

    plt.xlim((0, xmax))
    plt.ylim((0., ymax+5.0))
    plt.legend(loc='upper center',ncol=2,edgecolor='yellowgreen',fontsize=16)
    plt.tight_layout()
    plt.savefig("band-{:s}.pdf".format(label))
    plt.close()

def force_unit():
    ''' 二阶力常数的长度单位是Unit of length: au 转换成 Angstrom '''
    f0 = open('FORCE_CONSTANTS', 'r')
    f1 = open('FORCE_CONSTANTS_2ND', 'w')

    lines = f0.readlines()
    for line in lines:
        l = line.split()
        if len(l) == 3:
            print('  {:17.12f} {:17.12f} {:17.12f}'.format(
                float(l[0]) / 0.52917721,
                float(l[1]) / 0.52917721,
                float(l[2]) / 0.52917721), file=f1)
        else:
            print(line, end=' ', file=f1)

    f0.close()
    f1.close()

def get_supercell():
    n = 0
    files = listdir()
    for f in files:
        if f.startswith('supercell-') and f.endswith('.fdf'):
            n += 1
    return n

def write_fdf(gen='POSCAR.unitcell'):
    A = read(gen, index=-1)
    print('\n-  writing siesta input ...')
    write_siesta_in(A, coord='cart', md=False, opt='CG',
                    KgridCutoff=10.0,
                    VariableCell='true', xcf='GGA', xca='PBE',
                    basistype='split')

def phonon_force(n, calc, ncpu):
    spec = parse_fdf_species(fdf='in.fdf')
    if n < 10:
        atoms = parse_fdf('supercell-00{:d}'.format(n), spec=spec)
    elif n < 100:
        atoms = parse_fdf('supercell-0{:d}'.format(n), spec=spec)
    else:
        atoms = parse_fdf('supercell-{:d}'.format(n), spec=spec)

    if calc == 'gulp':
        atoms = get_gulp_forces([atoms])
    elif calc == 'gap':
        atoms = get_lammps_forces(atoms, pair_style='quip',
                                  pair_coeff='* * Carbon_GAP_20_potential/Carbon_GAP_20.xml "" 6',
                                  units='metal', atom_style='atomic')
    elif calc == 'siesta':
        atoms = single_point(atoms, cpu=ncpu, id=n, xcf='GGA', xca='PBE', basistype='split')
    else:
        atoms = get_lammps_forces(atoms)
    forces = atoms.get_forces()

    with open('Forces.FA', 'w') as ff:
        print(len(atoms), file=ff)
        for i, f in enumerate(forces):
            print('{:4d} {:12.8f} {:12.8f} {:12.8f}'.format(
                i + 1, f[0], f[1], f[2]), file=ff)
    system('mv Forces.FA Forces-00{:d}.FA'.format(n))


def print_banner(title):
    '''打印步骤标题'''
    bar = '=' * 60
    print('\n' + bar)
    print('  {:s}'.format(title))
    print(bar)


if __name__ == '__main__':
    t_start = time.time()

    parser = argparse.ArgumentParser(description='Phonon dispersion calculation workflow')
    parser.add_argument('--c', default='ReaxFF-nn', type=str,help='calculator: ReaxFF-nn, siesta, gap, lammps')
    parser.add_argument('--n', default=8, type=int,help='number of CPU cores (MPI)')
    parser.add_argument('--g', default='md.traj',help='geometry file / atomic configuration')
    parser.add_argument('--s', default=0, type=int,help='from this step start calculation')
    args = parser.parse_args(sys.argv[1:])

    g    = args.g
    ncpu = args.n
    calc = args.c
    step = args.s
    
    if step <=1:
        print_banner('Phonon Dispersion Calculation Workflow')
        print('  System:     CL-20/TNT co-crystal')
        print('  Calculator: {:s}'.format(calc))
        print('  CPU cores:  {:d}'.format(ncpu))
        print('  Geometry:   {:s}'.format(g))
        print('  Supercell:  2 x 2 x 2')
        print('  Displacement amplitude: 0.01 Ang')

        # ============================================================
        # Step 1: 结构优化
        # ============================================================
        print_banner('Step 1/7: Structure Optimization')

        if calc == 'ReaxFF-nn':
            cmd = './gmd.py opt --s=1000 --g={:s} --n={:d}  --l=1'.format(g, ncpu) # --x=2 --y=2 --z=2 优化不使用超胞
            print('  Running GULP optimization (2x2x2 supercell, variable cell)...')
            print('  Command: {:s}'.format(cmd))
            subprocess.call(cmd, shell=True)
        elif calc == 'siesta':
            cmd = './smd.py opt --s=200 --g={:s} --n={:d} --l=0'.format(g, ncpu)
            print('  Running Siesta optimization (fixed cell)...')
            print('  Command: {:s}'.format(cmd))
            subprocess.call(cmd, shell=True)
        else:
            print('  Skipping optimization (using provided structure for {:s})'.format(calc))

        t1 = time.time()
        print('  [Done] Time elapsed: {:.1f} s'.format(t1 - t_start))
        
    # ============================================================
    # Step 2: 生成输入文件
    # ============================================================
    if step <=2:
        print_banner('Step 2/7: Generate Calculator Input (in.fdf)')

        if calc == 'siesta':
            print('  Writing Siesta input from id_unitcell.traj ...')
            subprocess.call('./smd.py w --g=id_unitcell.traj', shell=True)
        else:
            print('  Writing Siesta-format input from POSCAR.unitcell ...')
            write_fdf('POSCAR.unitcell')

        print('  [Done] in.fdf generated')

    # ============================================================
    # Step 3: 生成位移超胞
    # ============================================================
    n_supercell = get_supercell()
    if step <=3:
        print_banner('Step 3/7: Generate Displaced Supercells')

        try:
            subprocess.call('rm supercell-00*.fdf', shell=True)
        except Exception:
            pass

        print('  Running: phonopy --siesta -c=in.fdf -d --dim="2 2 2" --amplitude=0.01')             # 修改dim参数计算不同的超胞数，
        subprocess.call('phonopy --siesta -c=in.fdf -d --dim="2 2 2" --amplitude=0.01', shell=True)  # 以取得较好的计算结果

        print('  Generated {:d} displaced supercells'.format(n_supercell))
        print('  [Done]')

    # ============================================================
    # Step 4: 计算每个位移超胞的受力
    # ============================================================
    if step <=4:
        print_banner('Step 4/7: Calculate Forces for Each Displacement')

        for i in range(n_supercell):
            i1 = i + 1
            t_step_start = time.time()
            print('  [{:d}/{:d}] Calculating supercell-00{:d} ... \r'.format(i1, n_supercell, i1), end='', flush=True)
            phonon_force(i1, calc, ncpu)
            t_step = time.time() - t_step_start
            print(' done ({:.1f} s)'.format(t_step))

        t4 = time.time()
        print('  [Done] All {:d} force calculations completed ({:.1f} s)'.format(n_supercell, t4 - t1))

    # ============================================================
    # Step 5: 收集受力 → 力常数
    # ============================================================
    if step <=5:
        print_banner('Step 5/7: Collect Forces → FORCE_SETS')

        fs = ['Forces-00{:d}.FA'.format(i) for i in range(1, n_supercell + 1)]
        fs_str = ' '.join(fs)
        cmd = 'phonopy -f {:s} --siesta'.format(fs_str)
        print('  Running: {:s}'.format(cmd))
        subprocess.call(cmd, shell=True)
        print('  [Done] FORCE_SETS generated')

    # ============================================================
    # Step 6: 计算声子色散曲线
    # ============================================================
    if step <=6:
        print_banner('Step 6/7: Compute Phonon Band Structure')     # 修改能带计算路径

        band_path = ('0.0 0.0 0.0  '    # Gamma
                     '1/4 0.0 0.0  '     # M
                     '0.5 0.0 0.0  '     # K (approximate)
                     '2/3 -1/3 1/2  '    # Gamma (in hexagonal)
                     '1/3 -1/6 0.0  '    # intermediate point
                     '0.0 0.0 0.0')      # Gamma
        print('  K-path: G(0,0,0) -> M(1/4,0,0) -> K(1/2,0,0) -> G(2/3,-1/3,1/2) -> (1/3,-1/6,0) -> G(0,0,0)')

        cmd = 'phonopy --siesta -c in.fdf --dim="2 2 2" --band="{:s}"'.format(band_path) # -p
        print('  Running: {:s}'.format(cmd))
        subprocess.call(cmd, shell=True)
        print('  [Done] band.yaml and band.pdf generated')

    # ============================================================
    # Step 7: 数据提取与绘图
    # ============================================================
    if step <=7:
        print_banner('Step 7/7: Extract Band Data and Plot')

        print('  Extracting band data via phonopy-bandplot ...')
        subprocess.call('phonopy-bandplot --gnuplot band.yaml > band.dat', shell=True)

        cmd = 'mv band.dat band-{:s}.dat'.format(calc)
        subprocess.call(cmd, shell=True)
        print('  Saved as band-{:s}.dat'.format(calc))

        print('  Plotting comparison figure ...')
        # subprocess.call('./plotband.py', shell=True)
        plotband(calc)
        print('  [Done] band-{:s}.pdf generated'.format(calc))

    # ============================================================
    # 完成
    # ============================================================
    t_total = time.time() - t_start
    print_banner('Phonon Calculation Complete!')
    print('  Total time:  {:.1f} s ({:.1f} min)'.format(t_total, t_total / 60.0))
    print('  Calculator:  {:s}'.format(calc))
    print('  Supercells:  {:d}'.format(n_supercell))
    print('')
    print('  Output files:')
    print('    band-{:s}.dat    - Band structure data'.format(calc))
    print('    band.yaml        - Phonopy band data')
    print('    band.pdf         - Band structure plot')
    print('    FORCE_SETS       - Second-order force constants')
    print('    POSCAR.unitcell  - Optimized unit cell')
    print('')

    # 可选：二阶力常数单位转换 (au -> Angstrom)
    # system('phonopy --writefc --full-fc')
    # force_unit()

    # 可选：三阶力常数计算
    # system('./thirdorder_gulp.py sow 8 8 1  1  ')  # (最后一个1：指1nm，即截断半径10埃)
    # system('./thirdorder_gulp.py reap 8 8 1  1 ')
