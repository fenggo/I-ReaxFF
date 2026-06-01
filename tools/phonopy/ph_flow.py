#!/usr/bin/env python
from os import system, listdir
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
# from pymatgen.symmetry.kpath import SeekpathKPath
# from pymatgen.symmetry.kpath import HighSymmKpath
# from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

'''
phonon compute work flow
   使用Phononpy和GULP/Siesta/LAMMPS计算声子色散曲线
   体系: CL-20/TNT 共晶
'''


def get_band_path(structure_file='POSCAR.unitcell'):
    '''使用pymatgen HighSymmKpath动态生成高对称K点路径。

    return:
        band_str  : 给 phonopy --band="..." 用的字符串 (segment之间用 ',' 分隔)
        label_str : 给 phonopy --band-labels="..." 用的字符串
        labels    : 标签列表（每个segment的顺序）
        kpoints   : dict {label: [kx,ky,kz]}
        path_segs : 多段路径列表（嵌套list）
    '''
    atoms = read(structure_file, index=-1)
    structure = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(structure)
    primitive_std = sga.get_primitive_standard_structure(international_monoclinic=False)

    bz_kpath = HighSymmKpath(primitive_std)
    kpath = bz_kpath.kpath
    path_segs = kpath['path']        # e.g. [['\\Gamma','M','K','\\Gamma'], ...]
    kpoints   = kpath['kpoints']     # {label: [kx,ky,kz]}

    band_str_segments  = []
    label_str_segments = []
    flat_labels        = []
    for seg in path_segs:
        coords = []
        for lab in seg:
            kp = kpoints[lab]
            coords.append('{:.6f} {:.6f} {:.6f}'.format(kp[0], kp[1], kp[2]))
            flat_labels.append(lab)
        band_str_segments.append('  '.join(coords))
        label_str_segments.append(' '.join(seg))

    band_str  = ' , '.join(band_str_segments)
    label_str = ' '.join(label_str_segments)
    print('-  band path  : {:s}'.format(' | '.join(label_str_segments)))
    print('-  band kpts  : {:s}'.format(band_str))
    return band_str, label_str, flat_labels, kpoints, path_segs

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

def _tex_label(lab):
    '''把 pymatgen 标签转成 matplotlib mathtext。'''
    s = lab.replace('\\Gamma', r'\Gamma')
    # 已经是 Gamma/Sigma 等带反斜杠的，用 $...$ 包起来
    if '\\' in s or '_' in s:
        return r'${:s}$'.format(s)
    return r'${:s}$'.format(s)


def _segment_tick_positions(path_segs, kpoints, xmax_total):
    '''按高对称点之间的笛卡尔间距（用分数坐标欧氏距离近似）算累计x坐标，再
    线性缩放到 [0, xmax_total]，使其与phonopy band.dat里的x对齐。
    '''
    cum = [0.0]
    labels = [path_segs[0][0]]
    for seg in path_segs:
        for i in range(1, len(seg)):
            k0 = np.array(kpoints[seg[i - 1]])
            k1 = np.array(kpoints[seg[i]])
            cum.append(cum[-1] + float(np.linalg.norm(k1 - k0)))
            labels.append(seg[i])
        # segment 之间的不连续：phonopy band.dat 也会在此处重置x，但为简单起见
        # 这里仅在 segment 之间留一个 0 间距的占位符（视觉上重叠的两个label）
    cum = np.array(cum)
    if cum[-1] > 0:
        cum = cum * (xmax_total / cum[-1])
    return cum.tolist(), labels


def plotband(label='', path=None, kpoints=None):
    data_nn = read_banddata('band-{:s}.dat'.format(label))

    plt.figure(figsize=(8, 6))
    plt.grid(visible=True, which='major', axis='x', color='r', lw=1)
    ax = plt.subplot()
    ax.set_ylabel(r"$Frequency$ ($THz$)", weight="medium", fontdict={"fontsize": 18})

    xmax = 0.0
    ymax = 0.0
    X = []
    Y = []

    for db in data_nn:
        if len(db) == 0:
            continue
        db = np.array(db)
        x = db[:, 0]
        y = db[:, 1]
        xmax = max(xmax, float(np.max(x)))
        ymax = max(ymax, float(np.max(y)))
        X.append(x)
        Y.append(y)

    # 动态设置xticks
    if path is not None and kpoints is not None and xmax > 0:
        tick_pos, tick_labs = _segment_tick_positions(path, kpoints, xmax)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([_tex_label(l) for l in tick_labs])
    plt.xticks(fontsize=20)

    for x, y in zip(X, Y):
        ax.plot(x, y, color='b')

    plt.xlim((0, xmax))
    plt.ylim((0., ymax + 5.0))
    plt.tight_layout()
    plt.savefig("band-{:s}.svg".format(label))
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
    subprocess.call('mv Forces.FA Forces-{:d}.FA'.format(n), shell=True)


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
            print('  Running GULP optimization (variable cell)...')
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
            # subprocess.call('./smd.py w --g=id_unitcell.traj', shell=True)
            write_fdf('id_unitcell.traj')
        else:
            print('  Writing Siesta-format input from POSCAR.unitcell ...')
            write_fdf('POSCAR.unitcell')

        print('  [Done] in.fdf generated')

    # ============================================================
    # Step 3: 生成位移超胞
    # ============================================================
   
    if step <=3:
        print_banner('Step 3/7: Generate Displaced Supercells')

        try:
            subprocess.call('rm supercell-00*.fdf', shell=True)
        except Exception:
            pass

        print('  Running: phonopy --siesta -c=in.fdf -d --dim="2 2 2" --amplitude=0.01')             # 修改dim参数计算不同的超胞数，
        subprocess.call('phonopy --siesta -c=in.fdf -d --dim="2 2 2" --amplitude=0.01', shell=True)  # 以取得较好的计算结果
    n_supercell = get_supercell()
    print('  Generated {:d} displaced supercells'.format(n_supercell))
    print('  [Done]')

    # print('--------->',n_supercell)
    # ============================================================
    # Step 4: 计算每个位移超胞的受力
    # ============================================================
    if step <=4:
        print_banner('Step 4/7: Calculate Forces for Each Displacement')

        for i in range(n_supercell):
            i1 = i + 1
            t_step_start = time.time()
            print('  [{:d}/{:d}] Calculating supercell-{:d} ... \r'.format(i1, n_supercell, i1), end='\r', flush=True)
            phonon_force(i1, calc, ncpu)
            # t_step = time.time() - t_step_start
            # print(' done ({:.1f} s)'.format(t_step))

        t4 = time.time()
        print('  [Done] All {:d} force calculations completed ({:.1f} s)'.format(n_supercell, t4 - t1))

    # ============================================================
    # Step 5: 收集受力 → 力常数
    # ============================================================
    if step <=5:
        print_banner('Step 5/7: Collect Forces → FORCE_SETS')

        fs = ['Forces-{:d}.FA'.format(i) for i in range(1, n_supercell + 1)]
        fs_str = ' '.join(fs)
        cmd = 'phonopy -f {:s} --siesta'.format(fs_str)
        # print('  Running: {:s}'.format(cmd))
        subprocess.call(cmd, shell=True)
        print('  [Done] FORCE_SETS generated')

    # ============================================================
    # Step 6: 计算声子色散曲线
    # ============================================================
    if step <=6:
        print_banner('Step 6/7: Compute Phonon Band Structure')

        # 使用pymatgen HighSymmKpath动态生成高对称K点路径
        if calc == 'siesta':
            struct_file = 'id_unitcell.traj'
        else:
            struct_file = 'POSCAR.unitcell'
        band_path, label_str, _, _, _ = get_band_path(struct_file)

        cmd = ('phonopy --siesta -c in.fdf --dim="2 2 2" '
               '--band="{:s}" --band-labels="{:s}"').format(band_path, label_str)
        print('  Running: {:s}'.format(cmd))
        subprocess.call(cmd, shell=True)
        print('  [Done] band.yaml and band.dat generated')

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

        # 重用Step 6生成的K路径（如果跳过Step 6，则在此处重新生成）
        if calc == 'siesta':
            struct_file = 'id_unitcell.traj'
        else:
            struct_file = 'POSCAR.unitcell'
        _, _, _, kpoints, path = get_band_path(struct_file)

        plotband(calc, path=path, kpoints=kpoints)
        print('  [Done] band-{:s}.svg generated'.format(calc))

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
    print('    band.svg         - Band structure plot')
    print('    FORCE_SETS       - Second-order force constants')
    print('    POSCAR.unitcell  - Optimized unit cell')
    print('')

    # 可选：二阶力常数单位转换 (au -> Angstrom)
    # system('phonopy --writefc --full-fc')
    # force_unit()

    # 可选：三阶力常数计算
    # system('./thirdorder_gulp.py sow 8 8 1  1  ')  # (最后一个1：指1nm，即截断半径10埃)
    # system('./thirdorder_gulp.py reap 8 8 1  1 ')
