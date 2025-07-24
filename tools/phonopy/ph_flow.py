#!/usr/bin/env python
from os import system, listdir #,popen
import sys
import argparse
from irff.md.gulp import get_gulp_forces
from irff.md.lammps import get_lammps_forces
from irff.dft.siesta import parse_fdf,parse_fdf_species,single_point
from irff.irff import IRFF

'''phonon compute work flow
   使用Phononpy和GULP计算声子色散曲线
'''

def force_unit():
    ''' 二阶力常数的长度单位是Unit of length: au 转换成 Angstrom '''
    f0 = open('FORCE_CONSTANTS','r')
    f1 = open('FORCE_CONSTANTS_2ND','w')
    
    lines = f0.readlines()
    for line in lines:
        l = line.split()
        if len(l)==3:
           print('  {:17.12f} {:17.12f} {:17.12f}'.format(float(l[0])/0.52917721,
                 float(l[1])/0.52917721,float(l[2])/0.52917721),file=f1)
        else:
           print(line,end=' ',file=f1)
        
    f0.close()
    f1.close()

def get_supercell():
    n     = 0
    files = listdir()
    for f in files:
        if f.startswith('supercell-') and f.endswith('.fdf'):
           n += 1
    return n
    
def phonon_force(n,calc,ncpu):
    spec  = parse_fdf_species(fdf='in.fdf')
    atoms = parse_fdf('supercell-00{:d}'.format(n),spec=spec)
    #view(atoms)
    print('-  calculating structure {:d} ...'.format(n))
    if calc=='gulp':  
       atoms = get_gulp_forces([atoms])
    elif calc=='gap':
       atoms = get_lammps_forces(atoms,pair_style='quip',
           pair_coeff='* * Carbon_GAP_20_potential/Carbon_GAP_20.xml "" 6',
           units='metal',atom_style='atomic')
    elif calc=='siesta':
       atoms = single_point(atoms,cpu=ncpu,id=n, xcf='GGA',xca='PBE',basistype='split') 
    else:
       atoms = get_lammps_forces(atoms)
    forces = atoms.get_forces()
    
    with open('Forces.FA', 'w') as ff:
        print(len(atoms), file=ff)
        for i, f in enumerate(forces):
            print('{:4d} {:12.8f} {:12.8f} {:12.8f}'.format(
                  i+1, f[0], f[1], f[2]), file=ff)   
    system('mv Forces.FA Forces-00{:d}.FA'.format(n))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='./train_torch.py --e=1000')
    parser.add_argument('--c',default='gulp',type=str, help='calculator: gulp, siesta, or lammps')
    parser.add_argument('--n',default=8,type=int, help='number of cpu core')
    args = parser.parse_args(sys.argv[1:])
    
    # 1、 优化结构
    
    if args.c=='gulp':
       system('./gmd.py opt --s=1000 --g=POSCAR.unitcell  --n=8 --x=8 --y=8 --l=1')
    elif args.c=='siesta': # for siesta
       system('./smd.py opt --s=200 --g=POSCAR.unitcell  --n=8 --l=0')
    
    # 2 、先将结构文件转换为siesta输入文件
    if args.c=='siesta': # for siesta
       system('./smd.py w --g=id_unitcell.traj')
    else:
       system('./smd.py w --g=POSCAR.unitcell')
     
    # 3 、生成位移文件
    try:
       system('rm supercell-00*.fdf')
    except:
       pass
    system('phonopy --siesta -c=in.fdf -d --dim="8 8 1" --amplitude=0.01')
    
    n = get_supercell()
    # 4 、计算每个位移文件受力
    for i in range(n):
        # system('./phonon_force.py --n={:d} --c={:s}'.format(i+1,args.c))
        phonon_force(i+1,args.c,args.n)
    # system('cp force.0 lammps_forces_gp.0')
    
    fs = ['Forces-00{:d}.FA'.format(i) for i in range(1,n+1)]
    fs = ' '.join(fs)
    
    system('phonopy -f {:s} --siesta'.format(fs))
    system('phonopy --siesta -c in.fdf -p --dim="8 8 1" --band="0.0 0.0 0.0 1/4 0.0 0.0  0.5 0.0 0.0  2/3 -1/3 1/2 1/3 -1/6 0.0  0.0 0.0 0.0"')
    system('phonopy-bandplot --gnuplot band.yaml > band.dat')
    system('mv band.dat band-{:s}.dat'.format(args.c))
    system('./plotband.py')
    
    # 使用Phonopy计算二阶力常数
    # system('phonopy --writefc --full-fc')
    
    # 此时计算的二阶力常数的长度单位是Unit of length: au 转换成 AA
    # force_unit()
    
    # 计算三阶力常数
    # system('./thirdorder_gulp.py sow 8 8 1  1  ') # (最后一个1：指1nm，即截断半径10埃)
    # system('./thirdorder_gulp.py reap 8 8 1  1 ')
