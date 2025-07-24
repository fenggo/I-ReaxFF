#!/usr/bin/env python
from os import system, listdir #,popen
import sys
import argparse

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

parser = argparse.ArgumentParser(description='./train_torch.py --e=1000')
parser.add_argument('--c',default=0,type=int, help='calculator: 0 gulp 1 siesta 2 lammps')
args = parser.parse_args(sys.argv[1:])

# 1、 优化结构

if args.c==0:
   system('./gmd.py opt --s=1000 --g=POSCAR.unitcell  --n=8 --x=8 --y=8 --l=1')
elif args.c==1: # for siesta
   system('./smd.py opt --s=200 --g=POSCAR.unitcell  --n=8 --l=1')

# 2 、先将结构文件转换为siesta输入文件
if args.c==1: # for siesta
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
    system('./phonon_force.py --n={:d}'.format(i+1))
# system('cp force.0 lammps_forces_gp.0')

fs = ['Forces-00{:d}.FA'.format(i) for i in range(1,n+1)]
fs = ' '.join(fs)

system('phonopy -f {:s} --siesta'.format(fs))
system('phonopy --siesta -c in.fdf -p --dim="8 8 1" --band="0.0 0.0 0.0 1/4 0.0 0.0  0.5 0.0 0.0  2/3 -1/3 1/2 1/3 -1/6 0.0  0.0 0.0 0.0"')

system('phonopy-bandplot --gnuplot band.yaml > band.dat')

if args.c==0:
   calc='gulp'
elif args.c==1:
   calc='siesta'
    
system('mv band.dat band-{:s}.dat'.format(calc))
system('./plotband.py')

# 使用Phonopy计算二阶力常数
# system('phonopy --writefc --full-fc')

# 此时计算的二阶力常数的长度单位是Unit of length: au 转换成 AA
# force_unit()

# 计算三阶力常数
# system('./thirdorder_gulp.py sow 8 8 1  1  ') # (最后一个1：指1nm，即截断半径10埃)
# system('./thirdorder_gulp.py reap 8 8 1  1 ')
