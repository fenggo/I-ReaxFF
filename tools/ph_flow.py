#!/usr/bin/env python
from os import system, listdir #,popen

'''phonon compute work flow'''

# 1、 优化结构
system('./gmd.py opt --s=300 --g=POSCAR.unitcell  --n=4 --x=8 --y=8 --l=1')

# 2 、先将结构文件转换为siesta输入文件
system('./smd.py wi --g=POSCAR.unitcell')
 
# 3 、生成位移文件
system('rm supercell-00*.fdf')
system('phonopy --siesta -c=in.fdf -d --dim="8 8 1" --amplitude=0.02')

n     = 0
files = listdir()
for f in files:
    if f.startswith('supercell-') and f.endswith('.fdf'):
       n += 1

for i in range(n):
    system('./phonon_force.py --n={:d}'.format(i+1))
# system('cp force.0 lammps_forces_gp.0')

fs = ['Forces-00{:d}.FA'.format(i) for i in range(1,n+1)]
fs = ' '.join(fs)

system('phonopy -f {:s} --siesta'.format(fs))
system('phonopy --siesta -c in.fdf -p --dim="8 8 1" --band="0.0 0.0 0.0 1/4 0.0 0.0  0.5 0.0 0.0  2/3 -1/3 1/2 1/3 -1/6 0.0  0.0 0.0 0.0"')

system('phonopy-bandplot --gnuplot band.yaml > band.dat')

system('mv band.dat band-nn-gulp.dat')
system('./plotband.py')

# 使用Phonopy计算二阶力常数
system('phonopy --writefc --full-fc')

# 此时计算的二阶力常数的长度单位是Unit of length: au 转换成 AA
system('./force_unit.py')

# 计算三阶力常数
# system('./thirdorder_gulp.py sow 8 8 1  1  ') # (最后一个1：指1nm，即截断半径10埃)
# system('./thirdorder_gulp.py reap 8 8 1  1 ')
