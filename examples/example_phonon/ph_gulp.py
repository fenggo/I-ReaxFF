#!/usr/bin/env python
from os import system #,popen

#先将结构文件转换为siesta输入文件
system('./smd.py wi --g=POSCAR')
 
#生成位移文件
system('phonopy --siesta -c=in.fdf -d --dim="8 8 1" --amplitude=0.02')
system('./phonon_force.py --n=1')
# system('cp force.0 lammps_forces_gp.0')

system('phonopy -f Forces-001.FA --siesta')
system('phonopy --siesta -c in.fdf -p --dim="6 6 1" --band="0.0 0.0 0.0 1/4 0.0 0.0  0.5 0.0 0.0  2/3 -1/3 1/2 1/3 -1/6 0.0  0.0 0.0 0.0"')

system('phonopy-bandplot --gnuplot band.yaml > band.dat')

system('mv band.dat band-nn-gulp.dat')
system('./plotband.py')

