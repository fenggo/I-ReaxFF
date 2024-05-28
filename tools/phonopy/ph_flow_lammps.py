#!/usr/bin/env python
from os import system, listdir 
'''
    phonon compute work flow with lammps
'''

inp = ('units real',
       'atom_style     charge',
       'read_data supercell-001',
       'pair_style     reaxff control nn yes checkqeq yes',
       'pair_coeff     * * ffield C',
       'fix    rex all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff',
       'mass 1  12.0000',
       'dump phonopy all custom 1 force.* id type x y z fx fy fz',
       'dump_modify phonopy format line \"%d %d %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\"',
       'run 0')

def add_charge(datafile='data'):
    ''' add charge for lammps data '''
    # print(datafile)
    with open(datafile, 'r') as df:
         lines = df.readlines()
    writpos = False
    with open(datafile, 'w') as df:
         for line in lines:
             if line.find('Atoms')>=0:
                writpos = True
             l = line.split()
             if writpos and len(l)>0 and line.find('Atoms')<0:
                txt = '{:s} {:s} {:f} {:s} {:s} {:s} \n'.format(l[0],l[1],0.0,l[2],l[3],l[4])
                df.write(txt)
             else:
                df.write(line)

def write_input(n):
    with open('in.lammps','w') as f:
         for line in inp:
             if line.find('read_data')>=0:
                # print('read_data supercell-{:03d}'.format(n))
                print('read_data supercell-{:03d} \n'.format(n),file=f)
             else:
                print(line,file=f)
         


# 1、优化晶胞结构
system('./lopt.py')

# 2、Generate supercells using a python script
system('python generate_displacements.py')

# 3、compute forces with lammps
n     = 0
files = listdir()
for f in files:
    if f.startswith('supercell-'):
       n += 1
# print(n)

for i in range(n):
    datafile = 'supercell-{:03d}'.format(i+1)
    add_charge(datafile)
    write_input(i+1)
    system('lammps < in.lammps')
    system('cp force.0 lammps_forces_gp.{:d}'.format(i))

txt = ['lammps_forces_gp.{:d}'.format(i) for i in range(n)]
txt = ' '.join(txt)
system('phonopy -f {:s}'.format(txt))
system('phonopy -c unitcell.yaml -p --dim="8 8 1" --band="0.0 0.0 0.0 1/4 0.0 0.0  0.5 0.0 0.0  2/3 -1/3 1/2 1/3 -1/6 0.0  0.0 0.0 0.0"')

system('phonopy-bandplot --gnuplot band.yaml > band.dat')

system('mv band.dat band-nn.dat')
system('./plotband.py')

# 使用Phonopy计算二阶力常数
system('phonopy --writefc --full-fc')

# 此时计算的二阶力常数的长度单位是Unit of length: au 转换成 AA
system('./force_unit.py')
