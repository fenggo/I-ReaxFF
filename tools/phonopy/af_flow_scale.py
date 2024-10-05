#!/usr/bin/env python
from os import system, listdir #,popen
from os import getcwd,chdir
from os.path import isfile
import numpy as np
from ase.io import read # ,write
from irff.md.gulp import opt,phonon

'''phonon compute work flow'''

# 0、 拷贝参数文件
# system('cp ../../ffield.json ffield.json')
# system('./json_to_lib.py')

def get_kappa(f):
    ''' '''
    with open(f) as f:
        for line in f:
            if 'Allen-Feldman' in line:
                return float(line.split()[4])
            
opt(gen='POSCAR.unitcell',step=1000,l=1,t=0.0000001,n=16, x=1,y=1,z=8)
# system('cp POSCAR.unitcell POSCAR.uc')
atoms = read('POSCAR.unitcell')
cell = atoms.get_cell()
# x = atoms.get_positions()
# m = np.min(x,axis=0)
# x_ = x - m
# atoms.set_positions(x_)

for i in [0.95,0.96,0.97,0.98,1.0,1.05,1.1,1.15,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]:
    # i = 18
    # 1、 优化结构
    # 2 、使用GULP计算二阶力常数
    atoms.set_cell(cell*i)
    atoms.write('POSCAR')

    opt(gen='POSCAR',step=100,l=0,t=0.0000001,n=16, x=1,y=1,z=8)
    phonon(gen='POSCAR.unitcell',T=300,step=300,t=0.0000001,n=16, x=1,y=1,z=22)
    system('cp phonon.out phonon_{:f}.out'.format(i))

    k = get_kappa('phonon.out')
    with open('af_cnt.txt','a') as f:
         print(i,k,file=f)

    
