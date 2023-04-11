#!/usr/bin/env python
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from scipy.optimize import leastsq

parser = argparse.ArgumentParser(description='fit a morse potential model')
parser.add_argument('--i',default=0,type=int, help='i atom')
parser.add_argument('--j',default=1,type=int, help='j atom')
args = parser.parse_args(sys.argv[1:])

def fmorse(x,p=[1.518,1.0,8.0,0.0]):
    ro,De,alpha,e0 = p
    r_ = x - ro
    return De*(np.exp(-2.0*alpha*r_)-2.0*np.exp(-alpha*r_)) + e0

def residuals(p, y, x):
    """
    计算目标值y和拟合值之间的代价
    :param p: 拟合用到的参数
    :param y: 样本结果
    :param x: 训练样本
    :return: 目标值y和拟合值之间的差
    """
    return y - fmorse(x, p)


images = Trajectory('md.traj')
i = args.i
j = args.j
X = []
Y = []
for atoms in images:
    positions = atoms.get_positions()
    vr = positions[i] - positions[j]
    r  = np.sqrt(np.sum(vr*vr,axis=0))
    X.append(r)
    Y.append(atoms.get_potential_energy())
X = np.array(X)
Y = np.array(Y)

e0 = max(Y)    
p0 = [1.518,1.0,8.0,e0]     
p  = leastsq(residuals,p0,args=(Y, X))
print('The optimal paramter is ro={:f}, De={:f}, alpha={:f}.'.format(p[0][0],p[0][1],p[0][2]))
  

Y_ = fmorse(X,p[0])

plt.figure()     
plt.plot(X,Y,alpha=0.8,linewidth=2,linestyle='-',color='y',label=r'$Energy$')
plt.plot(X,Y_,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$E_{morse}(fitted)$')

plt.legend(loc='best',edgecolor='yellowgreen')
plt.show() # if show else plt.savefig('deb_bo.pdf')
plt.close()

