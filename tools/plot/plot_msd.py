#!/usr/bin/env python
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

help_  = 'run commond with: ./plot_msd.py --f=msd.data'

parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--f',default='msd.data',type=str, help='force field file name')
args = parser.parse_args(sys.argv[1:])

data     = np.loadtxt(args.f)

t     = data[:,0]*0.00001
msd_x = data[:,1]
msd_y = data[:,2]
msd_z = data[:,3]
msd   = data[:,4]

print('the average msd: ',np.mean(msd))

plt.figure()     
plt.plot(t,msd_x,alpha=0.8,linewidth=1,linestyle='-',color='k',
         marker='v', markeredgecolor='b', markeredgewidth=1,markersize=5,markerfacecolor='none',
         label=r'$msd@X$')
plt.plot(t,msd_y,alpha=0.8,linewidth=1,linestyle='-',color='k',
         marker='>', markeredgecolor='g', markeredgewidth=1,markersize=5,markerfacecolor='none',
         label=r'$msd@Y$')
plt.plot(t,msd_z,alpha=0.8,linewidth=1,linestyle='-',color='k',
         marker='<', markeredgecolor='y', markeredgewidth=1,markersize=5,markerfacecolor='none',
         label=r'$msd@Z$')
plt.plot(t,msd,alpha=0.8,linewidth=2,linestyle='-',color='r',
         marker='o', markeredgecolor='r', markeredgewidth=1,markersize=5,markerfacecolor='none',
         label=r'$msd@Average$')
plt.legend(loc='best',edgecolor='yellowgreen')
plt.savefig('msd.pdf')
# plt.show()
plt.close()

