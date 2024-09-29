#!/usr/bin/env python
import sys
import random
#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# data = np.loadtxt('kappa_vs_qgrid_gp.txt')
data = np.loadtxt('af.txt')

x = data[:,0]
y = data[:,1]*10/3.35 # (3.609*0.53)

print('x\n',x)
print('y\n',y)

plt.figure(figsize=(8,6))

ax = plt.subplot()

ax.set_ylabel(r"$\kappa$", weight="medium",fontdict={"fontsize":18})
ax.set_xticks([8, 10, 12, 14,16,18,20,22,24,26,28])
ax.set_xticklabels([r"$8 \times 8$", r"$10 \times 10 $",
                    r"$12 \times 12 $", r"$14 \times 14 $",
                    r"$16 \times 16 $", r"$18 \times 18 $",
                    r"$20 \times 20 $", r"$22 \times 22 $",
                    r"$24 \times 24 $", r"$26 \times 26 $",
                    r"$28 \times 28 $"],rotation=45)

plt.xticks(fontsize=10)

plt.plot(x,y,alpha=0.8,
         linestyle='-',marker='s',markerfacecolor='none',
         markeredgewidth=1,markeredgecolor='r',markersize=10,
         color='r',label=r'$\kappa$ ($Graphene$)')
plt.legend(loc='best',edgecolor='yellowgreen',fontsize=16) 
plt.savefig('convergence.pdf')
plt.show()

