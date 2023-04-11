#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
# import csv
import pandas as pd

d = pd.read_csv('Results.csv')
Y = d['Edft']
Yp = d['Epred']

c = np.abs(Y-Yp)
# c_max = np.max(c)
# c = c_max -c # 1.0 - c/(c_max*2.0)
# print(c)

cmap = plt.get_cmap("Reds")  # cm.coolwarm # 

plt.figure()
plt.xlabel('E(DFT)')
plt.ylabel('E(ReaxFF-nn)')
cb = plt.scatter(Y, Yp,
                 marker='o', edgecolor='r', s=25,
                 color='none',# cmap=cmap, c=c,  # zorder=-10,
                 alpha=0.6, label=r'$E(DFT)$ $V.S.$ $E(ReaxFF-nn)$')
#plt.colorbar(cb)

# loc = lower left upper right best
plt.legend(loc='best', edgecolor='yellowgreen')
plt.savefig('Result.svg', transparent=True)
plt.close()
