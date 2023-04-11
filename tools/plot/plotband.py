#!/usr/bin/env python
import sys
import random
#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# data_ref = np.loadtxt('band_ref.txt')

# data=np.loadtxt('band.dat')
# x = data[:,0]
# y = data[:,1]
def read_banddata(bdfile):
    data = [[]]
    with open(bdfile,'r') as f:
        lines = f.readlines()
        ib = 0
        for i,line in enumerate(lines):
            if i<2: continue
            l = line.split()
            if len(l) == 0:
                data.append([])
                ib += 1
            else:
                #print(ib,l)
                data[ib].append((float(l[0]),float(l[1])))
    return data

data_nn = read_banddata('band-nn.dat')
data_qe = read_banddata('band-siesta.dat')

plt.figure(figsize=(8,6))
plt.grid(visible=True,which='major',axis='x',color='r',lw=1)
ax = plt.subplot()
ax.set_ylabel(r"$Frequency$ ($THz$)", weight="medium",fontdict={"fontsize":18})
ax.set_xticks([0.00000000, 0.12362750, 0.19554850, 0.33884600],fontdict={"fontsize":18})
ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"],fontdict={"fontsize":18})

xmax= 0.0
ymax= 0.0
X   = []
Y   = []

for db in data_nn:
    #print(db)
    if len(db)==0: continue
    db = np.array(db)
    n = len(db[:,0])
    x = db[:,0]
    y = db[:,1]
    index_ = random.sample(range(n),20)
    x     = x[index_]
    y     = y[index_]
    xmax = np.max(x)
    ymax = np.max(y)
    # ax.plot(x,y,color='r',label='ReaxFF-nn')
    X.extend(x)
    Y.extend(y)
ax.scatter(X,Y,marker='^',color='none',edgecolors='r',s=10,label='ReaxFF-nn')

X   = []
Y   = []
xmax = 0.0
ymax = 0.0
for db in data_qe:
    if len(db)==0: continue
    db = np.array(db)
    x = db[:,0]
    y = db[:,1]
    n = len(db[:,0])
    # print(n)
    index_ = random.sample(range(n),15)
    x     = x[index_]
    y     = y[index_]
    xmax_ = np.max(x)
    ymax_ = np.max(y)
    if xmax_>xmax:
       xmax = xmax_
    if ymax_>ymax:
       ymax = ymax_
    #ax.plot(x,y,color='b',label='DFT(Siesta)')
    # ax.scatter(x,y,marker='^',color='none',edgecolors='b',s=10)
    X.extend(x)
    Y.extend(y)
ax.scatter(X,Y,marker='o',color='none',edgecolors='b',s=10,label='DFT(Siesta)')
# ax.scatter(data_ref[:,0],data_ref[:,1],marker='v',color='none',edgecolors='b',s=50)

plt.xlim((0, xmax))
plt.ylim((0., ymax+5.0))
plt.legend(loc='upper center',ncol=2,edgecolor='yellowgreen',fontsize=16)
plt.tight_layout()
plt.savefig("band.pdf")
plt.show()

