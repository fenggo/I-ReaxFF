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
    
def plotdata(data,yunit=1.0,marker='^',edgecolors='r',label='ReaxFF-nn',unit=1.0):
    global xmax
    global ymax
    X   = []
    Y   = []

    for db in data:
        #print(db)
        if len(db)==0: continue
        db = np.array(db)
        n = len(db[:,0])
        x = db[:,0]*unit
        y = db[:,1]*yunit
        
        index_ = random.sample(range(n),20)
        x     = x[index_]
        y     = y[index_]
        xmax_ = np.max(x)
        ymax_ = np.max(y)
        if xmax_>xmax:
           xmax = xmax_
        if ymax_>ymax:
           ymax = ymax_
        # ax.plot(x,y,color='r',label='ReaxFF-nn')
        X.extend(x)
        Y.extend(y)
    ax.scatter(X,Y,marker=marker,color='none',edgecolors=edgecolors,s=10,label=label)

if __name__ == '__main__':
    data_nn   = read_banddata('band-nn-gulp.dat')
    data_qe   = read_banddata('band-siesta.dat')
    data_quip = read_banddata('band-quip.dat')
    data_dp = read_banddata('band-dp.dat')
    
    plt.figure(figsize=(8,6))
    plt.grid(visible=True,which='major',axis='x',color='r',lw=1)
    ax = plt.subplot()
    ax.set_ylabel(r"$Frequency$ ($THz$)", weight="medium",fontdict={"fontsize":18})
    ax.set_xticks([0.00000000, 0.12362750, 0.19554850, 0.34])
    ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"])
    plt.xticks(fontsize=25)
    
    xmax= 0.0
    ymax= 0.0
    
    plotdata(data_nn,marker='^',edgecolors='r',label='ReaxFF-nn') # 
    plotdata(data_qe,marker='o',edgecolors='b',label='DFT(Siesta)')
    plotdata(data_quip,unit=0.529177,marker='v',edgecolors='g',label='GAP-20')
    plotdata(data_dp,unit=0.529177,marker='s',edgecolors='y',label='DeePMD')
    
    plt.xlim((0, xmax+0.001))
    plt.ylim((0., ymax+5.0))
    plt.legend(loc='upper center',ncol=2,edgecolor='yellowgreen',fontsize=16)
    plt.tight_layout()
    plt.savefig("band.pdf")
    plt.show()

