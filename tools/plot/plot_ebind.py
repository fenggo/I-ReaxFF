#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# nlayer = 110

### Obtain the hac data
raw_data = np.loadtxt('ebind.dat', skiprows=1) 

ratio          = raw_data[:,0]                    
ebind_gulp     = raw_data[:,1]
ebind_gulp_    = raw_data[:,2]
ebind_siesta   = raw_data[:,3]
ebind_siesta_  = raw_data[:,4]

i_ = [i for i,_ in enumerate(ratio)]

raw_data       = np.loadtxt('q.dat')# delimiter='  ' 
q              = raw_data[:,1]  

plt.figure()

# plt.ylabel(r'$\kappa$ $(K)$', fontsize = 14)

# plt.yticks([0.00000000],labels=['     '])
# plt.xticks([])

plt.subplot(2,1,1)
err  = np.abs(ebind_siesta_ - ebind_gulp_)
plt.plot(ebind_gulp_, linewidth = 2.0,linestyle='-',marker='o',markerfacecolor='none',
           markeredgewidth=1,markeredgecolor='#4886B2',markersize=14,
           color='#4886B2',label='ReaxFF-nn(GULP)')  
plt.plot(ebind_siesta_, linewidth = 2.0,linestyle='-',marker='s',markerfacecolor='none',
           markeredgewidth=1,markeredgecolor='#D36F8A',markersize=12,
           color='#D36F8A',label='DFT(SIESTA)')  
# plt.errorbar(i_,ebind_gulp_,yerr=err,
#             fmt='-s',ecolor='b',color='b',ms=10,markerfacecolor='none',mec='blue',
#             elinewidth=2,capsize=2,label='ReaxFF-nn(GULP)')
plt.xticks(i_,
    labels=[' ',' ',' ',' ',' ',' ',' ',' ',' '])
plt.ylabel(r'$E_{bind}^*$', fontsize = 14)

plt.ylim(-1.4,-0.9)
plt.legend(loc='best',edgecolor='#CBB549') # lower left upper right

plt.subplot(2,1,2)

# plt.plot(q, linewidth = 2.0,linestyle='-',marker='s',markerfacecolor='none',
#            markeredgewidth=1,markeredgecolor='r',markersize=10,
#            color='r',label='DFT(SIESTA)') 
plt.bar(i_,q+1.1, facecolor='#4E8872',edgecolor='white',width=0.5,label='DFT(SIESTA)')

for x,y in zip(i_,q+1.1):
    plt.text(x,y,'{:.3f}'.format(y-1.1),ha='center',va='bottom')

# print(q+1.1)
plt.xticks(i_,
    labels=['4:1','3:1','2:1','1:1','1:2','1:3','1:4','1:5','1:6'])
plt.yticks([0.0,0.1,0.2,0.3],labels=[-1.1,-1.0,-0.9,-0.8])
plt.xlabel(r'$Molar$ $Ratio$', fontsize = 14)
plt.ylabel(r'$Enthalpy$ $Q$', fontsize = 14)
plt.legend(loc='best',edgecolor='#CBB549') # lower left upper right

# plt.subplots_adjust(wspace=0.3)
plt.savefig("Ebind.pdf", bbox_inches='tight',transparent='true')
# plt.show()



