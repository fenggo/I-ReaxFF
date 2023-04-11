#!/usr/bin/env python
from os import getcwd, chdir,listdir
from os.path import isfile,isdir
import matplotlib.pyplot as plt
import numpy as np

# data     = np.loadtxt('kappaVsT.txt')
# data_pbe = np.loadtxt('PBE.txt')
# data_exp = np.loadtxt('Exp.txt')
kappa       = []
temperature = []

direcs  = listdir()

fdata = open('data_kappa.txt','w')
for d in direcs:
    if isdir(d):
       if d.startswith('T') and d.endswith('K'):
          t = float(d[1:-1])
          temperature.append(t)
          f_ = d+'/BTE.kappa_scalar'
          if isfile(f_):
             with open(f_,'r') as f:
                  line = f.readlines()[-1]
                  k = float(line.split()[-1])
                  kappa.append(k)
                  # print(t,k)
          else:
             print('no BTE.kappa_scalar file found in {:s}'.format(d))

kappa = np.array(kappa)
temperature = np.array(temperature)
ind_ = np.argsort(temperature)
kappa = kappa[ind_]
temperature = temperature[ind_]

for t,k in zip(temperature,kappa):
    fdata.write('{:f} {:f} \n'.format(t,k))
fdata.close()

plt.figure()
plt.ylabel(r'$Thermal$ $conducitvity$ ($\times 1000$ $Wm^{-1}K^{-1}$)')
plt.xlabel(r'$Temperature$ $(K)$')


plt.plot(temperature,kappa,alpha=0.9,
         linestyle='-',marker='^',markerfacecolor='none',
         markeredgewidth=1,markeredgecolor='blue',markersize=7,
         color='blue',label='ReaxFF-nn')

# plt.plot(data_pbe[:,0],data_pbe[:,1],alpha=0.9,
#          linestyle='-',marker='s',markerfacecolor='none',
#          markeredgewidth=1,markeredgecolor='r',markersize=7,
#          color='r',label='PBE')

# plt.scatter(data_exp[:,0],data_exp[:,1],marker='o',
#             color='none',s=40,alpha=0.7,edgecolors='darkorange',
#             label='Experiments')

plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
plt.savefig('kappaVsT.pdf',transparent=True) 
plt.close() 

