#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from irff.tools.load_individuals import load_density_energy


I,D,E = load_density_energy('Individuals')
# print(I)
data_hb = {}
data_d  = {}
data_eb = {}
data_s  = {} # structure factor

with open('hbond.dat','r') as fh:
     for line in fh.readlines():
         if line.startswith('#'):
            continue
         l = line.split()
         if len(l)>0:
            data_hb[l[0]] = float(l[1])
            data_eb[l[0]] = float(l[3])
            data_d[l[0]]  = float(l[-1])

xx,X,Y,Z,ids = [],[],[],[],[]
for id_ in data_hb:
    X.append(data_d[id_])
    Y.append(data_hb[id_])
    # Y.append(data_s[id_])
    xx.append([data_d[id_],data_hb[id_]])
    Z.append(data_eb[id_])
    ids.append(id_)

# ml_model = RandomForestRegressor(n_estimators=100,max_depth=10,oob_score=True).fit(xx,Z)
# score    = ml_model.score(xx,Z)      # cross_val_score(rfr,x,y,cv=10).mean()
# print('score: ',score)

fig = plt.figure(figsize=(8,6))    ### 散点图
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z,color='#0ddbf5',s=100, depthshade=True,marker='*')

C = 0.875*(X - np.min(X))/(np.max(X)-np.min(X))

emph    = []
nx      = []
emph_id = ['epsilon','beta','gamma']
nx_id   = ['426'] # '263','271',

for i in range(len(X)):
    if X[i]<1.7:
       continue
    if ids[i] in emph_id:
       emph.append(i)
    elif ids[i] in nx_id:
       nx.append(i)
    else:
       ax.scatter(X[i], Y[i], Z[i], 
                  s=60, depthshade=True,marker='*', alpha=0.7,
                  facecolor='none',
                  color=plt.cm.viridis(C[i]))   # 通过Viridis颜色图实现渐变
       # if ids[i] in I:
       ax.text(X[i],Y[i],Z[i]+0.04,ids[i].upper(),ha="center",va="center",fontsize=4)

# print(nx)
for i in nx:
    label_ = ids[i] # 'A'+
    ax.text(X[i],Y[i],Z[i]+0.04,label_,ha="center",va="center",color='b',fontsize=4)
    ax.scatter(X[i], Y[i], Z[i], 
               s=60, depthshade=True,marker='*', alpha=0.7,
               facecolor='none',
               color=plt.cm.inferno(C[i]))   # 通过Viridis颜色图实现渐变

for i in emph:
    if ids[i]=='beta':
       label_ = r'$\beta$-CL-20'
    elif ids[i]=='epsilon':
       label_ = r'$\varepsilon$-CL-20'
    elif ids[i]=='gamma':
       label_ = r'$\gamma$-CL-20'
    else:
       label_ = ids[i]
    ax.text(X[i],Y[i],Z[i]-0.04,label_,ha="center",va="center",fontsize=6)
    ax.scatter(X[i], Y[i], Z[i], 
               s=100, depthshade=True,marker='*', alpha=0.5,
               facecolor='b', # #2ec0c2
               color='r')   # 通过Viridis颜色图实现渐变
    #  else:
    #     ax.scatter(X[i], Y[i], Z[i], 
    #             s=240,marker='*', alpha=0.8,depthshade=True,
    #             color='r')   # 通过Viridis颜色图实现渐变
     
    #     ax.text(X[i],Y[i],Z[i]+0.045,ids[i].upper(),ha="center",va="center",fontsize=6)

ax.set_xlabel('Density', fontdict={'fontsize': 9})
ax.set_ylabel(r'$-$ Hbond Energy', fontdict={'fontsize': 9})
ax.set_zlabel('Binding Energy', fontdict={'fontsize': 9})
# plt.show()
plt.savefig('ebind.svg',transparent=True,bbox_inches='tight',pad_inches=0.25)
plt.close()

