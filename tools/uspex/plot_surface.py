#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from sklearn.ensemble import RandomForestRegressor

data_hb = {}
data_d  = {}
data_eb = {}
data_s  = {} # structure factor

with open('hbond.dat','r') as fh:
     for line in fh.readlines():
         if line.startswith('#'):
            continue
         l = line.split()
         data_hb[l[0]] = float(l[1])

with open('ebind.dat','r') as fe:
     for line in fe.readlines():
         if line.startswith('#'):
            continue
         l = line.split()
         data_eb[l[0]] = float(l[3])
         data_d[l[0]]  = float(l[-1])
s,ids = [],[]
with open('id.txt','r') as fi:
     for i,line in enumerate(fi.readlines()):
         ids.append(line.strip())

with open('Individuals','r') as fi:
     for i,line in enumerate(fi.readlines()):
         if i<=1:
            continue
         l = line.split()
         s.append(float(l[-1]))
data_s = {i_:s_ for i_,s_ in zip(ids,s)}
# print(data_hb)
# print(data_eb)
xx,X,Y,Z,ids = [],[],[],[],[]
for id_ in data_hb:
    X.append(data_d[id_])
    Y.append(data_hb[id_])
    # Y.append(data_s[id_])
    xx.append([data_d[id_],data_hb[id_]])
    Z.append(-data_eb[id_])
    ids.append(id_)

ml_model = RandomForestRegressor(n_estimators=100,max_depth=10,oob_score=True).fit(xx,Z)
score    = ml_model.score(xx,Z)      # cross_val_score(rfr,x,y,cv=10).mean()
print('score: ',score)

fig = plt.figure(figsize=(8,6))    ### 散点图
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z,color='#0ddbf5',s=100, depthshade=True,marker='*')

C = 0.8*(X - np.min(X))/(np.max(X)-np.min(X))

for i in range(len(X)):
    if X[i]<1.89:
       continue
    ax.scatter(X[i], Y[i], Z[i], 
               s=60, depthshade=True,marker='*', alpha=0.7,
               color=plt.cm.viridis(C[i]))   # 通过Viridis颜色图实现渐变
    ax.text(X[i],Y[i],Z[i]+0.03,ids[i].upper(),ha="center",va="center",fontsize=4)
    
X   = np.array(X)
Y   = np.array(Y)

ind = np.argsort(X)
X   = X[ind]
Y   = Y[ind]

X_, Y_ = np.meshgrid(X, Y)
n,_ = X_.shape

x_ = np.expand_dims(X_.flatten(),axis=1)
y_ = np.expand_dims(Y_.flatten(),axis=1)
xx = np.concatenate([x_,y_],axis=1)

z_ = ml_model.predict(xx)
# ax.scatter(X_.flatten(), Y_.flatten(), z_)
# plt.show()
Z_ = z_.reshape([n,n])  

ax.set_xlabel('Density', fontdict={'fontsize': 9})
ax.set_ylabel(r'$-$ Hbond Energy', fontdict={'fontsize': 9})
ax.set_zlabel('Binding Energy', fontdict={'fontsize': 9})
# plt.show()
plt.savefig('ebind.svg',transparent=True,bbox_inches='tight',pad_inches=0.25)
plt.close()

