#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_validate,train_test_split,cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


# normalize 归一化
normalizeFactor_x_w = [15000.0,300.0,300.0,1000.0]
normalizeFactor_x_b = [0.0,36200.0,36100.0,-500.0]
normalizeFactor_y_w = 2000.0
normalizeFactor_y_b = 50.0


def calculate_hugstate(mass=100.0,N=88,EV=None,epot=-36000.0,p0=None,t0=600,model=None):
    '''
        a machine learning object 
        N partical number 
        p0  e0  v0 = volu; t0  d0 = dens :  initial thermal condition
    '''
    e0,v0 = EV[0]
    # vol   = v0 
    d0  = (mass/v0)*10.0000/6.02253  # g/cm^-3

    # with open('hugstat.txt', 'a') as fhug:
    #      print('# T,P,E_tot,D,V,Up,Us,Dt',file=fhug)
    #      print(0.0, p0, e0, dens, vol, 0,0,0,file=fhug)

    print('*  Machine learning calculation of the Hugoniot state ...')

    T = t0
    for (energy,vol) in EV:
        dt = 131.0
        iter_ = 0
        epot_ = epot
        while abs(dt)>=5.0 and iter_<1000 and T>1.0:  # 5.0 3.0
              x = np.divide(np.array([T,epot_,energy,vol])+normalizeFactor_x_b,normalizeFactor_x_w)
              press = np.squeeze(model.predict([x]))  # Pressure: GPa, energy: eV
              press_ = (press*normalizeFactor_y_w-normalizeFactor_y_b)/10.0
              dt = (0.5*(press_+p0)*(v0-vol)*1.0e-3 + (e0 - energy)*0.1602177)/((3*N-3)*1.381*1.0e-5) 
              # print(press_,press,T,epot,energy,vol,dt)

              if dt>100.0:
                 dt_ =  100.0
              elif dt< -100.0:
                 dt_ = -100.0
              else:
                 dt_ = dt
              
              # p_ = 0.5*(press+p0)*(v0-vol)*1.0e-3 
              # e_ = (e0 - energy)*0.1602177
              # print(p_,e_,dt)
              if T<10000.0:
                 T      = T + dt_
                 energy = energy +  dt_*0.1*((3*N-3)*1.381*1.0e-5)/0.1602177
              else:
                 epot_  = epot_ -    dt_*0.1*((3*N-3)*1.381*1.0e-5)/0.1602177
              


              iter_ += 1

        dens = (mass/vol)*10.0000/6.02253  # g/cm^-3
        
        if abs(d0*(1.0-vol/v0))<0.00001:
           us = 0.0
        else:
           us = 0.001*(press_-p0)/(d0*(1.0-vol/v0))  #  m/s

        if us>0:
           us =np.sqrt(us)*0.001    # shock velocity # Km/s
        else:
           us = 0.0
        up = us*(1-vol/v0)    
        print(T,press_,epot_,energy,dens,vol,up,us,dt)
    


def plot_results(X_train,y_train,X_test,y_test,model):
    plt.figure()
    yp = model.predict(X_train)
    plt.scatter(y_train,yp,c='b',edgecolors='blue',linewidths=1,
                marker='x',s=16,label='train set',
                alpha=0.6)

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('LearningResults.pdf')
    plt.close()

    ##            performence of Test DataSet
    plt.figure()   
    yp = model.predict(X_test)
    plt.scatter(y_test,yp,c='r',edgecolors='r',linewidths=1,
                marker='x',s=16,label='test set',
                alpha=0.6)

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('LearningResultsTestSet.pdf')



d = pd.read_csv('MDE.csv')

# X = d['T'].tolist()
# Y = d['P'].tolist()

x_columns=['T','E_KS','E_tot','Vol']
y_columns=['P']

X = d[x_columns]
Y = d[y_columns]


X = np.divide(X+normalizeFactor_x_b,normalizeFactor_x_w)
Y = np.divide(Y+normalizeFactor_y_b,normalizeFactor_y_w)

# yp = y_pred*normalizeFactor_y_w-normalizeFactor_y_b


xmi = 0.95*np.min(d['Vol'])
xma = 1.05*np.max(d['Vol'])

####  分割数据集
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

####  机器学习模型定义、训练  

# gpr = GaussianProcessRegressor(kernel=ExpSineSquared() + WhiteKernel(1e-1),
#                                random_state=0).fit(X, Y)
# gpr.score(X, Y)
# model = RandomForestRegressor(random_state=37, n_estimators=300,
#                               min_weight_fraction_leaf=0.0,
#                               oob_score=True).fit(X_train,y_train)


model = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(32, 32), 
                     max_iter=10000).fit(X_train,y_train)

# model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
#                           n_estimators=500, random_state=37,
#                           learning_rate=0.5).fit(X_train,y_train)

####  模型得分
score = model.score(X_train, y_train)
print(' * train set score: ',score)
score = model.score(X_test, y_test)
print(' * test set score: ',score)
# print(' * oob score: ',rf.oob_score)


####  机器学习模型预测  
V  = np.linspace(630,780,50)
X_ = []
for v in V:
    X_.append([2000.0,-36000.0,-36060.0,v])
# X_ = np.expand_dims(X_,1)

X_ = np.divide(np.array(X_)+normalizeFactor_x_b,normalizeFactor_x_w)
Y_ = model.predict(X_)*normalizeFactor_y_w-normalizeFactor_y_b

# print(model.predict([X_[5]]))
# print(X_[5])
# print(model.predict([[2000.0,-36000.0,-36060.0,650.0]]))
# y_pred*normalizeFactor_y_w-normalizeFactor_y_b
plot_results(X_train,y_train,X_test,y_test,model)

###  画图  
plt.figure()   
plt.ylabel(r'$Pressure$ ($GPa$)')
plt.xlabel(r'$Volume$ ($\AA^3$)')
# plt.scatter(X,Y,alpha=0.9,marker='X',s=20,color='r',label='P-T')
plt.plot(V,Y_,alpha=0.9,linestyle='-',color='blue',label='P-V(fitted)')
plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
plt.savefig('PV.pdf',transparent=True) 
plt.close() 


EV = [( -36013.283,765.952), (-36009.97,700.400),(-36003.88,685.440),(-35958.452,658.560),
      (-35861.46,626.726),(-35917.997,649.147),(-35994.857,725.757)]
     

calculate_hugstate(mass=1631.891967824,N=88,EV=EV,epot=-36099.0,p0=58.2,t0=7677.0,model=model)


