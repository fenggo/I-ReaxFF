#!/usr/bin/env python
from os.path import isfile
from os import system
import tensorflow as tf
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import json as js


def init_bonds(p_):
    spec,bonds,offd,angs,torp,hbs = [],[],[],[],[],[]
    for key in p_:
        # key = key.encode('raw_unicode_escape')
        # print(key)
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           # print(kk)
           if len(kk)==2:
              offd.append(k[1])
           elif len(kk)==1:
              spec.append(k[1])
        elif k[0]=='theta0':
           angs.append(k[1])
        elif k[0]=='tor1':
           torp.append(k[1])
        elif k[0]=='rohb':
           hbs.append(k[1])
    return spec,bonds,offd,angs,torp,hbs


def vdwpar(vdwlayer=[4,1]):
    with open('ffield.json','r') as lf:
         j = js.load(lf)
    p  = j['p']
    
    lf.close()
    spec,bonds,offd,angs,torp,hbs = init_bonds(p)

    with open('vdwtaper.json','r') as lf:
         jv = js.load(lf)

    # for bd in bonds:
    #     m_['fvwi_'+bd] = jv['wi']
    #     m_['fvbi_'+bd] = jv['bi']
    #     m_['fvwo_'+bd] = jv['wo']
    #     m_['fvbo_'+bd] = jv['bo']
    #     m_['fvw_'+bd] = jv['wh']
    #     m_['fvb_'+bd] = jv['bh']
    j['m']['fvwi'] = jv['wi']
    j['m']['fvbi'] = jv['bi']
    j['m']['fvwo'] = jv['wo']
    j['m']['fvbo'] = jv['bo']
    j['m']['fvw'] = jv['wh']
    j['m']['fvb'] = jv['bh']
    system('mv ffield.json ffield_.json')

    with open('ffield.json','w') as fj:
         js.dump(j,fj,sort_keys=True,indent=2)
    


def sigmoid(x):
    s = 1.0/(1.0+np.exp(-x))
    return s

def morseX(r,a=0.90,re=1.098):
    y = np.exp(-(r-re)/a)
    return y


def morseY(r,a=1.0,b=1.5,re=1.098):
    y = np.exp(-(r-re)/a-((r-re)**2.0)/b)
    return y


def vdwtaper(r,vdwcut=10.0):
    tp = 1.0+np.divide(-35.0,np.power(vdwcut,4.0))*np.power(r,4.0)+ \
         np.divide(84.0,np.power(vdwcut,5.0))*np.power(r,5.0)+ \
         np.divide(-70.0,np.power(vdwcut,6.0))*np.power(r,6.0)+ \
         np.divide(20.0,np.power(vdwcut,7.0))*np.power(r,7.0)
    return tp


def plotMorse():
    x    = np.linspace(0.0,5.0,50)
    x_   = np.linspace(0.0,5.0,50,dtype=np.float32)[:, np.newaxis]  
    s1   = morseX(x)
    s2   = morseY(x)
    s3   = morseX(x,a=0.37)
    NNa  = Linear(J='WBa.json')
    sNNa = NNa(x_)  
    # sNNa = np.squeeze(sNNa)
    NNb  = Linear(J='WBb.json')
    sNNb = NNb(x_)  
    # sNNb = np.squeeze(sNNb)
    a  = 0.90
 
    plt.figure() 
    plt.ylabel('X, Y and NN')
    plt.xlabel(r'r($\AA$)')
    plt.xlim(0,5.0)
    plt.ylim(0,3.0)

    plt.plot(x,s1,label=r'$X,a=0.90$ $\AA$',
             color='black', linewidth=1, linestyle='-')
    plt.plot(x,s3,label=r'$X,a=0.37$ $\AA$',
             color='blue', linewidth=1, linestyle='-')
    plt.plot(x,s2,label=r'$Y,a=1.0$ $\AA$ and $b=1.5$ $\AA^2$',
             color='red', linewidth=1, linestyle='-')

    plt.scatter(x,sNNa,label=r'$NeuralNetwork-fit$ $to$ $X$',
                marker = 'o',color='r',alpha=0.75,s=10)
    plt.scatter(x,sNNb,label=r'$NeuralNetwork-fit$ $to$ $Y$',
                marker = '^',color='b',alpha=0.75,s=10)

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('morse.pdf') 
    plt.close()


class Linear(tf.keras.Model):
    def __init__(self,J=None,nnlayer=[4,1]):
        super().__init__()
        self.hl       = nnlayer[1]
        self.wh       = []
        self.bh       = []
        self.wbf      = 'vdwtaper.json' if J is None else J
        if J is None or not isfile(self.wbf):
           self.wi  = tf.Variable(tf.random.normal([1,nnlayer[0]],stddev=0.2),name='wi')
           self.bi  = tf.Variable(tf.random.normal([nnlayer[0]],stddev=0.2),name='bi')
           
           for i in range(nnlayer[1]):
               self.wh.append(tf.Variable(tf.random.normal([nnlayer[0],nnlayer[0]],
                                stddev=0.2),name='wh_%d' %i))
               self.bh.append(tf.Variable(tf.random.normal([nnlayer[0]],
                                stddev=0.2),name='bh_%d' %i))

           self.wo  = tf.Variable(tf.random.normal([nnlayer[0],1],stddev=0.2),name='wo')
           self.bo  = tf.Variable(tf.random.normal([1],stddev=0.2),name='bo')
        else:
           with open(J,'r') as lf:
                j = js.load(lf)
           hl_ = len(j['wh'])
           self.wi  = tf.Variable(j['wi'],name='wi')
           self.bi  = tf.Variable(j['bi'],name='bi')

           for i in range(self.hl):
               if i <=hl_-1:
                  self.wh.append(tf.Variable(j['wh'][i],name='wh_%d' %i))
                  self.bh.append(tf.Variable(j['bh'][i],name='bh_%d' %i))
               else:
                  self.wh.append(tf.Variable(tf.random.normal([nnlayer[0],nnlayer[0]],
                                 stddev=0.2),name='wh_%d' %i))
                  self.bh.append(tf.Variable(tf.random.normal([nnlayer[0]],
                                 stddev=0.2),name='bh_%d' %i))

           self.wo  = tf.Variable(j['wo'],name='wo')
           self.bo  = tf.Variable(j['bo'],name='bo')


    def call(self, r):
        o  = []
        o.append(tf.sigmoid(tf.matmul(r,self.wi)+self.bi))
        for i in range(self.hl):
            o.append(tf.sigmoid(tf.matmul(o[-1],self.wh[i])+self.bh[i]))
        output   = tf.sigmoid(tf.matmul(o[-1],self.wo)+self.bo)
        return tf.squeeze(output)


    def save(self):
        with open(self.wbf,'w') as fj:
             j = {}
             wh = []
             bh = []
             for i in range(self.hl):
                 wh.append(self.wh[i].numpy().tolist())
                 bh.append(self.bh[i].numpy().tolist())
             j['wi'] = self.wi.numpy().tolist()
             j['bi'] = self.bi.numpy().tolist()
             j['wh'] = wh
             j['bh'] = bh
             j['wo'] = self.wo.numpy().tolist()
             j['bo'] = self.bo.numpy().tolist()
             js.dump(j,fj,sort_keys=True,indent=2)


def fitMorse(func,interactive=False,nnlayer=[8,1],convergence=0.00001):
    x   = np.linspace(0.01,10.0,100,dtype=np.float32)[:, np.newaxis]  
    y   = func(x)
    y   = np.squeeze(y)
 
    # neural network layers
    model = Linear(J='vdwtaper.json',nnlayer=nnlayer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    if interactive:
       plt.ion()   # something about plotting


    for step in range(5000):
        # train and net output
        with tf.GradientTape() as tape:
             y_pred = model(x)     
             loss = tf.reduce_mean(tf.square(y_pred - y))
        
        grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        if step % 10 == 0:
           if interactive:
              plt.cla()
              plt.scatter(x,y)
              plt.plot(x,y_pred, 'r-', lw=3)
              plt.text(0.5, 0, 'Step: %d Loss=%.4f' %(step,loss), fontdict={'size': 20, 'color': 'red'})
              plt.pause(0.1)
           else:
              print('Step: %d Loss=%.4f' %(step,loss))

        if loss<convergence:
           break

    if interactive: plt.ioff()
    
    plt.figure()   
    plt.scatter(x,y,c='none',edgecolors='blue',linewidths=1,
                marker='o',s=28,label=r'$Burchart$',
                alpha=1.0)
    
    yp_ = np.squeeze(y_pred.numpy())
    err= np.squeeze(y) - yp_

    plt.errorbar(x,yp_,yerr=err,
                 fmt='s',ecolor='r',color='r',ms=6,markerfacecolor='none',mec='r',
                 elinewidth=2,capsize=1,label=r'$Neural$ $Network-fit$')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('vdwtaper.pdf')

    model.save()


def func(x):
    s = sigmoid(x)
    return (s*s - 0.5)**2


def resolve():
    res = minimize_scalar(func,method='brent')
    print(res.x)
  
    t = sigmoid(0.8813735870089523)
    print(t*t)


if __name__ == '__main__':
   '''  
   '''
   # plotMorse()
   # fitMorse(morseX)
   fitMorse(vdwtaper,nnlayer=[9,0])
   vdwpar(vdwlayer=[9,0])


   
