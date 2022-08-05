#!/usr/bin/env python
from os.path import isfile
from os import system
import argh
import argparse
import json as js
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from irff.ml.gpdata import get_data
from irff.data.ColData import ColData
from irff.initCheck import init_bonds

def resolve():
    res = minimize_scalar(func,method='brent')
    print(res.x)
    t = sigmoid(0.8813735870089523)
    print(t*t)


class Linear_be(tf.keras.Model):
    def __init__(self,bonds=None,level=1):
        super().__init__()
        with open('ffield.json','r') as lf:
            self.j = js.load(lf)
        self.spec,bonds_,offd,angs,torp,hbs = init_bonds(self.j['p'])
        self.bonds = bonds_ if bonds is None else bonds 
        self.m = {}
        hidelayer  = self.j['be_layer'][1]
        self.level = level
        for bd in self.bonds:
            self.m['fewi_'+bd] = tf.Variable(self.j['m']['fewi_'+bd],name='fewi_'+bd)
            self.m['febi_'+bd] = tf.Variable(self.j['m']['febi_'+bd],name='febi_'+bd)
            self.m['fewo_'+bd] = tf.Variable(self.j['m']['fewo_'+bd],name='fewo_'+bd)
            self.m['febo_'+bd] = tf.Variable(self.j['m']['febo_'+bd],name='febo_'+bd)
            self.m['few_'+bd]  = []
            self.m['feb_'+bd]  = []
            for i in range(hidelayer):
                self.m['few_'+bd].append(tf.Variable(self.j['m']['few_'+bd][i],name='fewh_'+bd))
                self.m['feb_'+bd].append(tf.Variable(self.j['m']['feb_'+bd][i],name='febh_'+bd))

        if self.level==2:
            for sp in self.spec:
                self.m['fmwi_'+sp] = self.j['m']['fmwi_'+sp]
                self.m['fmbi_'+sp] = self.j['m']['fmbi_'+sp]
                self.m['fmwo_'+sp] = self.j['m']['fmwo_'+sp]
                self.m['fmbo_'+sp] = self.j['m']['fmbo_'+sp]
                self.m['fmw_'+sp]  = []
                self.m['fmb_'+sp]  = []
                for i in range(self.j['mf_layer'][1]):
                    self.m['fmw_'+sp].append(self.j['m']['fmw_'+sp][i])
                    self.m['fmb_'+sp].append(self.j['m']['fmb_'+sp][i])
        elif self.level==3:
            for sp in self.spec:
                self.m['fmwi_'+sp] = tf.Variable(self.j['m']['fmwi_'+sp],name='fmwi_'+sp)
                self.m['fmbi_'+sp] = tf.Variable(self.j['m']['fmbi_'+sp],name='fmbi_'+sp)
                self.m['fmwo_'+sp] = tf.Variable(self.j['m']['fmwo_'+sp],name='fmwo_'+sp)
                self.m['fmbo_'+sp] = tf.Variable(self.j['m']['fmbo_'+sp],name='fmbo_'+sp)
                self.m['fmw_'+sp]  = []
                self.m['fmb_'+sp]  = []
                for i in range(self.j['mf_layer'][1]):
                    self.m['fmw_'+sp].append(tf.Variable(self.j['m']['fmw_'+sp][i],name='fmwh_'+sp))
                    self.m['fmb_'+sp].append(tf.Variable(self.j['m']['fmb_'+sp][i],name='fmbh_'+sp))

    def call(self,Bp,D,B,E):
        # compute F
        loss = 0.0
        for bd in self.bonds:
            bp = np.array(Bp[bd]).astype(np.float32) 
            d  = np.array(D[bd]).astype(np.float32) 
            d_t= d[:,[2,1,0]] 
            # b  = np.array(B[bd]).astype(np.float32) 
            e  = np.expand_dims(E[bd],axis=1).astype(np.float32) 
            atomi,atomj = bd.split('-')

            if self.level>1:
               ai   = tf.sigmoid(tf.matmul(d,self.m['fmwi_'+atomi])  + self.m['fmbi_'+atomi])
               ah   = tf.sigmoid(tf.matmul(ai,self.m['fmw_'+atomi][0]) + self.m['fmb_'+atomi][0])
               ao   = tf.sigmoid(tf.matmul(ah,self.m['fmwo_'+atomi]) + self.m['fmbo_'+atomi])

               ai_t = tf.sigmoid(tf.matmul(d_t,self.m['fmwi_'+atomj]) + self.m['fmbi_'+atomj])
               ah_t = tf.sigmoid(tf.matmul(ai_t,self.m['fmw_'+atomj][0]) + self.m['fmb_'+atomj][0])
               ao_t = tf.sigmoid(tf.matmul(ah_t,self.m['fmwo_'+atomj]) + self.m['fmbo_'+atomj])

               b = bp*ao*ao_t
            else:
               b  = np.array(B[bd]).astype(np.float32)  

            ai   = tf.sigmoid(tf.matmul(b,self.m['fewi_'+bd])  + self.m['febi_'+bd])
            if self.j['be_layer'][1]>0:
               ah = tf.sigmoid(tf.matmul(ai,self.m['few_'+bd][0]) + self.m['feb_'+bd][0])
               ao = tf.sigmoid(tf.matmul(ah,self.m['fewo_'+bd]) + self.m['febo_'+bd])
            else:
               ao = tf.sigmoid(tf.matmul(ai,self.m['fewo_'+bd]) + self.m['febo_'+bd])

            e_pred = ao
            # loss+= tf.sqrt(tf.reduce_sum(tf.square((e-e_pred)*self.j['p']['Desi_'+bd]*4.3364432032e-2)))
            loss  += tf.nn.l2_loss((e-e_pred)*self.j['p']['Desi_'+bd]*4.3364432032e-2)
        return loss

    def save(self):
        if self.level>2:
           for sp in self.spec:
                self.j['m']['fmwi_'+sp] = self.m['fmwi_'+sp].numpy().tolist()
                self.j['m']['fmbi_'+sp] = self.m['fmbi_'+sp].numpy().tolist()
                self.j['m']['fmwo_'+sp] = self.m['fmwo_'+sp].numpy().tolist()
                self.j['m']['fmbo_'+sp] = self.m['fmbo_'+sp].numpy().tolist()

                for i in range(self.j['mf_layer'][1]):
                    self.j['m']['fmw_'+sp][i] = self.m['fmw_'+sp][i].numpy().tolist()
                    self.j['m']['fmb_'+sp][i] = self.m['fmb_'+sp][i].numpy().tolist()

        for bd in self.bonds:
            self.j['m']['fewi_'+bd] = self.m['fewi_'+bd].numpy().tolist()
            self.j['m']['febi_'+bd] = self.m['febi_'+bd].numpy().tolist()
            self.j['m']['fewo_'+bd] = self.m['fewo_'+bd].numpy().tolist()
            self.j['m']['febo_'+bd] = self.m['febo_'+bd].numpy().tolist()

            for i in range(self.j['be_layer'][1]):
                self.j['m']['few_'+bd][i] = self.m['few_'+bd][i].numpy().tolist()
                self.j['m']['feb_'+bd][i] = self.m['feb_'+bd][i].numpy().tolist()
        
        with open('ffield.json','w') as fj:
             js.dump(self.j,fj,sort_keys=True,indent=2)

class Linear_bo(tf.keras.Model):
    def __init__(self):
        super().__init__()
        with open('ffield.json','r') as lf:
            self.j = js.load(lf)
        self.spec,self.bonds,offd,angs,torp,hbs = init_bonds(self.j['p'])

        self.m = {}
        for sp in self.spec:
            self.m['fmwi_'+sp] = tf.Variable(self.j['m']['fmwi_'+sp],name='fmwi_'+sp)
            self.m['fmbi_'+sp] = tf.Variable(self.j['m']['fmbi_'+sp],name='fmbi_'+sp)
            self.m['fmwo_'+sp] = tf.Variable(self.j['m']['fmwo_'+sp],name='fmwo_'+sp)
            self.m['fmbo_'+sp] = tf.Variable(self.j['m']['fmbo_'+sp],name='fmbo_'+sp)
            self.m['fmw_'+sp]  = []
            self.m['fmb_'+sp]  = []
            for i in range(self.j['mf_layer'][1]):
                self.m['fmw_'+sp].append(tf.Variable(self.j['m']['fmw_'+sp][i],name='fmwh_'+sp))
                self.m['fmb_'+sp].append(tf.Variable(self.j['m']['fmb_'+sp][i],name='fmbh_'+sp))
        
    def call(self,Bp,D,B,E):
        # compute F
        loss = 0.0
        for bd in Bp:
            bp = np.array(Bp[bd]).astype(np.float32) 
            d  = np.array(D[bd]).astype(np.float32) 
            d_t= d[:,[2,1,0]] 
            b  = np.array(B[bd]).astype(np.float32) 
            atomi,atomj = bd.split('-')

            ai   = tf.sigmoid(tf.matmul(d,self.m['fmwi_'+atomi])  + self.m['fmbi_'+atomi])
            ah   = tf.sigmoid(tf.matmul(ai,self.m['fmw_'+atomi][0]) + self.m['fmb_'+atomi][0])
            ao   = tf.sigmoid(tf.matmul(ah,self.m['fmwo_'+atomi]) + self.m['fmbo_'+atomi])

            ai_t = tf.sigmoid(tf.matmul(d_t,self.m['fmwi_'+atomj]) + self.m['fmbi_'+atomj])
            ah_t = tf.sigmoid(tf.matmul(ai_t,self.m['fmw_'+atomj][0]) + self.m['fmb_'+atomj][0])
            ao_t = tf.sigmoid(tf.matmul(ah_t,self.m['fmwo_'+atomj]) + self.m['fmbo_'+atomj])

            b_pred = bp*ao*ao_t
            # loss+= tf.sqrt(tf.reduce_sum(tf.square(b-b_pred)))
            loss  += tf.nn.l2_loss(b-b_pred)
        return loss

    def save(self):
        for sp in self.spec:
            self.j['m']['fmwi_'+sp] = self.m['fmwi_'+sp].numpy().tolist()
            self.j['m']['fmbi_'+sp] = self.m['fmbi_'+sp].numpy().tolist()
            self.j['m']['fmwo_'+sp] = self.m['fmwo_'+sp].numpy().tolist()
            self.j['m']['fmbo_'+sp] = self.m['fmbo_'+sp].numpy().tolist()

            for i in range(self.j['mf_layer'][1]):
                self.j['m']['fmw_'+sp][i] = self.m['fmw_'+sp][i].numpy().tolist()
                self.j['m']['fmb_'+sp][i] = self.m['fmb_'+sp][i].numpy().tolist()
        
        with open('ffield.json','w') as fj:
             js.dump(self.j,fj,sort_keys=True,indent=2)


def train(Bp,D,B,E,convergence=0.000001,
        learning_rate=0.01,
        step=5000,fitobj='BO',
        level=1):
 
    # neural network layers
    if fitobj == 'BO':
       model = Linear_bo()
    elif fitobj == 'BE':
       model = Linear_be(level=level)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for step in range(step):
        # train and net output
        with tf.GradientTape() as tape:
             # y_pred = model(x)     
             loss =  model(Bp,D,B,E)  # tf.reduce_mean(tf.square(y_pred - y))
        
        grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        if step % 10 == 0:
           print('Step: %d Loss=%.8f' %(step,loss))
        if step % 1000 == 0:
           model.save()
        if loss<convergence:
           model.save()
           break


def fit(step=1000,obj='BO',level=1):
    dataset = {'dia-0':'data/dia-0.traj',
            'dia-1':'data/dia-1.traj',
            'dia-2':'data/dia-2.traj',
            # 'dia-3':'data/dia-3.traj',
            'gp2-0':'data/gp2-0.traj',
            'gp2-1':'data/gp2-1.traj',
            'gpd-0':'data/gpd-0.traj',
            'gpd-1':'data/gpd-1.traj',
            'gpd-2':'data/gpd-2.traj',
            'gpd-3':'data/gpd-3.traj',
            'gpd-4':'data/gpd-4.traj',
            'gpd-5':'data/gpd-5.traj',
            #'gpd-6':'data/gpd-6.traj',
            #'gpd-7':'data/gpd-7.traj',
            #'gpd-8':'data/gpd-8.traj',
            #'gpd-9':'data/gpd-9.traj',
            }

    trajdata   = ColData()
    strucs = [#'c32',
             #'c6',
             #'c10',
             #'ch4',
             ]
    batchs = {'others':50}

    for mol in strucs:
        b = batchs[mol] if mol in batchs else batchs['others']
        trajs = trajdata(label=mol,batch=b)
        dataset.update(trajs)

    bonds = ['C-C']
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    for bd in B:
        # print(len(B[bd]))
        B[bd] =np.array(B[bd])/1.1

    train(Bp,D,B,E,step=step,fitobj=obj,level=level)

   
if __name__ == '__main__':
   ''' Run with commond: ./gpfit.py fit --o=BE --s=3000 '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [fit])
   argh.dispatch(parser)  



